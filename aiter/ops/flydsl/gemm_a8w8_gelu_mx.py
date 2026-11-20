# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 gated-MLP prefill path for gfx950: gate_up GEMM with a fused GeLU(tanh)*up + MX quantization epilogue, and the
matching MX (1x32 E8M0 activation scales) down GEMM with a per-channel FP8 weight scale.

``flydsl_gate_up_gelu_mx`` turns per-token FP8 activations and the merged gate_up weight into the FP8 hidden
activation ``[M, I]`` plus E8M0 scales ``[M, I/32]`` in one kernel; ``flydsl_mx_down`` consumes them. Weights go
through ``prepare_gate_up_weight`` / ``shuffle_weight`` once at load time.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

BLOCK_M = BLOCK_N = 256
BLOCK_K = 128
HALF_TILE = 128  # hidden units per N tile: 128 gate rows + 128 up rows
_I32_MAX = 2**31

_fx = None
_run_compiled = None


def _lazy_import() -> None:
    global _fx, _run_compiled
    if _fx is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.tensor_shim import _run_compiled as run_compiled

    _fx = fx_mod
    _run_compiled = run_compiled


@functools.lru_cache(maxsize=8)
def _interleave_index(n_half: int, device: str) -> Tensor:
    """Row permutation: tile ``t`` of the interleaved weight is [gate rows t*128.., up rows n_half + t*128..]."""
    nb = n_half // HALF_TILE
    dev = torch.device(device)
    return (
        torch.arange(2, device=dev).view(1, 2, 1) * n_half
        + torch.arange(nb, device=dev).view(nb, 1, 1) * HALF_TILE
        + torch.arange(HALF_TILE, device=dev).view(1, 1, HALF_TILE)
    ).reshape(-1)


def prepare_gate_up_weight(w: Tensor, w_scale: Tensor) -> tuple[Tensor, Tensor]:
    """``w`` ``[2*I, K]`` fp8 (gate rows then up rows), ``w_scale`` ``[2*I]`` fp32 -> (16x16 pre-shuffled and
    gate/up-interleaved weight, interleaved scale)."""
    from aiter.ops.shuffle import shuffle_weight

    n = w.shape[0]
    if n % (2 * HALF_TILE) != 0:
        raise ValueError(f"[FlyDSL gelu_mx] intermediate size must be a multiple of {HALF_TILE}, got {n // 2}")
    idx = _interleave_index(n // 2, str(w.device))
    return (
        shuffle_weight(w[idx].contiguous(), layout=(16, 16)),
        w_scale.reshape(-1)[idx].contiguous().to(torch.float32),
    )


def is_supported_shape(m: int, n: int, k: int) -> bool:
    return (
        n % (2 * HALF_TILE) == 0
        and k % BLOCK_K == 0
        and k >= 2 * BLOCK_K
        and m * (n // 2) < _I32_MAX
        and m * k < _I32_MAX
        and n * k < _I32_MAX
    )


@functools.lru_cache(maxsize=64)
def _compile_gate_up(k: int, waves_per_eu: int, xcd_swizzle: int):
    _lazy_import()
    from .kernels.gemm_a8w8_8wave_gelu_mx import compile_fp8_gemm_8w_gelu_mx

    return compile_fp8_gemm_8w_gelu_mx(
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        b_preshuffled=True,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
    )


@functools.lru_cache(maxsize=64)
def _compile_down(k: int, waves_per_eu: int, xcd_swizzle: int):
    _lazy_import()
    from .kernels.mxfp8_gemm_8wave_chan import compile_mxfp8_gemm_8w_chan

    return compile_mxfp8_gemm_8w_chan(
        K=k,
        b_preshuffled=True,
        scale_preshuffled=False,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
    )


def _i8(t: Tensor) -> Tensor:
    return t.view(torch.int8) if t.dtype != torch.int8 else t


def _u8(t: Tensor) -> Tensor:
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def flydsl_gate_up_gelu_mx(
    XQ: Tensor,
    W: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    out_scale: Tensor,
    *,
    waves_per_eu: int = 3,
    xcd_swizzle: int = 4,
) -> None:
    """XQ ``[M, K]`` fp8; W the ``prepare_gate_up_weight`` weight ``[N, K]``; x_scale per-token fp32 ``[M]`` / ``[M, 1]``;
    w_scale the interleaved fp32 ``[N]``. Writes ``out`` ``[M, N/2]`` fp8 and ``out_scale`` ``[M, N/64]`` uint8 (E8M0)."""
    _lazy_import()
    m, k = XQ.shape
    n = W.shape[0]
    if not is_supported_shape(m, n, k):
        raise ValueError(f"[FlyDSL gelu_mx] unsupported shape M={m} N={n} K={k}")
    if tuple(out.shape) != (m, n // 2) or tuple(out_scale.shape) != (m, n // 64):
        raise ValueError("[FlyDSL gelu_mx] output shapes must be [M, N/2] and [M, N/64]")
    exe = _compile_gate_up(k, waves_per_eu, xcd_swizzle)
    _run_compiled(
        exe,
        _i8(XQ.contiguous()).view(-1),
        _i8(W).view(-1),
        _i8(out).view(-1),
        x_scale.reshape(-1).contiguous().to(torch.float32),
        w_scale,
        _i8(out_scale).view(-1),
        m,
        n,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
    )


def flydsl_mx_down(
    XQ: Tensor,
    x_scale: Tensor,
    W: Tensor,
    w_unit_scale: Tensor,
    w_chan_scale: Tensor,
    out: Tensor,
    *,
    waves_per_eu: int = 3,
    xcd_swizzle: int = 4,
) -> None:
    """XQ ``[M, K]`` fp8 with E8M0 ``x_scale`` ``[M, K/32]``; W ``[N, K]`` 16x16 pre-shuffled fp8; ``w_unit_scale``
    ``[N, K/32]`` uint8 filled with 127 (2**0); ``w_chan_scale`` fp32 ``[N]``; out ``[M, N]`` bf16."""
    _lazy_import()
    m, k = XQ.shape
    n = W.shape[0]
    if k % 256 != 0 or m * n * 2 >= _I32_MAX or m * k >= _I32_MAX or n * k >= _I32_MAX:
        raise ValueError(f"[FlyDSL mx_down] unsupported shape M={m} N={n} K={k}")
    exe = _compile_down(k, waves_per_eu, xcd_swizzle)
    _run_compiled(
        exe,
        _u8(XQ.contiguous()).view(-1),
        _u8(W).view(-1),
        out.view(-1),
        _u8(x_scale).view(-1),
        _u8(w_unit_scale).view(-1),
        w_chan_scale,
        m,
        n,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
    )
