# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-size MX (1x32 E8M0 activation scales) FP8 down projection with a per-channel fp32 weight scale, on the
strided-batched mxscale kernel (``kernels/bmm_mxfp8_chan_gfx950.py``) at batch 1: ``Out[m, n] = bf16(sum_k XQ[m, k] *
2**(x_scale[m, k // 32] - 127) * WQ[n, k] * chan[n])`` with the weight ``WQ`` 16x16 pre-shuffled and the weight E8M0 scales
all 2**0. Split-K runs in one launch (the last arriving split reduces)."""

from __future__ import annotations

import functools
import inspect

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from torch import Tensor

from aiter.utility.graph_alloc import persistent_alloc

from .batched_gemm_a8w8_gfx950 import _launch_config
from .kernels.bmm_mxfp8_chan_gfx950 import launch_bmm_mxfp8_chan
from .kernels.kernels_common import ceildiv
from .kernels.tensor_shim import _compile_and_run, ptr_arg

_SPLIT_COUNTERS = 1 << 16
_RUNTIME_PARAMS = 10  # eight pointers, M and the stream
_COMPILED: dict[tuple, flyc.CompiledFunction] = {}


@functools.cache
def _split_counters(device: int, stream: int) -> Tensor:
    with persistent_alloc(torch.device("cuda", device)):
        return torch.zeros(_SPLIT_COUNTERS, dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=256)
def _constexprs(name: str, n: int, k: int) -> tuple:
    cfg = _launch_config(name)
    if cfg is None:
        raise ValueError(f"[FlyDSL mx down] unrecognised kernelName: {name!r}")
    bound = inspect.signature(launch_bmm_mxfp8_chan.func).bind(
        *(None,) * _RUNTIME_PARAMS, n, k, 1, **cfg, x_scale_k=32, w_scale_n=32, w_scale_k=32, x_scale_transposed=False
    )
    bound.apply_defaults()
    return bound.args[_RUNTIME_PARAMS:]


def flydsl_mx_down_decode(
    XQ: Tensor, x_scale: Tensor, W: Tensor, w_unit_scale: Tensor, w_chan_scale: Tensor, out: Tensor, kernel_name: str
) -> Tensor:
    """XQ [M, K] fp8; x_scale [M, K // 32] uint8 row-major E8M0; W [N, K] 16x16 pre-shuffled fp8; w_unit_scale
    [N // 32, K // 32] uint8 (all 127); w_chan_scale [N] fp32; out [M, N] bf16; kernel_name a ``flydsl_bmm_mxfp8_mfma_...``
    config name."""
    m, k = XQ.shape
    n = W.shape[0]
    cfg = _launch_config(kernel_name)
    stream = torch.cuda.current_stream(XQ.device)
    partials = counters = out
    if cfg["splits"] > 1:
        tiles = ceildiv(m, cfg["tile_m"]) * (n // cfg["tile_n"])
        partials = torch.empty(tiles * cfg["splits"] * cfg["tile_m"] * cfg["tile_n"], dtype=torch.float32, device=XQ.device)
        counters = _split_counters(XQ.device.index, stream.cuda_stream)
    tensors = (out, XQ, W, x_scale, w_unit_scale, partials, counters, w_chan_scale)
    constexprs = _constexprs(kernel_name, n, k)
    args = (*map(ptr_arg, tensors), m, fx.Stream(stream), *constexprs)
    compiled = _COMPILED.get(constexprs)
    if compiled is not None:
        compiled(*args)
        return out
    compiled = _compile_and_run(launch_bmm_mxfp8_chan, *args)
    if compiled is not None:
        _COMPILED[constexprs] = compiled
    return out
