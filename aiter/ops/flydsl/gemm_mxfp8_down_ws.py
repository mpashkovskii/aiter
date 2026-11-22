# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runner of the weight-streaming decode-size MX down kernel (``kernels/mxfp8_down_ws_gfx950.py``) and the torch helper that
puts an fp8 activation into its tile layout (tests; the decode gate_up epilogue writes the layout directly)."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from torch import Tensor

from .kernels.mxfp8_down_ws_gfx950 import K, K_STEPS, launch_mx_down_ws
from .kernels.tensor_shim import _compile_and_run, ptr_arg

_COMPILED: dict[tuple, flyc.CompiledFunction] = {}


def x_tile_layout(x: Tensor) -> Tensor:
    """fp8 ``[M, K]`` row-major -> the tile layout (rows padded to a multiple of 16), flat uint8."""
    m = x.shape[0]
    rt = -(-m // 16)
    xp = torch.zeros(rt * 16, K, dtype=torch.uint8, device=x.device)
    xp[:m] = x.view(torch.uint8)
    # [rt, r16, ks, half, g, within]
    t = xp.view(rt, 16, K_STEPS, 2, 4, 16)
    # block (rt, ks): lane = 16 g + r16, piece half: offset half * 1024 + lane * 16 + within
    return t.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


def flydsl_mx_down_ws(X: Tensor, xs: Tensor, W: Tensor, chan: Tensor, out: Tensor, *, prefetch_b: int = 3) -> Tensor:
    """X: fp8 tile layout (flat, ``ceil(M/16) * 16 * K`` bytes); xs: row-major E8M0 uint8 ``[>= M, K / 32]``; W: ``[N, K]`` 16x16
    pre-shuffled fp8; chan: fp32 ``[N]``; out: bf16 ``[M, N]``."""
    m, n = out.shape
    stream = torch.cuda.current_stream(out.device)
    tensors = (out, X, W, xs, chan)
    constexprs = (n, prefetch_b)
    args = (*map(ptr_arg, tensors), m, fx.Stream(stream), *constexprs)
    compiled = _COMPILED.get(constexprs)
    if compiled is not None:
        compiled(*args)
        return out
    compiled = _compile_and_run(launch_mx_down_ws, *args)
    if compiled is not None:
        _COMPILED[constexprs] = compiled
    return out
