# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL gfx950 FP8 gated-MLP decode path (M <= 256): gate_up GEMM with a fused GeLU(tanh)*up + MX (1x32 E8M0) quantization
epilogue on 128 x 160-hidden tiles (``flydsl_gate_up_gelu_mx_dec``) and the weight-streaming MX down GEMM that consumes its
tile-layout activation and "mq"-layout scales (``flydsl_mx_down_ws``).

Shapes are Gemma-4-E4B: gate_up N = 2 * 10240, K = 2560; down N = 2560, K = 10240.
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_gelu_mx import (
    flydsl_gate_up_gelu_mx_dec,
    prepare_gate_up_weight_dec,
    scale_numel,
)
from aiter.ops.flydsl.gemm_mxfp8_down_ws import flydsl_mx_down_ws, x_tile_layout
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
FP8 = dtypes.fp8


def _tile_offsets(rows, k, device):
    """Byte offset in the MFMA tile layout of every (row, column) of an [rows, k] activation."""
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(k, device=device).view(1, -1)
    cs = c % 128
    return (
        ((r // 16) * (k // 128) + c // 128) * 2048
        + (cs // 64) * 1024
        + (((cs % 64) // 16) * 16 + r % 16) * 16
        + cs % 16
    )


def _mq_offsets(rows, k, device):
    """Byte offset in the "mq" scale layout of every (row, 32-column group) of an [rows, k] activation."""
    r = torch.arange(rows, device=device).view(-1, 1)
    g = torch.arange(k // 32, device=device).view(1, -1)
    return (((r // 64) * (k // 128) + g // 4) * 4 + g % 4) * 64 + (r % 16) * 4 + (r % 64) // 16


def _make_gate_up(m, inter, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = 2 * inter
    w = (torch.randn(n, k, generator=g) * 0.7).to(FP8)
    w_scale = torch.rand(n, generator=g) * 0.004 + 0.002
    x = torch.randn(m, k, generator=g) * (torch.rand(m, 1, generator=g) * 3 + 0.2)
    x_scale = x.abs().amax(1, keepdim=True) / 448
    xq = (x / x_scale).to(FP8)
    return xq, x_scale, w, w_scale


@benchmark()
def test_gate_up_gelu_mx_dec(m, inter, k):
    xq, x_scale, w, w_scale = _make_gate_up(m, inter, k)
    y = (xq.to(dtypes.fp32) * x_scale) @ (w.to(dtypes.fp32) * w_scale.view(-1, 1)).t()
    ref = F.gelu(y[:, :inter], approximate="tanh") * y[:, inter:]
    w_il, w_scale_il = prepare_gate_up_weight_dec(w, w_scale)
    out = torch.full((-(-m // 16) * 16 * inter,), 0x7F, dtype=torch.uint8).view(FP8)
    out_scale = torch.zeros(scale_numel(m, inter), dtype=torch.uint8)

    _, us = run_perftest(flydsl_gate_up_gelu_mx_dec, xq, w_il, x_scale, w_scale_il, out, out_scale)

    flat = out.view(torch.uint8)
    deq_q = flat[_tile_offsets(m, inter, flat.device)].view(FP8).to(dtypes.fp32)
    sc = out_scale[_mq_offsets(m, inter, flat.device)].to(dtypes.fp32)
    deq = deq_q.view(m, inter // 32, 32) * torch.pow(2.0, sc - 127).unsqueeze(-1)
    amax = ref.view(m, inter // 32, 32).abs().amax(-1, keepdim=True)
    err_el = ((deq - ref.view(m, inter // 32, 32)).abs() / amax.clamp_min(1e-6)).amax().item()
    assert err_el < 0.09, f"max per-element error {err_el:.3f} of the group maximum"
    err = checkAllclose(ref, deq.view(m, inter), rtol=0.1, atol=0.02 * ref.abs().max().item(), msg="gate_up_gelu_mx_dec")
    return {"gfx": get_gfx_runtime(), "flydsl us": us, "flydsl err": err, "flydsl max group-rel err": err_el}


@benchmark()
def test_mx_down_ws(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(n, k, generator=g) * 0.7).to(FP8)
    chan = torch.rand(n, generator=g) * 0.004 + 0.002
    a = torch.randn(m, k, generator=g).to(FP8)
    e8 = torch.randint(122, 132, (m, k // 32), generator=g, dtype=torch.uint8, device="cuda")
    ref = (
        (a.to(dtypes.fp32).view(m, k // 32, 32) * torch.pow(2.0, e8.to(dtypes.fp32) - 127).unsqueeze(-1)).view(m, k)
        @ (w.to(dtypes.fp32) * chan.view(-1, 1)).t()
    )
    xt = x_tile_layout(a)
    sc = torch.zeros(scale_numel(m, k), dtype=torch.uint8)
    sc[_mq_offsets(m, k, sc.device)] = e8
    w_sh = shuffle_weight(w.contiguous(), layout=(16, 16))
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16)

    _, us = run_perftest(flydsl_mx_down_ws, xt, sc, w_sh, chan, out)

    err = checkAllclose(ref, out.to(dtypes.fp32), rtol=2e-2, atol=2e-2 * ref.abs().amax().item(), msg="mx_down_ws")
    assert err == 0, f"{err:.2%} of elements outside rtol/atol"
    row_err = ((out.to(dtypes.fp32) - ref).pow(2).mean(1).sqrt() / ref.pow(2).mean(1).sqrt()).amax().item()
    assert row_err < 0.02, f"worst row relative RMS error {row_err:.3f}"
    return {"gfx": get_gfx_runtime(), "flydsl us": us, "flydsl err": err, "flydsl worst row rms err": row_err}


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("decode gelu_mx GEMMs unsupported on %s; skipping", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="config input of test")
    parser.add_argument("-m", "--tokens", type=int, nargs="*", default=[1, 33, 136, 200, 248, 256], help="M values")
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(10240, 2560, 10240)],
        help="(intermediate, hidden, intermediate): gate_up is [2*I, hidden], down is [hidden, I]",
    )
    args = parser.parse_args()

    rows_gu, rows_dn = [], []
    for m, (inter, hidden, _) in itertools.product(args.tokens, args.mnk):
        rows_gu.append(test_gate_up_gelu_mx_dec(m, inter, hidden))
        rows_dn.append(test_mx_down_ws(m, hidden, inter))
    aiter.logger.info("decode gate_up + gelu*up + MX quant summary:\n%s", pd.DataFrame(rows_gu).to_string(index=False))
    aiter.logger.info("decode MX down summary:\n%s", pd.DataFrame(rows_dn).to_string(index=False))


if __name__ == "__main__":
    main()
