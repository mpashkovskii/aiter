# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL gfx950 FP8 gated-MLP prefill path: gate_up GEMM with a fused GeLU(tanh)*up + MX (1x32 E8M0) quantization
epilogue, and the MX down GEMM that consumes it.

Shapes are Gemma-4-E4B: gate_up N = 2 * 10240, K = 2560; down N = 2560, K = 10240 (a per-token x per-channel FP8
layer pair, bf16 output). M covers prefill chunks, including values that are not a multiple of 16 or 256.
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
    flydsl_gate_up_gelu_bf16,
    flydsl_gate_up_gelu_mx,
    flydsl_mx_down,
    prepare_gate_up_weight,
    scale_numel,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
FP8 = dtypes.fp8


def _unshuffle_scales(flat, rows, k):
    """Row-major [rows, k/32] E8M0 bytes from the ``shuffle_scale_w4`` layout (see ``gemm_a8w8_gelu_mx``)."""
    r = torch.arange(rows).view(-1, 1)
    j = torch.arange(k // 32).view(1, -1)
    kk = j // 4
    addr = (
        (((r // 32) * (k // 256) + kk // 2) * 64 + (j % 4) * 16 + r % 16) * 4
        + (kk % 2) * 2
        + (r // 16) % 2
    )
    return flat[addr]


def _make_gate_up(m, inter, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = 2 * inter
    w = (torch.randn(n, k, generator=g) * 0.7).to(FP8)
    w_scale = torch.rand(n, generator=g) * 0.004 + 0.002
    x = torch.randn(m, k, generator=g) * (torch.rand(m, 1, generator=g) * 3 + 0.2)  # distinct per-row magnitudes
    x_scale = x.abs().amax(1, keepdim=True) / 448
    xq = (x / x_scale).to(FP8)
    return xq, x_scale, w, w_scale


def run_torch_gate_up(xq, x_scale, w, w_scale):
    y = (xq.to(dtypes.fp32) * x_scale) @ (w.to(dtypes.fp32) * w_scale.view(-1, 1)).t()
    half = y.shape[1] // 2
    return F.gelu(y[:, :half], approximate="tanh") * y[:, half:]


@benchmark()
def test_gate_up_gelu_mx(m, inter, k):
    xq, x_scale, w, w_scale = _make_gate_up(m, inter, k)
    ref = run_torch_gate_up(xq, x_scale, w, w_scale)
    w_il, w_scale_il = prepare_gate_up_weight(w, w_scale)
    out = torch.full((m, inter), 0x7F, dtype=torch.uint8).view(FP8)  # NaN bytes: unwritten rows fail
    out_scale = torch.zeros(scale_numel(m, inter), dtype=torch.uint8)

    _, us = run_perftest(flydsl_gate_up_gelu_mx, xq, w_il, x_scale, w_scale_il, out, out_scale)

    sc = _unshuffle_scales(out_scale, m, inter).to(dtypes.fp32)
    deq = out.to(dtypes.fp32).view(m, inter // 32, 32) * torch.pow(2.0, sc - 127).unsqueeze(-1)
    amax = ref.view(m, inter // 32, 32).abs().amax(-1, keepdim=True)
    # per-element error against the fp32 reference, relative to its 32-column group maximum: e4m3 rounds to 2^-4 of
    # the largest magnitude of the group (scale rounded up to a power of two) plus the bf16 rounding of the GEMM
    err_el = ((deq - ref.view(m, inter // 32, 32)).abs() / amax.clamp_min(1e-6)).amax().item()
    assert err_el < 0.09, f"max per-element error {err_el:.3f} of the group maximum"
    err = checkAllclose(ref, deq.view(m, inter), rtol=0.1, atol=0.02 * ref.abs().max().item(), msg="gate_up_gelu_mx")
    flops = 2 * m * 2 * inter * k
    nbytes = m * k + 2 * inter * k + m * inter + m * inter // 32
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "flydsl max group-rel err": err_el,
    }


@benchmark()
def test_gate_up_gelu_bf16(m, inter, k):
    """bf16 epilogue (down_proj stays bf16): out = gelu_tanh(gate) * up, no quantization."""
    xq, x_scale, w, w_scale = _make_gate_up(m, inter, k)
    ref = run_torch_gate_up(xq, x_scale, w, w_scale)
    w_il, w_scale_il = prepare_gate_up_weight(w, w_scale)
    out = torch.full((m, inter), float("nan"), dtype=dtypes.bf16)  # NaN: unwritten rows fail

    _, us = run_perftest(flydsl_gate_up_gelu_bf16, xq, w_il, x_scale, w_scale_il, out)

    err = checkAllclose(ref, out.to(dtypes.fp32), rtol=2e-2, atol=5e-3 * ref.abs().max().item(), msg="gate_up_gelu_bf16")
    rel_rms = ((out.to(dtypes.fp32) - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rel_rms < 4e-3, f"relative RMS error {rel_rms:.4f}"
    flops = 2 * m * 2 * inter * k
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl err": err,
        "flydsl rel rms err": rel_rms,
    }


@benchmark()
def test_mx_down(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(n, k, generator=g) * 0.7).to(FP8)
    w_scale = torch.rand(n, generator=g) * 0.004 + 0.002
    a = torch.randn(m, k, generator=g).to(FP8)
    e8 = torch.randint(122, 132, (m, k // 32), generator=g, dtype=torch.uint8, device="cuda")  # distinct per-row/group
    ref = (
        (a.to(dtypes.fp32).view(m, k // 32, 32) * torch.pow(2.0, e8.to(dtypes.fp32) - 127).unsqueeze(-1)).view(m, k)
        @ (w.to(dtypes.fp32) * w_scale.view(-1, 1)).t()
    )
    # the kernel wants the activation scales in the shuffle_scale_w4 layout
    r = torch.arange(m).view(-1, 1)
    j = torch.arange(k // 32).view(1, -1)
    kk = j // 4
    addr = (
        (((r // 32) * (k // 256) + kk // 2) * 64 + (j % 4) * 16 + r % 16) * 4
        + (kk % 2) * 2
        + (r // 16) % 2
    )
    flat = torch.zeros(scale_numel(m, k), dtype=torch.uint8)
    flat[addr] = e8
    unit = torch.full((scale_numel(n, k),), 127, dtype=torch.uint8)
    w_sh = shuffle_weight(w.contiguous(), layout=(16, 16))
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16)

    _, us = run_perftest(flydsl_mx_down, a, flat, w_sh, unit, w_scale, out)

    err = checkAllclose(ref, out.to(dtypes.fp32), rtol=2e-2, atol=2e-2 * ref.abs().amax().item(), msg="mx_down")
    assert err == 0, f"{err:.2%} of elements outside rtol/atol"
    row_err = ((out.to(dtypes.fp32) - ref).pow(2).mean(1).sqrt() / ref.pow(2).mean(1).sqrt()).amax().item()
    assert row_err < 0.02, f"worst row relative RMS error {row_err:.3f}"
    flops = 2 * m * n * k
    nbytes = m * k + n * k + m * n * 2 + m * k // 32
    return {
        "gfx": get_gfx_runtime(),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "flydsl worst row rms err": row_err,
    }


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("gelu_mx GEMMs unsupported on %s; skipping", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="config input of test")
    parser.add_argument("-m", "--tokens", type=int, nargs="*", default=[1024, 2500, 4097, 8192, 16384], help="M values")
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
        rows_gu.append(test_gate_up_gelu_mx(m, inter, hidden))
        rows_gu.append(test_gate_up_gelu_bf16(m, inter, hidden))
        rows_dn.append(test_mx_down(m, hidden, inter))
    aiter.logger.info("gate_up + gelu*up + MX quant summary:\n%s", pd.DataFrame(rows_gu).to_string(index=False))
    aiter.logger.info("MX down summary:\n%s", pd.DataFrame(rows_dn).to_string(index=False))


if __name__ == "__main__":
    main()
