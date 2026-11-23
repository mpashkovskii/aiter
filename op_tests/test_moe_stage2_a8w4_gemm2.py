# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# iter-14 wire-in test: the FlyDSL a8w4 MoE stage-2 pipeline
# (moe_sorting + fused gather/requant + flydsl_a8w4_gemm2, DeepSeek-V4.1-Flash
# W4A16 shapes, TP4 ranks: E=384, topk=6, model_dim=5120, K_TRUE=576 at K_PAD=768)
# against the production tuned-triton `moe_gemm_a16w4` GEMM2 path, both checked
# against the fp32 torch reference `torch_moe_stage2`.
#
# Faithfulness to the model call:
#  * routing metadata comes from the real `aiter.ops.triton.moe_routing`
#    (score_mode=sqrtsoftplus, bias, renorm, routed_scaling_factor=1.5), which
#    returns topk_ids int16 / scatter uint16 and leaves moe_sorting's tail
#    [num_valid, P) uninitialised -- the exact in-server condition that drove
#    the old token<M-only gather mask out of bounds (see stage2_a8w4_server.py).
#  * the stage-1 intermediate is laid out in aiter-routing *sorted* order at
#    padded width K_PAD (model GEMM1 output layout), rebuilt from a canonical
#    (token, slot) tensor through scatter_idx.
#  * use_gamma=True/False reproduces the two production modes (router weight
#    applied at combine vs folded into the GEMM1 input).

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import QuantType, dtypes
from aiter.fused_moe import torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    moe_stage2_a8w4_forward,
    requant_w2_to_fp8col,
)
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
from aiter.ops.triton.moe.moe_routing import routing as _routing_mod
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]  # the a8w4 kernel + tuned configs are gfx942-validated

E, TOPK, HID = 384, 6, 5120
K_PAD = 768
K_EFF = 576


def _build_inputs(n, topk, k_true, use_gamma, seed):
    """Real aiter routing + fabricated production-layout weights and inter.

    Returns (inter_sorted, rd_pack, w2_triton, s2_triton, w8, s_w, ref_pack).
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gating = torch.randn((n, E), dtype=torch.float32, generator=gen)
    bias = torch.randn((E,), dtype=torch.float32, generator=gen) * 0.01
    routing_data, gather_idx, scatter_idx = _routing_mod.routing(
        gating,
        topk,
        score_mode="sqrtsoftplus",
        bias=bias,
        renorm=True,
        routed_scaling_factor=1.5,
        use_grouped_topk=False,
    )
    rows = gather_idx.numel()
    assert rows == n * topk, (rows, n, topk)

    # production weights: bf16 -> per_1x32 fp4 pack (triton layout)
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    wb = torch.randn((E, HID, K_PAD), dtype=torch.bfloat16, generator=gen) * K_PAD**-0.5
    wb[:, :, k_true:] = 0.0
    wq, ws = tq(wb, quant_dtype=aiter.dtypes.fp4x2)
    wb = None
    w2_packed = wq.view(torch.uint8).view(E, HID, K_PAD // 2).contiguous()
    w2_scale = ws.view(E, HID, K_PAD // 32).contiguous()
    w2_triton = w2_packed.transpose(1, 2)  # logical (E, K/2, N), stride(-2)==1
    s2_triton = w2_scale.view(torch.uint8).permute(0, 2, 1).contiguous()
    w8, s_w = requant_w2_to_fp8col(w2_triton, s2_triton, k_true, pair_k0=True)

    # canonical inter; for use_gamma=False the router weights are folded in,
    # as the production apply_router_weight_on_input path does before GEMM1.
    gate_canon = routing_data.gate_scal.float()[scatter_idx.long()]  # (n*topk,)
    canon_inter = torch.randn((n * topk, K_PAD), dtype=torch.bfloat16, generator=gen)
    if not use_gamma:
        canon_inter = (canon_inter.float() * gate_canon[:, None]).to(torch.bfloat16)
    inter_sorted = torch.empty((rows, K_PAD), dtype=torch.bfloat16)
    inter_sorted[scatter_idx.long()] = canon_inter

    topk_ids_i16 = routing_data.topk_ids.to(torch.int16)
    in_pack = dict(
        inter_sorted=inter_sorted,
        topk_ids=topk_ids_i16,
        gate_scal=routing_data.gate_scal,
        scatter_idx=scatter_idx,
        gather_idx=gather_idx,
        rd=routing_data,
    )
    ref_pack = dict(
        canon_inter=canon_inter,
        gate_canon=gate_canon,
        wq=wq,
        w2_scale=w2_scale,
        topk_ids_i32=topk_ids_i16.to(torch.int32),
        n=n,
        topk=topk,
    )
    return in_pack, w2_triton, s2_triton, w8, s_w, ref_pack


def run_torch(ref_pack, use_gamma, dtype=dtypes.bf16):
    """fp32 torch reference via aiter's torch_moe_stage2 (not timed)."""
    E_, topk = E, ref_pack["topk"]
    n = ref_pack["n"]
    w1_stub = torch.zeros((E_, 2 * K_PAD, HID // 2), dtype=torch.uint8)
    topk_w = (
        ref_pack["gate_canon"].view(n, topk)
        if use_gamma
        else torch.ones((n, topk), dtype=torch.float32)
    )
    return torch_moe_stage2(
        ref_pack["canon_inter"].view(n, topk, K_PAD),
        w1_stub.view(dtypes.fp4x2),
        ref_pack["wq"].view(dtypes.fp4x2),
        topk_w,
        ref_pack["topk_ids_i32"],
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=ref_pack["w2_scale"],
        a2_scale=None,
        doweight=True,
    )


@benchmark()
def test_g2(n, topk, k_true, use_gamma):
    use_gamma = bool(use_gamma)
    in_pack, w2_triton, s2_triton, w8, s_w, ref_pack = _build_inputs(
        n, topk, k_true, use_gamma, seed=1000 + n + int(use_gamma)
    )
    ref = run_torch(ref_pack, use_gamma).to(dtypes.fp32)
    del ref_pack["canon_inter"]
    torch.cuda.empty_cache()

    candidates = {
        "triton": lambda: moe_gemm_a16w4(
            in_pack["inter_sorted"],
            w2_triton,
            None,
            s2_triton,
            None,
            None,
            None,
            in_pack["rd"],
            scatter_indx=in_pack["scatter_idx"],
            gammas=in_pack["gate_scal"] if use_gamma else None,
            expert_map=None,
            gate_valid=None,
            backend="triton",
        ),
        "flydsl_a8w4": lambda: moe_stage2_a8w4_forward(
            in_pack["inter_sorted"],
            in_pack["topk_ids"],
            in_pack["gate_scal"],
            in_pack["scatter_idx"],
            w8,
            s_w,
            num_experts=E,
            model_dim=HID,
            topk=topk,
            k_true=k_true,
            use_gamma=use_gamma,
        ),
    }

    rows = n * topk
    flops = 2 * rows * HID * k_true
    # per-candidate traffic: inter read + weights + output write
    bytes_w = {
        "triton": E * HID * k_true / 2 + E * HID * (k_true // 32),
        "flydsl_a8w4": E * HID * k_true + E * HID * 4 // 1,
    }
    # cos_diff tolerance: fp8 A-requant requantises every activation to e4m3,
    # observed worst-case on production c128 routing dumps is 3.55e-4; the
    # production triton path stays <=1e-4. A broken gather (wrong rows/slots,
    # the in-server mask defect of iter-14's first wire-in) shows >=1e-1.
    cos_tol = {"triton": 1e-4, "flydsl_a8w4": 1.5e-3}

    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        y = out.to(dtypes.fp32)
        cos = 1 - 2 * (y * ref).sum() / ((y * y).sum() + (ref * ref).sum())
        cos = float(cos)
        assert cos <= cos_tol[name], f"{name}: cos_diff {cos:.3e} > {cos_tol[name]:.1e}"
        nbytes = rows * K_PAD * 2 + bytes_w[name] + n * HID * 2
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = cos
    del ref
    torch.cuda.empty_cache()
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "a8w4 stage-2 wire-in unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="config input of test"
    )
    parser.add_argument(
        "-s",
        "--snt",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1,),
            (3,),
            (64,),
            (1024,),
            (4096,),
            (16383,),
            (16384,),
        ],
        help="""list of (n,) token counts (topk/k_true fixed at model constants).
-s 64 1024 16384""",
    )
    parser.add_argument(
        "-g",
        "--use_gamma",
        type=int,
        nargs="*",
        default=[1, 0],
        help="list of use_gamma flags (1: router weight at combine; 0: folded into GEMM1 input)",
    )
    args = parser.parse_args()

    df = []
    for snt, ug in itertools.product(args.snt, args.use_gamma):
        if not isinstance(snt, tuple):
            snt = (snt,)
        n = snt[0]
        topk_ = snt[1] if len(snt) > 1 else TOPK
        k_true_ = snt[2] if len(snt) > 2 else K_EFF
        df.append(test_g2(n, topk_, k_true_, ug))
    df = pd.DataFrame(df)
    try:
        table = df.to_markdown(index=False)
    except ImportError:  # minimal env: tabulate may be absent
        table = df.to_string(index=False)
    aiter.logger.info(
        "a8w4 stage-2 wire-in summary (markdown):\n%s", table
    )


if __name__ == "__main__":
    main()
