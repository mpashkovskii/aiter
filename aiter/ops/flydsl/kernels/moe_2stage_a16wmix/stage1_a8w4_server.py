# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Server wire-in for the iter-17 FlyDSL a8w4 MoE stage-1 GEMM.

The production W4A16 path runs GEMM1 (gate+up + clamped-silu) as the triton
``moe_gemm_a16w4`` and hands stage-2 a bf16 intermediate in the *aiter routing
module's* sorted-row order. This module replaces that triton GEMM1 in three
steps:

1. ``requant_hidden_fp8``: per-token amax-quantize the bf16 hidden states
   [n, model_dim] to e4m3fnuz (scale to +-240, ``requant_a_to_fp8``
   semantics). No gather: the a8w4 G1 kernel itself gathers token rows through
   the moe_sorting sorted ids (the hidden row is shared by the token's topk
   slots, so gathering inside the GEMM does the requant 1x instead of topk-x).
2. ``aiter.fused_moe.moe_sorting`` (block_size = kernel BM = 32) supplies
   seids / cumsum / stids; a tiny host index expression computes the aiter
   sorted output row per moe-sort position (orow = scatter_idx[token*topk +
   slot], -1 for padding), which makes the kernel's epilog a masked bf16
   store directly in the production intermediate layout -- stage-2 is
   untouched.
3. :func:`flydsl_a8w4_gemm1` with the candidate tile config.

In-server overhead vs the plain kernel: one ~n x model_dim x 3-byte
requant pass (~0.06-0.09 ms at n=16384, K=5120 at read/copy-peak rates) plus
moe_sorting and the orow index triple (a few tiny elementwise ops).
"""

import torch
import triton
import triton.language as tl

from aiter.fused_moe import moe_sorting

from . import flydsl_a8w4_gemm1

__all__ = ["requant_hidden_fp8", "moe_stage1_a8w4_forward"]


@triton.jit
def _requant_hidden_fp8_kernel(
    x_ptr,  # bf16 [n, K] row-major
    a8_ptr,  # fp8 fnuz [n, K] out
    s_a_ptr,  # fp32 [n] out
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    kmask = offs < K
    x = tl.load(x_ptr + r * K + offs, mask=kmask, other=0.0).to(tl.float32)
    # requant_a_to_fp8 semantics: per-row amax clamped to >=1e-12, scale to
    # +-240 (this ROCm torch build's fp8fnuz cast overflows to NaN above
    # ~248; the fp32 scale loses no dynamic range).
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    s_a = amax / 240.0
    y = (x / s_a).to(tl.float8e4b8)  # e4m3fnuz on gfx942 (bitwise == torch cast)
    tl.store(a8_ptr + r * K + offs, y, mask=kmask)
    tl.store(s_a_ptr + r, s_a)


def requant_hidden_fp8(hidden, out=None):
    """bf16 [n, K] hidden -> (fp8 fnuz [n, K], s_a [n+1] fp32 (+1 pad row)).

    The +1 s_a pad row makes the kernel's sentinel-row scale load (token == M
    == n) defined garbage instead of an OOB read; those rows are masked in
    the epilog.
    """
    n, k = hidden.shape
    assert hidden.dtype == torch.bfloat16 and hidden.stride(1) == 1
    dev = hidden.device
    if out is None:
        a8 = torch.empty((n, k), dtype=torch.float8_e4m3fnuz, device=dev)
        s_a = torch.empty((n + 1,), dtype=torch.float32, device=dev)
    else:
        a8, s_a = out
    BLOCK_K = max(triton.next_power_of_2(k), 64)
    _requant_hidden_fp8_kernel[(n,)](
        hidden, a8, s_a, k, BLOCK_K=BLOCK_K, num_warps=8
    )
    return a8, s_a


def sorted_output_rows(stids, scatter_idx, n_tokens, topk):
    """aiter sorted output row per moe-sort position, -1 for padding rows.

    stids (i32 [P]): moe_sorting sorted ids (topk_slot << 24) | token.
    scatter_idx (int [n*topk]): canonical (token*topk + slot) -> aiter row.
    Rows whose token/slot are padding (sentinel token == n or junk) get -1;
    the kernel masks their stores.
    """
    fused = stids.to(torch.int64)
    token = fused & 0x00FFFFFF
    slot = (fused >> 24) & 0xFF
    valid = (token < n_tokens) & (slot < topk)
    canon = (token * topk + slot).clamp_(0, n_tokens * topk - 1)
    aiter_row = scatter_idx.to(torch.int64).index_select(0, canon)
    return torch.where(valid, aiter_row, -1).to(torch.int32).contiguous()


def moe_stage1_a8w4_forward(
    hidden,  # bf16 [n, model_dim] routed MoE input (post layernorm)
    topk_ids,  # [n, topk] canonical expert ids (routing_data.topk_ids)
    canon_gammas,  # fp32 [n, topk] canonical routing weights (moe_sorting inputs;
    # only its sorting is used here -- G1 applies no routing weight)
    scatter_idx,  # int [n*topk]: per canon position (token*topk+slot), its aiter row
    aiter_rows,  # number of aiter-sorted rows in the intermediate buffer
    w8_preshuffled,  # from requant_w2_to_fp8col(W1, pair_k0=True), cached per tensor
    w1_col_scale,  # fp32 (E, 2*inter) per-column scale, cached
    *,
    num_experts,
    inter_dim,
    topk,
    swiglu_limit=10.0,
    inter_sorted_bf16=None,  # optional preallocated [aiter_rows, inter_dim]
    tile_m=32,
    tile_n=192,
    tile_k=64,
    waves_per_eu=1,
    b_nt=0,
    a_stages=2,
    b_v16=1,
    xcd_swizzle=2,
    act="silu_clamp",
):
    """Full stage-1: A requant + moe_sorting + iter-17 a8w4 GEMM1.

    Returns the bf16 [aiter_rows, inter_dim] intermediate in the production
    aiter-routing-module sorted-row order, activation applied (silu_clamp at
    swiglu_limit), ready for the unchanged stage-2.
    """
    dev = hidden.device
    n = topk_ids.shape[0]

    a8, s_a = requant_hidden_fp8(hidden)

    stids, sweights, seids, nvid, mbuf = moe_sorting(
        topk_ids.to(torch.int32),
        canon_gammas.view(n, topk).float(),
        num_experts,
        inter_dim,
        torch.bfloat16,
        block_size=tile_m,
        accumulate=True,
    )

    orow = sorted_output_rows(stids, scatter_idx, n, topk)

    out = inter_sorted_bf16
    if out is None:
        out = torch.empty((aiter_rows, inter_dim), dtype=torch.bfloat16, device=dev)

    flydsl_a8w4_gemm1(
        a8_fp8=a8,
        s_a=s_a,
        w8_preshuffled=w8_preshuffled,
        w1_col_scale=w1_col_scale,
        sorted_expert_ids=seids,
        cumsum_tensor=nvid.to(torch.int32),
        sorted_token_ids=stids,
        sorted_out_rows=orow,
        inter_sorted_bf16=out,
        n_tokens=n,
        NE=num_experts,
        KLEN=hidden.shape[1],
        INTER=inter_dim,
        topk=topk,
        swiglu_limit=swiglu_limit,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        a_stages=a_stages,
        b_v16=b_v16,
        xcd_swizzle=xcd_swizzle,
        act=act,
    )
    return out
