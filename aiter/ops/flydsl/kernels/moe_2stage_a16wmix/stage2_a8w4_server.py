# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Server wire-in for the iter-10 FlyDSL a8w4 MoE stage-2 GEMM (iter-14).

The production W4A16 (mxfp4, a16w4 triton) path runs GEMM1+swiglu and GEMM2
from ``moe_op_gemm_a16w4``, with the stage-1 intermediate in the *aiter routing
module's* sorted-row order. The iter-10 ``gemm2_a8w4`` kernel consumes its A
rows in the ``moe_sorting`` order (seids per BM-block). This module bridges the
two in three steps:

1. ``gather_requant_fp8``: one fused pass -- for every moe-sorted row, look up
   the aiter-sorted source row (through ``scatter_idx``, whose values are the
   aiter sorted row for each canonical (token*topk+slot) position),
   read the bf16 intermediate, per-row amax-quantize to e4m3fnuz fp8 (scale to
   +-240, exactly ``requant_a_to_fp8`` semantics), and emit the combined
   routing weight ``cw = gamma * s_a`` (or ``s_a`` alone when the router
   weight was already applied on the GEMM1 input).
2. ``aiter.fused_moe.moe_sorting`` (block_size = kernel BM = 32) supplies
   seids / cumsum / stids / sweights and the output buffer.
3. :func:`flydsl_a8w4_gemm2` from iter-10 with the kept best config
   bm32/tn256/tk64/xcd2/bnt0/wpe1/as2/b_v16.

In-server overhead vs the plain kernel: one ~170 MB-traffic requant pass
(~0.05 ms/layer) plus moe_sorting and two tiny index/scatter ops -- the
"A-fp8 requant" the iter-14 task prices at 0.05-0.08 ms/layer, left standalone
(fusable into the GEMM1 epilogue later, not required to clear the gate).
"""

import os

import torch
import triton
import triton.language as tl

from aiter.fused_moe import moe_sorting

from . import flydsl_a8w4_gemm2

__all__ = ["gather_requant_fp8", "moe_stage2_a8w4_forward"]

_DEBUG = os.getenv("AITER_ITER14_G2_DEBUG", "0") == "1"
_debug_seen: set = set()


@triton.jit
def _gather_requant_fp8_kernel(
    inter_ptr,  # bf16 [rows_aiter, k_stride], row-major, valid cols [0, K_TRUE)
    stids_ptr,  # i32 [P]: moe_sorting sorted ids (topk_slot << 24) | token
    inv_ptr,  # i32 [n*topk]: canonical row -> aiter sorted row
    sweights_ptr,  # fp32 [P]: moe_sorting sorted gammas (unused when not USE_GAMMA)
    a8_ptr,  # fp8 fnuz [P, K_TRUE] out
    s_a_ptr,  # fp32 [P] out
    cw_ptr,  # fp32 [P] out
    M,
    TOPK,
    K_TRUE,
    k_stride,
    P,
    USE_GAMMA: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    kmask = offs < K_TRUE
    stid = tl.load(stids_ptr + r)
    token = stid & 0x00FFFFFF
    slot = (stid >> 24) & 0xFF
    # moe_sorting only fills [0, nvid0) (real routes + (M, topk) sentinels);
    # the tail up to P is uninitialized torch.empty memory whose junk decodes
    # to token < M with arbitrary slot bits (seen in-server: 5503/6432 junk
    # tail entries masked by token < M alone). Without the slot check,
    # canon = token*TOPK + slot reads inv_ptr past n*topk -> IMA at the
    # scatter buffer's segment boundary (the server's HSA memory fault).
    valid = (token < M) & (slot < TOPK)
    canon = token * TOPK + slot
    arow = tl.load(inv_ptr + canon, mask=valid, other=0).to(tl.int64)
    x = tl.load(
        inter_ptr + arow * k_stride + offs, mask=kmask & valid, other=0.0
    ).to(tl.float32)
    # requant_a_to_fp8 semantics: per-row amax over the true K cols, amax
    # clamped to >=1e-12, scale to +-240 (this ROCm torch build's fp8fnuz cast
    # overflows to NaN above ~248; 240 stays in range and the fp32 scale loses
    # no dynamic range).
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    s_a = amax / 240.0
    y = (x / s_a).to(tl.float8e4b8)  # e4m3fnuz on gfx942 (bitwise == torch cast)
    tl.store(a8_ptr + r * K_TRUE + offs, y, mask=kmask)
    tl.store(s_a_ptr + r, s_a)
    if USE_GAMMA:
        g = tl.load(sweights_ptr + r)
        tl.store(cw_ptr + r, g * s_a)
    else:
        tl.store(cw_ptr + r, s_a)


def gather_requant_fp8(
    intermediate,  # bf16 [rows_aiter, k_cols] stage-1 output, aiter-routing order
    stids,  # i32 [P] moe_sorting sorted ids
    scatter_idx,  # int [n*topk]: per canonical position, its aiter sorted row
    sweights,  # fp32 [P] moe_sorting sorted gammas
    n_tokens,
    topk,
    k_true,
    use_gamma=True,
    out=None,  # (a8, s_a, cw) triple, optional reuse
):
    """Gather+requant the GEMM1 intermediate into moe-sort order fp8."""
    P = stids.numel()
    dev = intermediate.device
    if out is None:
        a8 = torch.empty((P, k_true), dtype=torch.float8_e4m3fnuz, device=dev)
        s_a = torch.empty((P,), dtype=torch.float32, device=dev)
        cw = torch.empty((P,), dtype=torch.float32, device=dev)
    else:
        a8, s_a, cw = out
    assert intermediate.dtype == torch.bfloat16
    assert intermediate.stride(1) == 1
    BLOCK_K = max(triton.next_power_of_2(k_true), 64)
    _gather_requant_fp8_kernel[(P,)](
        intermediate,
        stids,
        scatter_idx,
        sweights,
        a8,
        s_a,
        cw,
        n_tokens,
        topk,
        k_true,
        intermediate.stride(0),
        P,
        USE_GAMMA=use_gamma,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return a8, s_a, cw


def moe_stage2_a8w4_forward(
    intermediate,  # bf16 [rows_aiter, k_cols] GEMM1 output (aiter-routing order)
    topk_ids,  # [n, topk] canonical expert ids (routing_data.topk_ids)
    gate_scal,  # fp32 [rows_aiter] per-sorted-row routing weights (gate_scal)
    scatter_idx,  # int [n*topk]: per canon position (token*topk+slot), its aiter row
    w8_preshuffled,  # from requant_w2_to_fp8col(pair_k0=True), cached per weight tensor
    w2_col_scale,  # fp32 (E, N) per-column scale, cached
    *,
    num_experts,
    model_dim,
    topk,
    k_true,
    use_gamma=True,  # False when apply_router_weight_on_input (gammas already applied)
    tile_m=32,
    tile_n=256,
    tile_k=64,
    waves_per_eu=1,
    b_nt=0,
    a_stages=2,
    b_v16=1,
    xcd_swizzle=2,
):
    """Full stage-2: moe_sorting + fused gather/requant + iter-10 a8w4 GEMM2.

    Returns the bf16 [n_tokens, model_dim] MoE output (routing-weighted atomic
    accumulation inside the kernel).
    """
    dev = intermediate.device
    n = topk_ids.shape[0]

    # Host-side shape sanity: a malformed signature previously crashed the GPU
    # with an 6.3M-program _gather_requant_fp8_kernel launch (IMA).
    rows = intermediate.shape[0]
    if _DEBUG:
        key = (topk_ids.shape, rows, tuple(scatter_idx.shape),
               tuple(gate_scal.shape), str(scatter_idx.dtype))
        if key not in _debug_seen:
            _debug_seen.add(key)
            print(
                f"[iter14-g2-debug] n={n} topk_ids={tuple(topk_ids.shape)}"
                f" {topk_ids.dtype} rows={rows} scatter={tuple(scatter_idx.shape)}"
                f" {scatter_idx.dtype} gate_scal={tuple(gate_scal.shape)}"
                f" topk={topk} E={num_experts}",
                flush=True,
            )

    # canonical (token, slot)-ordered gammas for moe_sorting's weight input:
    # canon_g[j] = gate_scal[scatter_idx[j]] (scatter_idx values are sorted rows)
    canon_g = gate_scal.float().index_select(0, scatter_idx.long())

    stids, sweights, seids, nvid, mbuf = moe_sorting(
        topk_ids.to(torch.int32),
        canon_g.view(n, topk),
        num_experts,
        model_dim,
        torch.bfloat16,
        block_size=tile_m,
        accumulate=True,
    )
    P = stids.numel()
    _capturing = torch.cuda.is_current_stream_capturing()
    if _DEBUG:
        print(
            f"[iter14-g2-debug] sorted: P={P} seids={tuple(seids.shape)}"
            f" mbuf={tuple(mbuf.shape)} capturing={_capturing}",
            flush=True,
        )
        if not _capturing:
            torch.cuda.synchronize()
            smax = (
                int(scatter_idx.to(torch.int64).max().item())
                if scatter_idx.numel()
                else -1
            )
            n0 = int(nvid[0].item())
            print(
                f"[iter14-g2-debug] nvid0={n0} scatter_max={smax}",
                flush=True,
            )
            assert P < (1 << 21), f"absurd sorted size P={P}"
            assert smax < rows, f"scatter value {smax} >= intermediate rows {rows}"

    a8, s_a, cw = gather_requant_fp8(
        intermediate,
        stids,
        scatter_idx,
        sweights,
        n,
        topk,
        k_true,
        use_gamma=use_gamma,
    )

    out = mbuf
    out.zero_()  # the atomic epilog accumulates; belt-and-braces zero (~0.03 ms)
    flydsl_a8w4_gemm2(
        inter_fp8=a8,
        w8_preshuffled=w8_preshuffled,
        w2_col_scale=w2_col_scale.view(-1).contiguous(),
        sorted_expert_ids=seids,
        cumsum_tensor=nvid.to(torch.int32),
        sorted_token_ids=stids,
        combined_weights=cw,
        flat_out=out.view(-1),
        M_logical=n,
        max_sorted=a8.shape[0],
        NE=num_experts,
        D_HIDDEN=model_dim,
        KLEN=k_true,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        a_stages=a_stages,
        b_v16=b_v16,
        xcd_swizzle=xcd_swizzle,
    )
    if _DEBUG and not _capturing:
        torch.cuda.synchronize()
    return out
