# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a16w4/a16wi4/a16w16 (bf16 A x mxfp4/int4/bf16 W) 2-stage MoE kernels.

CDNA MFMA pipeline. bf16 A (no A-scale), W1/W2 upconverted to bf16 in-kernel,
non-scaled ``MFMA(16,16,32,bf16)``:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU/SiTUv2 -> bf16 intermediate
    ``[sorted_size, inter_dim]`` by sorted position (no requant/scale).
  - stage2 (:mod:`gemm2`): down-proj GEMM + routing-weighted atomic bf16 scatter
    to ``[tokens, model_dim]``.

Reuses the standard sorting/cumsum/m_indices contract and the
shuffle_weight+e8m0_shuffle W layout. Shared low-level helpers live in
:mod:`gemm1` (imported by :mod:`gemm2`); host-side launch glue is defined below.

Launch args are raw device pointers (``fx.Int64``); tensors passed as
``.data_ptr()``.
"""

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid
from .gemm2_a8w4 import compile_gemm2_a8w4, gemm2_a8w4_grid
from .gemm1_a8w4fp4 import compile_gemm1_a8w4fp4, gemm1_a8w4fp4_grid

__all__ = [
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "compile_gemm2_a8w4",
    "compile_gemm1_a8w4fp4",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "flydsl_a8w4_gemm2",
    "flydsl_a8w4fp4_gemm1",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
    "gemm2_a8w4_grid",
    "gemm1_a8w4fp4_grid",
    "requant_w2_to_fp8col",
    "repack_w13_pairwise_to_a16w4_shuffled",
    "repack_w13_pairwise_to_a8w4fp4",
    "requant_a_to_fp8",
    "gather_requant_fp8",
    "moe_stage2_a8w4_forward",
]


def requant_w2_to_fp8col(w2_u8, w2_scale, k_true, device=None, pair_k0=False):
    """mxfp4 W2 (production layout) -> e4m3fnuz bytes with per-column exponent shift.

    ``w2_u8``:    (E, N, K_pack) uint8, physically k-inner (the triton a16w4 layout
                  transposed: logical (E, K_pack, N) with stride(-2)==1 also works --
                  the physical (N, K_pack) 2D content is what is consumed).
    ``w2_scale``: (E, K_groups, N) e8m0/uint8 in the same physical orientation as the
                  triton path stores it: physical (E, K_groups, N), group-major.
    ``k_true``:   true inter columns; columns beyond are padding (skipped).

    Returns ``(w8_preshuffled, s_w)``:
      w8_preshuffled: (E, N, k_true) e4m3fnuz bytes in the kernel's
                      (N/16, K/32, klane=4, nlane=16, kpack=8) byte layout,
                      contiguous per expert.
      s_w:            (E, N) fp32 per-column scale = 2^(max_e8m0_group - 127).

    Lossless whenever the per-column e8m0 spread is <= ~11 octaves (measured 5 on
    the DeepSeek-V4.1-Flash checkpoint); asserts bit-exactness on the whole tensor.
    """
    import torch

    w = w2_u8  # logical (E, K_pack, N) or (E, N, K_pack) -- see docstring
    s = w2_scale
    if w.stride(-2) == 1:  # triton layout: logical (E, K_pack, N), physical (E, N, K_pack)
        w = w.transpose(1, 2).contiguous()  # -> physical (E, N, K_pack)
    else:
        w = w.contiguous()
    s = s.contiguous()  # (E, K_groups, N)
    E, N, KP = w.shape
    KT = k_true
    G_true = KT // 32
    dev = w.device

    lo = (w & 0x0F).to(torch.int32)
    hi = (w >> 4).to(torch.int32)
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
    lo_v = torch.where((lo & 8) != 0, -e2m1[(lo & 7).long()], e2m1[(lo & 7).long()])
    hi_v = torch.where((hi & 8) != 0, -e2m1[(hi & 7).long()], e2m1[(hi & 7).long()])
    mag = torch.stack([lo_v, hi_v], dim=-1).view(E, N, KP * 2)[:, :, :KT]  # (E, N, KT)

    s_exp = s.view(torch.uint8).to(torch.int32)[:, :G_true, :]  # (E, G_true, N)
    shift = s_exp.amax(dim=1, keepdim=True)  # (E, 1, N) per-column max exponent
    ksh = (s_exp - shift).repeat_interleave(32, dim=1)  # (E, KT, N)
    shifted = mag * torch.exp2(ksh.float()).transpose(1, 2)  # (E, N, KT)

    w8 = shifted.to(torch.float8_e4m3fnuz)
    assert torch.equal(
        w8.float(), shifted.float()
    ), "requant_w2_to_fp8col: not bit-exact; per-column spread exceeds fp8 range"

    # kernel byte layout from (E, N, KT):
    #  - default:  (E, n_blk, k0, klane, nlane, p)          (two 8B loads per kt)
    #  - pair_k0:  (E, n_blk, kpair, klane, nlane, ku, p)   (one 16B load per kt;
    #              ku halves a K=64 tile: k = kpair*64 + ku*32 + klane*8 + p)
    if pair_k0:
        w8b = (
            w8.view(E, N // 16, 16, KT // 64, 2, 4, 8)
            .permute(0, 1, 3, 5, 2, 4, 6)
            .contiguous()
        )
    else:
        w8b = (
            w8.view(E, N // 16, 16, KT // 32, 4, 8)
            .permute(0, 1, 3, 4, 2, 5)
            .contiguous()
        )
    s_w = torch.exp2((shift.squeeze(1) - 127).float())  # (E, N) fp32
    return w8b.view(E, N * KT), s_w.contiguous()


def repack_w13_pairwise_to_a16w4_shuffled(w1_ktr, w1_scale_ktr, inter_real):
    """DeepSeek-V4.1 production w13 pairwise fp4 G1 -> FlyDSL a16w4 gemm1 inputs.

    Production vLLM storage (AiterW4A16ExpertsMonolithic, verified 100% against
    the DeepSeek-V4.1-Flash checkpoint in iter-17 w13_layout_solve2):

      - W: view shaped (E, K_pack, N_pad) with N the contiguous dim (physical
        (E, N_pad, K_pack)), N_pad rows: gate/up *pairwise* interleaved at row
        granularity (row = 2j + u; gate row 2j, up row 2j+1, j < inter_real),
        low nibble = even k; rows [2*inter_real, N_pad) are zero pad.
      - scales: view shaped (E, K//32, N_pad), e8m0 u8, same pairwise rows.

    This repacks them -- layout only, no requantization, fp4 bytes untouched --
    into the a16w4 gemm1 ``standard`` inputs on the REAL (unpadded) N_OUT =
    2*inter_real columns:

      - w1_shuf:    (E, 2*inter_real, K_pack) GGUU (gate block then up block)
        plain k-inner, through shuffle_weight_a16w4.
      - scale_shuf: matching shuffle_scale_a16w4 output.

    Shrinks the W1 footprint and the G1 N-compute by N_pad/(2*inter_real)
    (25% at 1536 -> 1152) and costs nothing at runtime (load-time pass).
    ``w1_ktr``/``w1_scale_ktr`` may be the transposed server views; they are
    re-materialized physically here.
    """
    import torch

    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    w = w1_ktr.view(torch.uint8) if w1_ktr.dtype != torch.uint8 else w1_ktr
    s = w1_scale_ktr.view(torch.uint8)
    I = inter_real
    E = w.shape[0]
    if w.ndim != 3 or s.ndim != 3:
        raise ValueError(
            f"w1/scale must be 3D, got {tuple(w.shape)} / {tuple(s.shape)}"
        )
    # Server contract: w1 as (E, K_pack, N_pad) and scales as (E, K//32,
    # N_pad) -- N rows come last in both. Reorient to physical (E, N_pad, K).
    server_view = s.shape[-2] * 16 == w.shape[-2] and s.shape[-1] == w.shape[-1]
    if server_view:
        w_nk = w.transpose(-1, -2).contiguous()  # (E, N_pad, K_pack)
        s_nk = s.transpose(-1, -2).contiguous()  # (E, N_pad, K//32)
    else:  # already physical (E, N_pad, K-dim)
        w_nk = w.contiguous()
        s_nk = s.contiguous()
    assert s_nk.shape[-1] * 16 == w_nk.shape[-1], (
        f"scale K groups {tuple(s_nk.shape)} vs packed W bytes {tuple(w_nk.shape)}"
    )
    n_pad = w_nk.shape[1]
    assert n_pad >= 2 * I, f"N_pad {n_pad} < 2*inter_real {2*I}"
    # GGUU row order [g0..g_{I-1}, u0..u_{I-1}] from pairwise rows [2j+u].
    pair = torch.cat(
        [
            torch.arange(0, 2 * I, 2, device=w_nk.device),
            torch.arange(1, 2 * I, 2, device=w_nk.device),
        ]
    )
    w_plain = w_nk.index_select(1, pair).contiguous()  # (E, 2I, K_pack)
    s_plain = s_nk.index_select(1, pair).contiguous()  # (E, 2I, K//32)
    k_scale = s_plain.shape[-1]
    w_shuf = shuffle_weight_a16w4(w_plain, 16, False)
    s_shuf = shuffle_scale_a16w4(s_plain.view(E * 2 * I, k_scale), E, False)
    return w_shuf, s_shuf


def repack_w13_pairwise_to_a8w4fp4(w1_ktr, w1_scale_ktr, inter_real):
    """DeepSeek-V4.1 production w13 pairwise fp4 G1 -> a8w4-FP4W stage1 inputs.

    Same reorientation as :func:`repack_w13_pairwise_to_a16w4_shuffled`
    (pairwise gate/up rows -> GGUU over the REAL 2*inter_real columns, layout
    only, fp4 bytes untouched) but for the fp8-synthesis kernel
    (:mod:`.gemm1_a8w4fp4`): plain per-(column, group) e8m0 scale plane, the
    per-column exponent shift and the fp32 epilog scale, with semantics
    identical to :func:`requant_w2_to_fp8col` (value = mag * 2^(raw - shift),
    s_w = 2^(shift - 127)) -- so the in-register fp8 bytes are bit-exact vs
    the iter-17 offline cache (proven in microbench/fp8byte_sim.py).

    Returns ``(w_shuf, sc_u8, shift_u8, sw_f32)``:
      w_shuf:   (E, 2*I, K/2-bytes) shuffle_weight_a16w4 (16, gate_up=False),
      sc_u8:    (E, 2*I, K/32) plain e8m0 group scales,
      shift_u8: (E, 2*I) per-column amax e8m0 (== production byte range base),
      sw_f32:   (E, 2*I) 2^(shift - 127).

    Raises if any column's e8m0 spread exceeds 5 octaves: the affine byte
    synthesis is only bit-exact while every shifted value stays an e4m3fnuz
    normal (checked column-by-column -- could be relaxed per-column later).
    """
    import torch

    from aiter.ops.shuffle import shuffle_weight_a16w4

    w = w1_ktr.view(torch.uint8) if w1_ktr.dtype != torch.uint8 else w1_ktr
    s = w1_scale_ktr.view(torch.uint8)
    I = inter_real
    E = w.shape[0]
    if w.ndim != 3 or s.ndim != 3:
        raise ValueError(
            f"w1/scale must be 3D, got {tuple(w.shape)} / {tuple(s.shape)}"
        )
    server_view = s.shape[-2] * 16 == w.shape[-2] and s.shape[-1] == w.shape[-1]
    if server_view:
        w_nk = w.transpose(-1, -2).contiguous()  # (E, N_pad, K_pack)
        s_nk = s.transpose(-1, -2).contiguous()  # (E, N_pad, K//32)
    else:
        w_nk = w.contiguous()
        s_nk = s.contiguous()
    n_pad = w_nk.shape[1]
    assert n_pad >= 2 * I, f"N_pad {n_pad} < 2*inter_real {2 * I}"
    pair = torch.cat(
        [
            torch.arange(0, 2 * I, 2, device=w_nk.device),
            torch.arange(1, 2 * I, 2, device=w_nk.device),
        ]
    )
    w_plain = w_nk.index_select(1, pair).contiguous()  # (E, 2I, K_pack)
    s_plain = s_nk.index_select(1, pair).contiguous()  # (E, 2I, K//32)
    w_shuf = shuffle_weight_a16w4(w_plain, 16, False)

    shift = s_plain.amax(dim=2)  # (E, 2I) u8, per-column max exponent byte
    spread = (shift.unsqueeze(-1).to(torch.int16) - s_plain.to(torch.int16)).max()
    if int(spread.item()) > 5:
        raise RuntimeError(
            "repack_w13_pairwise_to_a8w4fp4: e8m0 per-column spread "
            f"{int(spread.item())} > 5 octaves; the fp8 byte synthesis is not "
            "bit-exact (shifted values would leave the e4m3fnuz normal range)"
        )
    sw = torch.exp2((shift.to(torch.int32) - 127).float()).contiguous()
    return (
        w_shuf.contiguous(),
        s_plain.contiguous(),
        shift.contiguous(),
        sw.contiguous(),
    )


def requant_a_to_fp8(x_bf16, k_true):
    """bf16 [rows, K_pad] intermediate -> (fp8 [rows, k_true], s_a [rows] fp32).

    Prototype host version (torch ops); the production server path uses a fused
    kernel. Per-row amax over the true K columns.

    Scaled to +-240, NOT the e4m3fnuz max of 448: this torch build's
    ``.to(torch.float8_e4m3fnuz)`` cast overflows to NaN above ~248 (it applies
    e4m3fn saturating-NaN semantics, measured iter-10 on ROCm). 240 keeps every
    rounded value inside the cast's working range; the scale is fp32 so no
    dynamic range is lost, only <1 bit of the fp8 exponent vs scaling to 448.
    """
    import torch

    x = x_bf16[:, :k_true].float()
    amax = x.abs().amax(dim=-1).clamp(min=1e-12)
    s_a = amax / 240.0
    a8 = (x / s_a[:, None]).to(torch.float8_e4m3fnuz)
    return a8.contiguous(), s_a.contiguous()



# gfx942 (CDNA3) lacks K=32 mfma_f32_16x16x32_bf16 + v_cvt_pk_bf16_f32 -> K=16 MFMA +
# scalar dequant fallback; gfx950 (CDNA4) uses the K=32 path. compile_gemm{1,2}_a16w4_port
# are @functools.cache'd, so building through them directly (all-keyword) shares one
# compile with moe_kernels.py's AOT/runtime direct calls.


def flydsl_a16w4_gemm1(
    *,
    a_bf16,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=None,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    k_wave=1,
    b_nt=None,
    a_nt=None,
    xcd_swizzle=0,
    gate_mode="separated",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
    w_dtype="fp4",
    w_layout="standard",
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="fp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (``pack_int8_to_packed_int4(shuffle_weight(w, (16,16)))``),
    ``w1_scale_u8`` groupwise bf16 in the ``shuffle_scale_for_int4`` ``(E, G//2, N_OUT, 2)``
    bf16-pair layout (dword = e*(G//2*N_OUT) + (group//2)*N_OUT + n).
    ``"bf16"``: RAW bf16 W1 preshuffled ``shuffle_weight (16,16)``; ``w1_scale_u8`` unused.

    ``w_layout="standard"`` (default) consumes the N-major GGUU preshuffle. ``"guinterleave"``
    (mxfp4 only) consumes aiter's native GUGU stage1 W1+scale layout
    (``shuffle_weight_a16w4``/``shuffle_scale_a16w4``) directly, with no host relayout.

    ``a_bf16`` is bf16 ``[n_tokens, D_HIDDEN]``. Writes the bf16 intermediate
    ``[sorted_size, D_INTER]`` (by sorted position) into ``inter_sorted_bf16``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier (0=cached, 2=nt),
    ``xcd_swizzle`` -> XCD/HBM grid remap, ``k_wave`` -> intra-block slice-K ({1,2,4}).
    ``k_batch``/``gate_mode`` accepted for parity (only k_batch=1/separated supported).
    Tiles are CSV/registry-driven and always supplied by the caller; ``b_nt`` is
    the W-load cache modifier (0=cached, 2=nt), taken as-is (nt when unset).

    ``situ_beta``/``situ_linear_beta`` (SiTUv2 only) are runtime f32 scalars (nothing baked).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )

    BM = tile_m
    TILE_K = tile_k
    TILE_N = tile_n
    # Tile config is fully CSV/registry-driven: the caller resolves tile_n/tile_k/
    # k_wave/b_nt/xcd_swizzle from the tuned kernelName1 and always passes them.
    b_cache_mod = 2 if b_nt is None else b_nt
    # a_nt: A (activation) DMA cache modifier (None/2 = streaming, 0 = cached).
    # A rows are re-read once per N-block, so caching pays off at small
    # NUM_N_BLOCKS shapes like the unpadded DSV4.1 G1 (3 blocks at tn192).
    a_cache_mod = 2 if a_nt is None else a_nt
    if D_HIDDEN % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires D_HIDDEN (K) % {TILE_K} == 0, got H={D_HIDDEN}"
        )
    # Whole 16-col W blocks / 32-col e8m0 scale groups are what the layout
    # needs; 256 was the shipped-models' alignment (unpadded DSV4.1 G1 = 1152).
    if (2 * D_INTER) % 32 != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires 2*D_INTER % 32 == 0, got D_INTER={D_INTER}"
        )
    if D_INTER % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires D_INTER % TILE_N({TILE_N}) == 0, got D_INTER={D_INTER}"
        )

    launch = compile_gemm1_a16w4_port(
        BM=BM,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        act=act,
        b_cache_mod=b_cache_mod,
        a_cache_mod=a_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        w_layout=w_layout,
        k_wave=k_wave,
        rocm_arch=str(get_rocm_arch()),
    )
    max_m_blocks = int(sorted_expert_ids.numel())
    grid = gemm1_a16w4_grid(BM, INTER=D_INTER, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
    # SiTUv2 beta/linear_beta + swiglu_limit -> runtime f32 scalars (host precomputes
    # reciprocals; no device rcp). swiglu_limit is the SiTUv2 clamp bound (+inf =
    # no clamp), matching the a8w4/mixed_moe situv2 clamp so a16w4 == a8w4.
    _beta = float(situ_beta)
    _lbeta = float(situ_linear_beta)
    if _beta <= 0.0 or _lbeta <= 0.0:
        raise ValueError(
            f"situ_beta/situ_linear_beta must be > 0, got {_beta!r}/{_lbeta!r}"
        )
    _run_compiled(
        launch,
        a_bf16.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        m_indices.data_ptr(),
        int(n_tokens),
        int(grid),
        _beta,
        1.0 / _beta,
        _lbeta,
        1.0 / _lbeta,
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


def flydsl_a16w4_gemm2(
    *,
    inter_sorted_bf16,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=256,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    b_nt=None,
    xcd_swizzle=1,
    w_dtype="fp4",
    persist=None,
    pipe=None,
    epilog="atomic",
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage2 (down-proj). Consumes the bf16 [sorted_size,
    D_INTER] intermediate; scatters routing-weighted bf16 into ``flat_out``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM
    grid remap. ``k_batch`` for parity (must be 1). ``b_nt`` is the W-load cache
    modifier (0=cached, 2=nt), taken as-is (cached when unset).

    ``pipe`` (opt-in, default OFF; byte-identical when off): gemm1-aligned software
    pipeline -- double-buffered A-LDS + B prefetch (iter-10 DeepSeek-V4.1-Flash).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")

    BM = tile_m
    TILE_N = tile_n
    TILE_K = tile_k
    # Tile config is CSV/registry-driven: the caller resolves tile_n/tile_k/b_nt/
    # xcd_swizzle from the tuned kernelName2 and always passes them.
    if D_INTER % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_INTER (K) % {TILE_K} == 0, got D_INTER={D_INTER}"
        )
    if D_HIDDEN % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_HIDDEN (model_dim) % {TILE_N} == 0, got H={D_HIDDEN}"
        )

    # W-load cache modifier (0=cached, 2=nt): CSV/registry-driven via b_nt, taken
    # as-is (cached when unset).
    _b_cache_mod = 0 if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    # Persistent CU-limited grid (opt-in, default OFF; byte-identical when off): does NOT
    # close the E896 gap (padded launch's empty CTAs early-return ~free), kept as an
    # opt-in building block.
    _persist = False if persist is None else bool(persist)
    _pipe = False if pipe is None else bool(pipe)
    launch = compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=D_HIDDEN,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=_b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        persist=_persist,
        rocm_arch=str(get_rocm_arch()),
        epilog=epilog,
        topk=topk,
        pipe=_pipe,
    )
    grid = gemm2_a16w4_grid(
        BM, N_OUT=D_HIDDEN, TILE_N=TILE_N, max_m_blocks=max_m_blocks, persist=_persist
    )
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid),
        flat_out.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out


def flydsl_a8w4_gemm2(
    *,
    inter_fp8,
    w8_preshuffled,
    w2_col_scale,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    combined_weights,
    flat_out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    KLEN,
    topk,
    tile_m=128,
    tile_n=128,
    tile_k=64,
    waves_per_eu=None,
    b_nt=None,
    a_stages=2,
    a_mode=0,
    epilog_mode=0,
    b_v16=0,
    a_v16=0,
    xcd_swizzle=0,
    stream=None,
):
    """a8w4 stage2 (iter-10): fp8 A x requantized W4 (fp8, per-column scale).

    ``inter_fp8``        e4m3fnuz [sorted_size, KLEN] produced by requant_a_to_fp8.
    ``w8_preshuffled``   fp8 bytes from requant_w2_to_fp8col (per expert contiguous).
    ``w2_col_scale``     fp32 (E, N) per-column scale (epilogue multiply).
    ``combined_weights`` fp32 per sorted row = gamma * s_a[row] (host-side product).
    ``flat_out``         zeroed bf16 [M_logical, D_HIDDEN] (atomic accumulate).
    """
    _b_cache_mod = 2 if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    launch = compile_gemm2_a8w4(
        BM=tile_m,
        NE=NE,
        N_OUT=D_HIDDEN,
        KLEN=KLEN,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=_b_cache_mod,
        a_stages=a_stages,
        a_mode=a_mode,
        epilog_mode=epilog_mode,
        b_v16=b_v16,
        a_v16=a_v16,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        rocm_arch=str(get_rocm_arch()),
    )
    grid = gemm2_a8w4_grid(
        tile_m, N_OUT=D_HIDDEN, TILE_N=tile_n, max_m_blocks=max_m_blocks,
        full_n=(a_mode == 2),
    )
    _run_compiled(
        launch,
        inter_fp8.data_ptr(),
        w8_preshuffled.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.to(torch.int32).contiguous().data_ptr(),
        sorted_token_ids.data_ptr(),
        combined_weights.data_ptr(),
        w2_col_scale.view(-1).contiguous().data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid),
        flat_out.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out


def flydsl_a8w4fp4_gemm1(
    *,
    a8_fp8,
    s_a,
    w_fp4_shuffled,
    w_scale_u8,
    w_shift_u8,
    w_col_scale,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_out_rows,
    inter_sorted_bf16,
    n_tokens,
    NE,
    INTER,
    KLEN,
    topk,
    swiglu_limit=10.0,
    tile_m=32,
    tile_n=192,
    tile_k=64,
    waves_per_eu=None,
    b_nt=None,
    a_stages=2,
    xcd_swizzle=0,
    act="silu_clamp",
    dbg_bconst=0,
    dbg_braw=0,
    dbg_bdump=0,
    stream=None,
):
    """a8w4-FP4W stage1 (iter-18): fp8 A x native mxfp4 W1, in-register fp8 synth.

    ``a8_fp8``          e4m3fnuz [n_tokens, KLEN] from requant_hidden_fp8.
    ``s_a``             fp32 [n_tokens + 1] per-token A scales (+1 pad row).
    ``w_fp4_shuffled``  from repack_w13_pairwise_to_a8w4fp4 (a16w4 ``standard``
                        N-major GGUU preshuffle, real 2*INTER columns).
    ``w_scale_u8``      (E, 2*INTER, K/32) plain e8m0 group scales.
    ``w_shift_u8``      (E, 2*INTER) per-column shift (amax e8m0).
    ``w_col_scale``     fp32 (E, 2*INTER) = 2^(shift - 127) (epilog).
    ``sorted_out_rows`` i32 [P]: aiter sorted output row per moe-sort position
                        (-1 padding); from sorted_output_rows.
    ``inter_sorted_bf16`` bf16 [aiter_rows, INTER] output (production contract).
    """
    _b_cache_mod = 2 if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    launch = compile_gemm1_a8w4fp4(
        BM=tile_m,
        NE=NE,
        INTER=INTER,
        KLEN=KLEN,
        TOPK=topk,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=_b_cache_mod,
        a_stages=a_stages,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        act=act,
        dbg_bconst=dbg_bconst,
        dbg_braw=dbg_braw,
        dbg_bdump=dbg_bdump,
        rocm_arch=str(get_rocm_arch()),
    )
    grid = gemm1_a8w4fp4_grid(
        tile_m, INTER=INTER, TILE_N=tile_n, max_m_blocks=max_m_blocks
    )
    _run_compiled(
        launch,
        a8_fp8.data_ptr(),
        w_fp4_shuffled.data_ptr(),
        w_scale_u8.data_ptr(),
        w_shift_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.to(torch.int32).contiguous().data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_out_rows.data_ptr(),
        s_a.data_ptr(),
        w_col_scale.view(-1).contiguous().data_ptr(),
        int(n_tokens),
        int(max_m_blocks),
        int(grid),
        int(inter_sorted_bf16.shape[0]),
        1.0,
        1.0,
        1.0,
        1.0,
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


# iter-14 server wire-in (imports the wrappers above; keep at module end).
from .stage2_a8w4_server import gather_requant_fp8, moe_stage2_a8w4_forward
# iter-18 server wire-in (imports the wrappers above; keep at module end).
from .stage1_a8w4fp4_server import requant_hidden_fp8, moe_stage1_a8w4fp4_forward
