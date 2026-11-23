# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""a8w4 MoE stage1 GEMM (iter-17 DeepSeek-V4.1-Flash): fp8 A x requantized W4.

gfx942 fp8-MMA gate+up GEMM + fused activation. The production W4A16 path runs
a Triton GEMM1 (``_moe_gemm_a16w4``) at ~50% of the node's BF16 peak at the
16k prefill chunk (3,975.6 us/call, iter-17 measurement of profiles/08): the
dot_scaled e2m1 dequant is paid once per (m-block, W-tile). This kernel
converts the problem to plain fp8 x fp8, same as iter-14's a8w4 G2:

  * A (routed hidden states): e4m3fnuz fp8 rows [n_tokens, KLEN] written by an
    online per-token requant pass (amax -> scale to +-240, iter-10 semantics);
    per-token fp32 scale s_a applied in this kernel's epilog. The kernel
    gathers A rows by token through the moe_sorting sorted ids (arg_mind) --
    the hidden row is shared by the token's topk slots, so the gather is NOT
    pre-materialized (6x less requant work and A traffic).
  * W1 (gate|up): the disk mxfp4 (e2m1 x e8m0-per-32) weights requantized ONCE,
    OFFLINE, to e4m3fnuz bytes with a PER-COLUMN exponent shift (checkpoint
    per-column spread measured <= 5 octaves -> bit-exact; assert in
    requant_w2_to_fp8col). Per-column fp32 scale applied in the epilog.
  * KLEN = model_dim (5120); N_OUT = 2*INTER with the true (unpadded) inter:
    no K or N padding anywhere.
  * MFMA(16,16,32, fp8fnuz). Layout clone of gemm2_a8w4: same grid/xcd
    swizzle, same LDS-staged A with XOR16 swizzle, a_stages-deep software
    pipeline, 256-thread CTA.
  * Epilog: y = silu_clamp(acc_g * s_a* s_wg, acc_u * s_a * s_wu); bf16 masked
    store at the AITER-SORTED output row (arg_orow, host-precomputed per
    moe-sort position = scatter_idx[token*topk+slot], -1 for padding), so the
    intermediate keeps the production aiter-routing-module contract and stage2
    is untouched.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.act import gate_up_act, situ_params
from aiter.ops.flydsl.kernels.layout_utils import crd2idx
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr

from .utils import (
    _a16w4_swizzle_xor16,
    _global_i32_at,
    _raw,
    _udiv,
    _umod,
)

FP8 = fx.Float8E4M3FNUZ


def _fp8_frag8(v2i32):
    """Pack 2 i32 (8 fp8) as the operand fragment tensor fx.gemm expects."""
    t = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
    t.store(Vec.from_elements([_raw(v2i32[0]), _raw(v2i32[1])], fx.Int32))
    return t


def _gemm1_body_a8w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_eids,
    arg_cumsum,
    arg_mind,
    arg_orow,
    arg_sa,
    arg_swcol,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    i32_out_rows,
    situ,
    *,
    BM,
    TILE_N,
    TILE_K,
    INTER,
    KLEN,
    NE,
    TOPK,
    act="silu_clamp",
    b_cache_mod=2,
    a_stages=2,
    b_v16=0,
    rocm_arch="",
):
    """a8w4 stage1 body. K=KLEN (model_dim), N cols: gate [0,INTER), up +INTER.

    A = e4m3fnuz hidden states [n_tokens, KLEN], rows gathered by token id
    from arg_mind (moe_sorting sorted ids). W1 = e4m3fnuz bytes
    (per-column-shifted mxfp4, lossless), (16,16)-preshuffled byte layout
    (N_OUT/16, KLEN/64, 4, 16, 2, 8) per expert with b_v16, contiguous
    experts. Output = bf16 activation-stored intermediate in aiter-sorted row
    order (arg_orow per moe-sort position).
    """
    N_OUT = 2 * INTER
    KTILE_BYTES = TILE_K  # fp8: 1 byte per element
    K = KLEN
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    k_unroll = TILE_K // 32  # K32 MFMA steps per K tile
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KTILE_BYTES // 16
    NUM_N_BLOCKS = INTER // TILE_N
    A_STAGES = max(1, min(a_stages, K_TILES_TOTAL))
    A_SLOT_BYTES = BM * KTILE_BYTES

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first moe-sort row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)
    inter_i32 = fx.Int32(INTER)
    n_out_i32 = fx.Int32(N_OUT)

    # ---- B (W1 fp8, preshuffled bytes) ------------------------------------
    # Same byte layout as gemm2_a8w4 but over N_OUT = 2*INTER columns per
    # expert: gate col c in [0, INTER), its up partner at c + INTER. One
    # MFMA(16,16,32,fp8) B fragment per lane per (col, ku).
    expert_off = e * n_out_i32
    _k0_blocks = K // 32
    bl_stride_klane = 128
    bl_stride_k0 = 512
    bl_stride_n0 = _k0_blocks * bl_stride_k0
    layout_b = fx.make_layout(
        (N_OUT // 16, _k0_blocks, 4, 16, 8),
        (bl_stride_n0, bl_stride_k0, bl_stride_klane, 8, 1),
    )
    _w_bytes = NE * N_OUT * K
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _global_i32_buffer_tiles

    w_tiles8 = _global_i32_buffer_tiles(arg_bq, min(_w_bytes, 0xFFFFFFFF), 2)
    w_copy_atom64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(b_cache_mod), fx.Int32)
    w_reg_lay2 = fx.make_layout(2, 1)

    if b_v16:
        # paired-k0 16B layout (N/16, K/64, klane=4, nlane=16, ku=2, kpack=8):
        # one BufferCopy128b per (kt, col). Requires TILE_K == 64.
        layout_b16 = fx.make_layout(
            (N_OUT // 16, K // 64, 4, 16, 2, 8),
            ((K // 64) * 1024, 1024, 256, 16, 8, 1),
        )
        w_tiles16 = _global_i32_buffer_tiles(arg_bq, min(_w_bytes, 0xFFFFFFFF), 4)
        w_copy_atom128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)

        def load_b_raw16(kt, col):
            idx_bytes = fx.Int32(
                crd2idx(
                    [
                        fx.Int64(col.n_blk),
                        fx.Int64(kt),
                        fx.Int64(lane_div_16),
                        fx.Int64(col.n_intra),
                        fx.Int64(0),
                        fx.Int64(0),
                    ],
                    layout_b16,
                )
            )
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(w_copy_atom128, fx.slice(w_tiles16, (None, idx_bytes // fx.Int32(16))), r)
            v4 = fx.Vector(fx.memref_load_vec(r))
            return [[fx.Int32(v4[0]), fx.Int32(v4[1])],
                    [fx.Int32(v4[2]), fx.Int32(v4[3])]]

    class _Col:
        __slots__ = ("n_blk", "n_intra")

        def __init__(s, n_blk, n_intra):
            s.n_blk = n_blk
            s.n_intra = n_intra

    def col(*n_terms):
        t = n_terms[0]
        for x2 in n_terms[1:]:
            t = t + x2
        row = expert_off + t + lane_mod_16
        return _Col(row // fx.Int32(16), row % fx.Int32(16))

    def load_b_raw(kt, col):
        ku_per_tile = TILE_K // 32
        raw = []
        for ku in range_constexpr(ku_per_tile):
            k0 = kt * ku_per_tile + ku
            idx_bytes = fx.Int32(
                crd2idx(
                    [
                        fx.Int64(col.n_blk),
                        fx.Int64(k0),
                        fx.Int64(lane_div_16),
                        fx.Int64(col.n_intra),
                        fx.Int64(0),
                    ],
                    layout_b,
                )
            )
            r = fx.make_rmem_tensor(w_reg_lay2, fx.Int32)
            fx.copy(w_copy_atom64, fx.slice(w_tiles8, (None, idx_bytes // fx.Int32(8))), r)
            v2 = fx.Vector(fx.memref_load_vec(r))
            raw.append([fx.Int32(v2[0]), fx.Int32(v2[1])])
        return raw

    # ---- A (fp8 hidden states, gathered by token) --------------------------
    a_load_threads = 256
    _tile_dwords = BM * KTILE_BYTES // 4
    assert _tile_dwords % a_load_threads == 0
    _per_thread_dwords = _tile_dwords // a_load_threads
    if _per_thread_dwords % 4 == 0:
        chunk_i32 = 4
    elif _per_thread_dwords % 2 == 0:
        chunk_i32 = 2
    else:
        chunk_i32 = 1
    assert chunk_i32 * 4 <= KTILE_BYTES
    x_load_bytes = chunk_i32 * 4
    num_x_loads = _per_thread_dwords // chunk_i32
    tile_k_dwords = KTILE_BYTES // 4
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    tx_base = tx_i32 * fx.Int32(chunk_i32)

    x_row_local = []
    x_col_dw = []
    x_row_base_bytes = []
    for i in range_constexpr(num_x_loads):
        tile_idx = tx_base + fx.Int32(i * a_load_threads * chunk_i32)
        row_local = tile_idx // fx.Int32(tile_k_dwords)
        x_row_local.append(row_local)
        x_col_dw.append(tile_idx % fx.Int32(tile_k_dwords))
        # token id of this moe-sort row (sentinel token = M; the sized A
        # buffer below clamps the OOB row read to 0 -- same a16w4 contract).
        fused = fx.Int32(_global_i32_at(arg_mind, m_row + row_local))
        tok = fused & fx.Int32(0x00FFFFFF)
        x_row_base_bytes.append(tok * fx.Int32(K))

    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _global_i32_buffer_view

    x_buf = _global_i32_buffer_view(arg_a, fx.Int64(i32_M) * fx.Int64(K))
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(chunk_i32, 1))
    _bcopy = {4: fx.rocdl.BufferCopy128b, 2: fx.rocdl.BufferCopy64b, 1: fx.rocdl.BufferCopy32b}[
        chunk_i32
    ]
    x_dma_atom = fx.make_copy_atom(_bcopy(b_cache_mod), fx.Int32)
    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr), fx.make_layout(A_STAGES * BM * KTILE_BYTES // 4, 1)
    )
    tiles_lds = fx.logical_divide(s_x_i32_flat, fx.make_layout(chunk_i32, 1))
    lds_atom = fx.make_copy_atom(fx.UniversalCopy(32 * chunk_i32), fx.Int32)

    def store_a_tile(k_byte, slot):
        base_dw = k_byte // 4
        slot_base = slot * A_SLOT_BYTES
        for i in range_constexpr(num_x_loads):
            col_bytes = x_col_dw[i] * 4
            col_sw = _a16w4_swizzle_xor16(
                x_row_local[i], fx.Int32(col_bytes), fx.Int32(k_blocks16), enable=True
            )
            global_byte = x_row_base_bytes[i] + base_dw * 4 + col_sw
            lds_byte = slot_base + x_row_local[i] * KTILE_BYTES + col_bytes
            src = fx.slice(x_dma_tiles4, (None, global_byte // fx.Int32(x_load_bytes)))
            dst = fx.slice(tiles_lds, (None, lds_byte // fx.Int32(x_load_bytes)))
            r = fx.make_rmem_tensor(fx.make_layout(chunk_i32, 1), fx.Int32)
            fx.copy(x_dma_atom, src, r)
            fx.copy(lds_atom, r, dst)

    tiles_lds8 = fx.logical_divide(s_x_i32_flat, fx.make_layout(2, 1))
    lds_atom64 = fx.make_copy_atom(fx.UniversalCopy(64), fx.Int32)

    def lds_load_a(mi, ku, slot):
        # v2i32 fp8 fragment: A[row][kt*TK.. + ku*32 + lane_div_16*8 .. +8].
        row = lane_mod_16 + fx.Int32(mi * 16)
        blk = fx.Int32(ku * 2) + lane_div_16 // fx.Int32(2)
        half = lane_div_16 % fx.Int32(2)
        blk_sw = blk ^ (row & fx.Int32(k_blocks16 - 1))
        tile8_idx = fx.Int32((slot * A_SLOT_BYTES + row * KTILE_BYTES) // 8) + blk_sw * fx.Int32(2) + half
        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        fx.copy_atom_call(lds_atom64, fx.slice(tiles_lds8, (None, tile8_idx)), r)
        v = fx.Vector(fx.memref_load_vec(r))
        return [fx.Int32(v[0]), fx.Int32(v[1])]

    # ---- N-column addressing for gate/up (separated) ----------------------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    col_g_i32 = []  # plain i32 gate column per ni (for scales + store offset)
    cols_gate, cols_up = [], []
    for ni in range_constexpr(num_acc_n):
        _ni16 = fx.Int32(ni * 16)
        cb = by_n + n_tile_base + _ni16
        col_g_i32.append(cb + lane_mod_16)
        cols_gate.append(col(by_n, n_tile_base, _ni16))
        cols_up.append(col(by_n, n_tile_base + inter_i32, _ni16))

    # ---- accumulators ------------------------------------------------------
    acc_layout = fx.make_layout(4, 1)
    acc_gate = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    acc_up = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            acc_gate[mi][ni].store(zero4)
            acc_up[mi][ni].store(zero4)

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, FP8))

    def _load_b(lb, kt):
        return ([lb(kt, c) for c in cols_gate], [lb(kt, c) for c in cols_up])

    def preload_a(slot):
        return [
            [lds_load_a(mi, ku, slot) for ku in range_constexpr(k_unroll)]
            for mi in range_constexpr(m_repeat)
        ]

    def compute_tile(b_tile, a_frags):
        b_g, b_u = b_tile
        for ni in range_constexpr(num_acc_n):
            fg = b_g[ni]
            fu = b_u[ni]
            for ku in range_constexpr(k_unroll):
                bg = _fp8_frag8(fg[ku])
                bu = _fp8_frag8(fu[ku])
                for mi in range_constexpr(m_repeat):
                    aa = _fp8_frag8(a_frags[mi][ku])
                    fx.gemm(mma_atom, acc_gate[mi][ni], aa, bg, acc_gate[mi][ni])
                    fx.gemm(mma_atom, acc_up[mi][ni], aa, bu, acc_up[mi][ni])

    lb = load_b_raw16 if b_v16 else load_b_raw

    # ---- main K loop (gemm1-style software pipeline, a_stages deep) --------
    for s in range_constexpr(A_STAGES - 1):
        if const_expr(s < K_TILES_TOTAL):
            store_a_tile(s * TILE_K, s)
        if const_expr(s == 0):
            b_cur = _load_b(lb, 0)
    for kt in range_constexpr(K_TILES_TOTAL):
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        a_frags = preload_a(kt % A_STAGES)
        if const_expr(kt + A_STAGES - 1 < K_TILES_TOTAL):
            store_a_tile((kt + A_STAGES - 1) * TILE_K, (kt + A_STAGES - 1) % A_STAGES)
        if const_expr(kt + 1 < K_TILES_TOTAL):
            b_nxt = _load_b(lb, kt + 1)
        compute_tile(b_cur, a_frags)
        if const_expr(kt + 1 < K_TILES_TOTAL):
            b_cur = b_nxt

    # ---- epilog: activation -> bf16 intermediate in aiter-sorted order ------
    # Per moe-sort row: arg_orow = aiter sorted row (host: scatter_idx[token*
    # topk + slot]), -1 for padding/sentinel rows (masked). s_a per token
    # (arg_sa has a +1 pad row: sentinel token M reads sa[M], defined garbage,
    # masked anyway). Per-column W scales: gate at swcol[e*N_OUT + c], up at
    # swcol[e*N_OUT + c + INTER].
    out_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_out)),
        num_records_bytes=_raw(fx.Int64(i32_out_rows) * fx.Int64(INTER * 2)),
    )

    def _flat_f32_buffer(arg):
        ptr = global_typed_ptr(arg, T.f32, align=4)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    sa_buf = _flat_f32_buffer(arg_sa)
    swc_buf = _flat_f32_buffer(arg_swcol)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    def load_f32_scalar(src, index):
        frag = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy(load_f32, src[None, index], frag)
        return Vec(frag.load())[0]

    for mi in range_constexpr(m_repeat):
        orow_l, valid_l, sa_l = [], [], []
        for ii in range_constexpr(4):
            row_in_tile = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4) + fx.Int32(ii)
            sorted_row = m_row + row_in_tile
            fused = fx.Int32(_global_i32_at(arg_mind, sorted_row))
            tok = fused & fx.Int32(0x00FFFFFF)
            orow = fx.Int32(_global_i32_at(arg_orow, sorted_row))
            orow_l.append(orow)
            valid_l.append(orow >= fx.Int32(0))
            sa_l.append(load_f32_scalar(sa_buf, tok))
        for ni in range_constexpr(num_acc_n):
            vg = Vec(fx.memref_load_vec(acc_gate[mi][ni]))
            vu = Vec(fx.memref_load_vec(acc_up[mi][ni]))
            c = col_g_i32[ni]
            sw_g = load_f32_scalar(swc_buf, e * n_out_i32 + c)
            sw_u = load_f32_scalar(swc_buf, e * n_out_i32 + c + inter_i32)
            for ii in range_constexpr(4):
                g = fx.Float32(vg[ii]) * sa_l[ii] * sw_g
                u = fx.Float32(vu[ii]) * sa_l[ii] * sw_u
                y = gate_up_act(act, [g], [u], situ)[0]
                buffer_ops.buffer_store(
                    y.to(fx.BFloat16),
                    _raw(out_rsrc),
                    _raw(orow_l[ii] * inter_i32 + c),
                    mask=valid_l[ii],
                )


def gemm1_a8w4_grid(BM, *, INTER, TILE_N, max_m_blocks):
    return int(max_m_blocks) * (INTER // TILE_N)


@functools.cache
def compile_gemm1_a8w4(
    BM=32,
    *,
    NE,
    INTER,
    KLEN,
    TOPK,
    TILE_N=192,
    TILE_K=64,
    xcd_swizzle=0,
    b_cache_mod=2,
    a_stages=2,
    b_v16=0,
    act="silu_clamp",
    waves_per_eu=None,
    rocm_arch,
):
    """a8w4 (fp8 A x W4-requantized-to-fp8) MoE stage1 builder, gfx942.

    LDS: A a_stages slots x BM x TILE_K bytes (4 KB at bm32/tk64/as2).
    """
    N_OUT = 2 * INTER
    assert BM % 16 == 0
    assert TILE_N % 64 == 0
    assert INTER % TILE_N == 0, (
        f"INTER must be a multiple of TILE_N, got {INTER}/{TILE_N}"
    )
    assert KLEN % TILE_K == 0, f"KLEN must be a multiple of TILE_K, got {KLEN}/{TILE_K}"
    assert TILE_K % 64 == 0, f"TILE_K must be a multiple of 64, got {TILE_K}"
    assert not b_v16 or TILE_K == 64, "b_v16 (paired-k0 16B B loads) requires TILE_K == 64"
    assert act in (
        "silu",
        "silu_clamp",
        "swiglu",
        "situv2",
    ), f"a8w4 gemm1 act must be 'silu', 'silu_clamp', 'swiglu', or 'situv2', got {act!r}"
    _a_stages = max(1, min(a_stages, KLEN // TILE_K))
    lds_bytes = _a_stages * BM * TILE_K

    _name = f"gemm1_a8w4_ne{NE}_i{INTER}_h{KLEN}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
    _act_tag = "" if act == "silu" else f"_{act}"
    _name += _act_tag
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if a_stages != 2:
        _name += f"_as{a_stages}"
    if b_v16:
        _name += "_bv16"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        arg_orow: fx.Int64,
        arg_sa: fx.Int64,
        arg_swcol: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_out_rows: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(INTER // TILE_N)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = xc * _xq + fx.min(xc, _xr) + _udiv(pid, _NXCD)
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * (INTER // TILE_N))
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.min(remaining_m, fx.Int32(_SW))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(INTER // TILE_N) + n_block

        if bx_i32 < bound:
            _tile = _xcd_np(bx_i32)
            if const_expr(act in ("swiglu", "situv2", "silu_clamp")):
                _situ = situ_params(
                    fx.Float32(f32_situ_beta),
                    fx.Float32(f32_situ_beta_rcp),
                    fx.Float32(f32_situ_linbeta),
                    fx.Float32(f32_situ_linbeta_rcp),
                    fx.Float32(f32_swiglu_limit),
                )
            else:
                _situ = None
            _gemm1_body_a8w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_eids,
                arg_cumsum,
                arg_mind,
                arg_orow,
                arg_sa,
                arg_swcol,
                arg_out,
                _tile,
                lane,
                wave,
                i32_M,
                i32_out_rows,
                _situ,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                INTER=INTER,
                KLEN=KLEN,
                NE=NE,
                TOPK=TOPK,
                act=act,
                b_cache_mod=b_cache_mod,
                a_stages=a_stages,
                b_v16=b_v16,
                rocm_arch=rocm_arch,
            )

    @flyc.jit
    def launch_gemm1(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        arg_orow: fx.Int64,
        arg_sa: fx.Int64,
        arg_swcol: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        i32_out_rows: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm1_kernel(
            arg_a,
            arg_bq,
            arg_eids,
            arg_cumsum,
            arg_mind,
            arg_orow,
            arg_sa,
            arg_swcol,
            i32_M,
            i32_max_m_blocks,
            i32_out_rows,
            f32_situ_beta,
            f32_situ_beta_rcp,
            f32_situ_linbeta,
            f32_situ_linbeta_rcp,
            f32_swiglu_limit,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1
