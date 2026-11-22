# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""a8w4 MoE stage2 GEMM (iter-10 DeepSeek-V4.1-Flash): fp8 A x requantized W4.

gfx942 fp8-MMA MoE down-proj. The production W4A16 path runs a Triton GEMM2 at
362 TFLOPS = 49.5% of the node's BF16 peak (measured, iter-10): the dot_scaled
e2m1 dequant is paid once per (m-block, W-tile) and the fp4 pipeline does not
fit gfx942's 64KB LDS. This kernel converts the problem to plain fp8 x fp8:

  * A (stage1 intermediate): e4m3fnuz fp8 rows [sorted_size, KLEN] written by an
    online requant pass; per-row fp32 scale folded host-side into the sorted
    routing weights (gamma * s_a[row]).
  * W2: the disk mxfp4 (e2m1 x e8m0-per-32) weights are requantized ONCE, OFFLINE,
    to e4m3fnuz bytes with a PER-COLUMN exponent shift. Real DeepSeek-V4.1-Flash
    w2 scales span <= 5 octaves per column, so the shifted e2m1 values are
    bit-exact in e4m3fnuz (iter-10 measured on the checkpoint). The per-column
    fp32 scale (2^(shift-127)) is applied in this kernel's epilogue.
  * KLEN=576 (true inter): K-padding (768) is skipped; A rows carry 576 fp8 and
    W columns carry 576 fp8 (both prepared host-side).
  * MFMA(16,16,32, fp8fnuz): single K=32 atom per fragment (CDNA3 has fp8-K32).
  * Epilog: routing-weighted atomic bf16 scatter to [tokens, model_dim]
    (= the a16wmix _atomic_bf16_epilog with the s_w[column] multiply folded into
    the LDS staging).

Layout clone of the a16wmix stage2: same sorted/cumsum/stids contract, same
grid ((m-blocks x n-blocks), xcd swizzle), same 256-thread CTA.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.layout_utils import crd2idx

from .utils import (
    _a16w4_swizzle_xor16,
    _global_i32_at,
    _raw,
    _udiv,
    _umod,
)

# fp8 (e4m3fnuz) = 1 byte; the MFMA K=32 fp8 fragment per lane = 8 fp8 = 2 dwords.
# Passed as an i32-typed rmem frag (same pattern as gemm_a8w8_8wave's fp8 MFMA).
FP8 = fx.Float8E4M3FNUZ


def _fp8_frag8(v2i32):
    """Pack 2 i32 (8 fp8) as the operand fragment tensor fx.gemm expects."""
    t = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
    t.store(Vec.from_elements([_raw(v2i32[0]), _raw(v2i32[1])], fx.Int32))
    return t


@flyc.jit
def _atomic_bf16_epilog_ws(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    arg_swcol,
    e_x_N_out,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
    epilog_mode,
):
    """a16wmix _atomic_bf16_epilog + per-column weight scale.

    acc (f32) -> x s_w[col] -> bf16 -> LDS -> pairs x combined weight
    (gammas * s_a[row], host-side) -> global_bf16 atomic add.
    Per-column scale: fp32 at swcol[e*N_OUT + n_block_idx*BN + col]
    (``e_x_N_out`` = e * N_OUT, precomputed by the caller).
    """
    _kMChunks = BM // 16
    M_REPS = BM // 8
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base_bptr = lds_typed_ptr(lds_acc_base_i32, T.bf16)

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)

    def _flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    swc = _flat_buffer(arg_swcol, T.f32, 4)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    out_elems = global_typed_ptr(arg_out, T.bf16, align=4)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, fx.Int32))
        weight.append(load_scalar(load_f32, sweights, sorted_pos, fx.Float32))

    col_base = e_x_N_out + n_block_idx * fx.Int32(BN) + wave * fx.Int32(_n_per_wave) + lane_mod_16
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            sw = load_scalar(load_f32, swc, col_base + fx.Int32(J * 16), fx.Float32)
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                lds_base_bptr[idx] = (fx.Float32(vec[v]) * sw).to(fx.BFloat16)

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    lds_vec_load(
                        lds_acc_base_i32,
                        idx0 * fx.Int32(2),
                        Vec.make_type(2, fx.BFloat16),
                        fx.BFloat16,
                        align=4,
                    )
                )
                w2 = v2.to(fx.Float32)
                pk = Vec.from_elements(
                    [w2[0] * weight[mr], w2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                out_off = row_base_addr + fx.Int32(s * 64)
                if const_expr(epilog_mode == 0):
                    fx.atomic_add(
                        (out_elems + out_off).llvm_ptr,
                        pk,
                        syncscope=fx.rocdl.SyncScope.Agent,
                    )
                else:
                    # epilog_mode 1: TIMING ONLY -- plain store instead of the
                    # atomic add (wrong results for tokens hit by several rows).
                    # Measures the atomic's share of the end-to-end time.
                    from flydsl._mlir.dialects import llvm as _llvm

                    _llvm.store(
                        pk.ir_value(),
                        (out_elems + out_off).llvm_ptr,
                        alignment=4,
                    )


def _gemm2_body_a8w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_swcol,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    TILE_N,
    TILE_K,
    N_OUT,
    KLEN,
    NE,
    b_cache_mod=2,
    a_stages=2,
    a_mode=0,
    epilog_mode=0,
    rocm_arch="",
):
    """a8w4 stage2 body. K=KLEN (576 true inter), N=N_OUT (model_dim).

    A = e4m3fnuz stage1 intermediate by SORTED position [rows, KLEN].
    W2 = e4m3fnuz bytes (per-column-shifted mxfp4, lossless), (16,16)-preshuffled
    byte layout (N/16, KLEN/64, 4, 16, 16) per expert, contiguous experts.
    Output = bf16 atomic-fadd (routing-weighted) scatter to [tokens, model_dim].
    """
    _is_gfx942 = str(rocm_arch).startswith("gfx942")
    KTILE_BYTES = TILE_K  # fp8: 1 byte per element
    K = KLEN
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    k_unroll = TILE_K // 32  # K32 MFMA steps per K tile
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KTILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N
    A_STAGES = max(1, min(a_stages, K_TILES_TOTAL))
    A_SLOT_BYTES = BM * KTILE_BYTES

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)
    by_n = n_block_idx * fx.Int32(TILE_N)

    # ---- B (W2 fp8, preshuffled bytes) ------------------------------------
    # Byte layout per expert: (N/16, K/32, klane=4, nlane=16, kpack=8), strides
    # ((K/32)*512, 512, 128, 8, 1). One MFMA(16,16,32,fp8) B fragment per lane:
    # v2i32 = 8 fp8 = byte (n_blk, k0, klane=lane_div_16, nlane=lane_mod_16, p)
    # covering k = k0*32 + klane*8 + [0,8) -- one dwordx2 load, no conversion.
    expert_off = e * fx.Int32(N_OUT)
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
        # per ku: one dwordx2 (8 fp8) at k0 = kt*(TILE_K/32) + ku
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

    # ---- A (fp8 intermediate) ---------------------------------------------
    # LDS slab: A_STAGES slots of BM x TILE_K bytes, XOR16 swizzled; frag reads
    # ds_read 16B = 16 fp8 = 2 consecutive ku frags.
    # Copy width adapts to the tile: 16B when the tile gives >=16B per thread,
    # else 8B or 4B (small BM*TK tiles) -- anything else would silently stage
    # NOTHING (BM=32/TK=64 gave bytes_per_thread=8 -> num_x_loads=0 with the
    # original fixed 16B loop; found by cos_diff=nan).
    a_num_i32 = A_STAGES * BM * KTILE_BYTES // 4  # (unused naming ok)
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
        x_row_base_bytes.append((m_row + row_local) * fx.Int32(K))

    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _global_i32_buffer_view

    x_buf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF))
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(chunk_i32, 1))
    _bcopy = {4: fx.rocdl.BufferCopy128b, 2: fx.rocdl.BufferCopy64b, 1: fx.rocdl.BufferCopy32b}[
        chunk_i32
    ]
    x_dma_atom = fx.make_copy_atom(_bcopy(b_cache_mod), fx.Int32)
    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr), fx.make_layout(a_num_i32 * 1, 1)
    )
    tiles_lds = fx.logical_divide(s_x_i32_flat, fx.make_layout(chunk_i32, 1))
    lds_atom = fx.make_copy_atom(fx.UniversalCopy(32 * chunk_i32), fx.Int32)

    def store_a_tile(k_byte, slot):
        # stage A[all BM rows][k_byte : k_byte + TKTILE_BYTES] into LDS slot
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

    tiles_lds8 = fx.logical_divide(s_x_i32_flat, fx.make_layout(2, 1))  # 8-byte tiles
    lds_atom64 = fx.make_copy_atom(fx.UniversalCopy(64), fx.Int32)

    def lds_load_a(mi, ku, slot):
        # v2i32 fp8 fragment: A[row][kt*TK.. + ku*32 + lane_div_16*8 .. +8]:
        # one 8-byte ds_read. Block-swizzle operates on 16B blocks (store side);
        # a lane pair shares one 16B block and reads its low/high 8B half.
        row = lane_mod_16 + fx.Int32(mi * 16)
        blk = fx.Int32(ku * 2) + lane_div_16 // fx.Int32(2)
        half = lane_div_16 % fx.Int32(2)  # 0/1: low or high 8B of the 16B block
        blk_sw = blk ^ (row & fx.Int32(k_blocks16 - 1))
        tile8_idx = fx.Int32((slot * A_SLOT_BYTES + row * KTILE_BYTES) // 8) + blk_sw * fx.Int32(2) + half
        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        fx.copy_atom_call(lds_atom64, fx.slice(tiles_lds8, (None, tile8_idx)), r)
        v = fx.Vector(fx.memref_load_vec(r))
        return [fx.Int32(v[0]), fx.Int32(v[1])]

    # ---- N-column addressing ----------------------------------------------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    cols = [col(by_n, n_tile_base, fx.Int32(ni * 16)) for ni in range_constexpr(num_acc_n)]

    # ---- accumulators ------------------------------------------------------
    acc_layout = fx.make_layout(4, 1)
    accm = [
        [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)]
        for _ in range(m_repeat)
    ]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, FP8))

    def load_b_tile(kt):
        return [load_b_raw(kt, c) for c in cols]

    def preload_a(slot):
        return [
            [lds_load_a(mi, ku, slot) for ku in range_constexpr(k_unroll)]
            for mi in range_constexpr(m_repeat)
        ]

    def compute_tile(b_tile, a_frags):
        for ni in range_constexpr(num_acc_n):
            fa = b_tile[ni]
            for ku in range_constexpr(k_unroll):
                bb = _fp8_frag8(fa[ku])
                for mi in range_constexpr(m_repeat):
                    aa = _fp8_frag8(a_frags[mi][ku])
                    fx.gemm(mma_atom, accm[mi][ni], aa, bb, accm[mi][ni])

    # ---- a_mode 1: A fragments straight from global (no LDS, no K-loop
    # barrier). The same m-block's A tile is shared by N_OUT/TILE_N CTAs of
    # this kernel, so these loads hit L1/L2/MALL; register double buffering
    # hides the latency exactly like load_b_tile does for W.
    x_dma_tiles8 = fx.logical_divide(x_buf, fx.make_layout(2, 1))
    a_g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(0), fx.Int32)

    def load_a_g(kt):
        fr = []
        for mi in range_constexpr(m_repeat):
            row_b = (m_row + fx.Int32(mi * 16) + lane_mod_16) * fx.Int32(K) + (
                lane_div_16 * fx.Int32(8)
            )
            ktb = fx.Int32(kt * TILE_K)
            fi = []
            for ku in range_constexpr(k_unroll):
                byte = row_b + ktb + fx.Int32(ku * 32)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy(a_g_atom, fx.slice(x_dma_tiles8, (None, byte // fx.Int32(8))), r)
                v = fx.Vector(fx.memref_load_vec(r))
                fi.append([fx.Int32(v[0]), fx.Int32(v[1])])
            fr.append(fi)
        return fr

    # ---- main K loop (gemm1-style software pipeline, a_stages deep) --------
    if const_expr(a_mode == 1):
        a_cur = load_a_g(0)
        b_cur = load_b_tile(0)
        for kt in range_constexpr(K_TILES_TOTAL):
            if const_expr(kt + 1 < K_TILES_TOTAL):
                a_nxt = load_a_g(kt + 1)
                b_nxt = load_b_tile(kt + 1)
            compute_tile(b_cur, a_cur)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                a_cur = a_nxt
                b_cur = b_nxt
    elif const_expr(K_TILES_TOTAL == 1):
        store_a_tile(0, 0)
    elif const_expr(K_TILES_TOTAL == 1):
        store_a_tile(0, 0)
        b0 = load_b_tile(0)
        gpu.barrier()
        compute_tile(b0, preload_a(0))
        gpu.barrier()
    else:
        for s in range_constexpr(A_STAGES - 1):
            if const_expr(s < K_TILES_TOTAL):
                store_a_tile(s * TILE_K, s)
            if const_expr(s == 0):
                b_cur = load_b_tile(0)
        for kt in range_constexpr(K_TILES_TOTAL):
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            a_frags = preload_a(kt % A_STAGES)
            if const_expr(kt + A_STAGES - 1 < K_TILES_TOTAL):
                store_a_tile((kt + A_STAGES - 1) * TILE_K, (kt + A_STAGES - 1) % A_STAGES)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                b_nxt = load_b_tile(kt + 1)
            compute_tile(b_cur, a_frags)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                b_cur = b_nxt

    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr)) + (
        fx.Int32(0) if const_expr(a_mode == 1) else A_STAGES * A_SLOT_BYTES
    )
    accm_v = [
        [accm[i][J].load().ir_value() for J in range(num_acc_n)]
        for i in range(m_repeat)
    ]
    _atomic_bf16_epilog_ws(
        lds_acc_base_i32,
        accm_v,
        arg_out,
        arg_stids,
        arg_sweights,
        arg_swcol,
        e * fx.Int32(N_OUT),
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT,
        TILE_N,
        epilog_mode,
    )


def gemm2_a8w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks):
    return int(max_m_blocks) * (N_OUT // TILE_N)


@functools.cache
def compile_gemm2_a8w4(
    BM=128,
    *,
    NE,
    N_OUT,
    KLEN,
    TILE_N=128,
    TILE_K=64,
    xcd_swizzle=0,
    b_cache_mod=2,
    a_stages=2,
    a_mode=0,
    epilog_mode=0,
    waves_per_eu=None,
    rocm_arch,
):
    """a8w4 (fp8 A x W4-requantized-to-fp8) MoE stage2 builder, gfx942.

    LDS: A a_stages slots x BM x TILE_K bytes + epilog BM x TILE_N bf16 -> 48KB at
    the default tile (fits the 64KB gfx942 budget).
    """
    assert BM % 16 == 0
    assert TILE_N % 64 == 0 and TILE_N // 4 >= 16
    assert KLEN % TILE_K == 0, f"KLEN must be a multiple of TILE_K, got {KLEN}/{TILE_K}"
    assert TILE_K % 64 == 0, f"TILE_K must be a multiple of 64, got {TILE_K}"
    _num_n_blocks = N_OUT // TILE_N
    _a_stages = max(1, min(a_stages, (KLEN // TILE_K) if (KLEN // TILE_K) > 1 else 1))
    _lds_bytes = (0 if a_mode == 1 else _a_stages * BM * TILE_K) + BM * TILE_N * 2

    _name = f"gemm2_a8w4_ne{NE}_h{N_OUT}_i{KLEN}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if a_stages != 2:
        _name += f"_as{a_stages}"
    if epilog_mode != 0:
        _name += f"_ep{epilog_mode}"
    if a_mode != 0:
        _name += f"_ga{a_mode}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_swcol: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = xc * _xq + fx.min(xc, _xr) + _udiv(pid, _NXCD)
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.min(remaining_m, fx.Int32(_SW))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        if bx_i32 < bound:
            _tile = _xcd_np(bx_i32)
            _gemm2_body_a8w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_swcol,
                arg_out,
                _tile,
                lane,
                wave,
                i32_M,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                KLEN=KLEN,
                NE=NE,
                b_cache_mod=b_cache_mod,
                a_stages=a_stages,
                a_mode=a_mode,
                epilog_mode=epilog_mode,
                rocm_arch=rocm_arch,
            )

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_swcol: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_swcol,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
