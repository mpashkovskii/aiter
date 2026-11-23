# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""a8w4-FP4W MoE stage1 GEMM (iter-18 DeepSeek-V4.1-Flash): fp8 A x native fp4 W1.

hybrid of the iter-17 a8w4 G1 (fp8 A x offline-requantized-fp8 W1, rejected on
its +28.2 GiB/rank cache) and the iter-18 a16w4 G1 (bf16 A x native fp4 W1,
cache-free but bf16-MMA-bound at ~1.15x). This kernel keeps the fp8 MMA
pipeline of the iter-17 kernel (fp8 A + per-column-shifted fp8 W bytes +
epilog per-column scale) but SYNTHESIZES the fp8 W operand in registers from
the production-native mxfp4 W1 (e2m1 nibbles + e8m0 per-32 group scales):

  fp8 value = e2m1_mag * 2^(raw_e8m0 - shift_col) is exactly representable in
  e4m3fnuz when the per-column e8m0 spread is <= 5 octaves (checkpoint measured
  <= 5; the load-time repack asserts it), and its byte is

    code = LUT[mag] + ((raw_e8m0 - shift_col) << 3),  sign OR'd on bit 7,
           zero-magnitude forced to 0x00 (FNUZ has made 0x80 = NaN)

  (microbench/fp8byte_sim.py proves the affine byte form bit-exact vs the
  iter-17 offline requant cast for spread <= 5). The exponent add runs in
  16-bit lanes (v_pk-compatible): e16 = b & 0x00FF00FF plus the broadcast
  offset, masked back -- borrow-free because nonzero codes stay in [0x08, 0x4C].

W1 layout input is the SAME a16w4 ``standard`` preshuffle the iter-18 a16w4
candidate consumes (``repack_w13_pairwise_to_a16w4_shuffled``): plain
N-major GGUU, layout (N_OUT/16, K/128, klane=4, nlane=16, 16B) where byte p of
a (klane) slot holds k = k128*128 + klane*32 + 2p + {0,1} (low nibble even k).
The fp8 MMA(16,16,32) B fragment for 32-K group q = K32i % 4 and lane
ld = lane // 16 covers k = K32i*32 + ld*8 + [0,8), which is bytes
[ld*4, ld*4+4) of the (k128 = K32i//4, klane = q) slot: one BufferCopy32b per
(col, K32i, lane). e8m0 scales come from a PLAIN (E, N_OUT, K/32) u8 plane and
the per-column shift from a PLAIN (E, N_OUT) u8 plane (both layout-only
derivatives of the production scales; semantics identical to
requant_w2_to_fp8col's shift/s_w pair, including the epilog fp32 s_w here).

Everything else -- A path (fp8 hidden states gathered by token id through
moe_sorting sorted ids, a_stages-deep pipeline, XOR16 LDS swizzle), grid/XCD
swizzle, silu_clamp epilog with per-column fp32 s_w and per-token s_a -- is
verbatim iter-17 gemm1_a8w4.py.
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

# e4m3fnuz byte codes of the E2M1 magnitudes (1), [0x30,0x38,0x3C] for
# 0.5/1/1.5 and (2) [0x40,0x44,0x48,0x4C] for 2/3/4/6 -- fp8byte_sim.py LUT.
_FP8_LUT_LO = 0x3C383000  # mags 0..3 -> 0x00,0x30,0x38,0x3C
_FP8_LUT_HI = 0x4C484440  # mags 4..7 -> 0x40,0x44,0x48,0x4C


def _perm_b32(a, b, sel):
    return fx.Int32(rocdl.perm_b32(_raw(a), _raw(b), _raw(sel)))


def _fp4w_win_to_fp8x8(raw_i32, cb16_i32):
    """Synthesize the 8 fp8 codes of one (col, K32i) B MMA fragment.

    ``raw_i32``: 8 e2m1 nibbles, nibble j = k offset j (low nibble of byte b is
    the even k of the pair byte b holds -- production low-nibble-for-even-k).
    ``cb16_i32``: exponent offset broadcast into both 16-bit lanes'
    low bytes: cb = (((raw_e8m0 - shift_col) << 3) & 0xFF) * 0x00010001.

    Returns [i32, i32]: the v2i32 fp8 fragment, byte order = ascending k,
    with all carries contained in 16-bit lanes (code range [0x08, 0x4C]
    excludes borrows; zero nibbles masked to 0x00 so FNUZ NaN 0x80 and
    offset pollution of exact zeros both never occur).
    """
    c0 = fx.Int32(0x07070707)
    sel_e = raw_i32 & c0  # mags of k 0,2,4,6 at bytes 0..3
    sel_o = (raw_i32.shrui(fx.Int32(4))) & c0  # mags of k 1,3,5,7
    lut_lo = fx.Int32(_FP8_LUT_LO)
    lut_hi = fx.Int32(_FP8_LUT_HI)
    out = []

    def synth(sel, sign_shifted):
        base = _perm_b32(lut_hi, lut_lo, sel)
        nz = ((sel + fx.Int32(0x7F7F7F7F)).shrui(fx.Int32(7))) & fx.Int32(0x01010101)
        m16 = fx.Int32(0x00FF00FF)
        e16 = base & m16
        o16 = (base.shrui(fx.Int32(8))) & m16
        se = (e16 + cb16_i32) & m16
        so = (o16 + cb16_i32) & m16
        # per-16-bit-lane masks 0 or 0xFF from the nz low byte
        em = ((nz & m16) * fx.Int32(0xFF)) & m16
        om = (((nz.shrui(fx.Int32(8))) & m16) * fx.Int32(0xFF)) & m16
        res = (se & em) | ((so & om).shl(fx.Int32(8)))
        # sign: masked by nz (zero-magnitude must stay +0; 0x80 is FNUZ NaN)
        res = res | (sign_shifted & (nz << fx.Int32(7)))
        return fx.Int32(res)

    # even-k half: sign bits are raw bit3 of each byte of sel parity
    sgn_e = (raw_i32 & fx.Int32(0x08080808)) << fx.Int32(4)
    # odd-k half: sign already sits at bit 7
    sgn_o = raw_i32 & fx.Int32(0x80808080)
    codes_e = synth(sel_e, sgn_e)  # fp8 of k 0,2,4,6 at bytes 0..3
    codes_o = synth(sel_o, sgn_o)  # fp8 of k 1,3,5,7 at bytes 0..3
    # interleave byte-wise: out0 = [e0,o0,e1,o1], out1 = [e2,o2,e3,o3]
    out.append(_perm_b32(codes_o, codes_e, fx.Int32(0x5140)))
    out.append(_perm_b32(codes_o, codes_e, fx.Int32(0x7362)))
    return out


def _fp8_frag8(v2i32):
    """Pack 2 i32 (8 fp8) as the operand fragment tensor fx.gemm expects."""
    t = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
    t.store(Vec.from_elements([_raw(v2i32[0]), _raw(v2i32[1])], fx.Int32))
    return t


def _gemm1_body_a8w4fp4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_bshift,
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
    rocm_arch="",
):
    """a8w4-FP4W stage1 body. K=KLEN (model_dim), N cols: gate [0,INTER), up +INTER.

    A = e4m3fnuz hidden states [n_tokens, KLEN] (gathered by token id).
    W1 = production-native mxfp4 (a16w4 ``standard`` preshuffle); the fp8 MMA
    operands are synthesized in registers (module docstring). arg_bscale:
    plain u8 (E, N_OUT, K/32); arg_bshift: plain u8 (E, N_OUT). Epilog scale
    s_w fp32 (E, N_OUT) = 2^(shift-127), iter-17 requant semantics.
    """
    N_OUT = 2 * INTER
    KTILE_BYTES = TILE_K  # fp8: 1 byte per element
    K = KLEN
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    k_unroll = TILE_K // 32  # K32 MMA steps per K tile
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KTILE_BYTES // 16
    NUM_N_BLOCKS = INTER // TILE_N
    A_STAGES = max(1, min(a_stages, K_TILES_TOTAL))
    A_SLOT_BYTES = BM * KTILE_BYTES
    G = K // 32  # e8m0 scale groups per column

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first moe-sort row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)
    inter_i32 = fx.Int32(INTER)
    n_out_i32 = fx.Int32(N_OUT)

    # ---- B (W1 fp4 packed, a16w4 standard preshuffle) -----------------------
    # layout (N_OUT/16, K/128, klane=4, nlane=16, 16B): byte =
    #   n_blk*bl_stride_n0 + k128*1024 + klane*256 + n_intra*16 + p
    bl_stride_n0 = (K // 128) * 1024
    layout_b = fx.make_layout(
        (N_OUT // 16, K // 128, 4, 16, 16),
        (bl_stride_n0, 1024, 256, 16, 1),
    )
    _w_bytes = NE * N_OUT * (K // 2)
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
        _global_i32_buffer_tiles,
    )

    # 1-dword tile view: every (col, K32i) window is a single 4B read.
    w_tiles1 = _global_i32_buffer_tiles(arg_bq, min(_w_bytes, 0xFFFFFFFF), 1)
    w_copy_atom32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(b_cache_mod), fx.Int32)

    class _Col:
        __slots__ = ("n_blk", "n_intra", "c_plain")

        def __init__(s, n_blk, n_intra, c_plain):
            s.n_blk = n_blk
            s.n_intra = n_intra
            s.c_plain = c_plain  # plain (e, N_OUT) column index for scale planes

    expert_off = e * n_out_i32

    def col(*n_terms):
        t = n_terms[0]
        for x2 in n_terms[1:]:
            t = t + x2
        c = expert_off + t + lane_mod_16
        return _Col(c // fx.Int32(16), c % fx.Int32(16), c)

    # Per (col, K32i): bytes [ld*4, ld*4+4) of the (K32i//4, K32i%4) slot ->
    # read ONE dwordx2 (8B, the whole 8-nibble... no: 4B) -- a 4B window:
    #   idx = n_blk*n0s + (K32i//4)*1024 + (K32i%4)*256 + n_intra*16 + ld*4
    def load_b_pair(kt, col):
        """Raw 8-nibble u32 per K32i of the tile (TILE_K // 32 of them)."""
        raw = []
        for h in range_constexpr(TILE_K // 32):
            k32i = kt * (TILE_K // 32) + h
            idx = fx.Int32(
                crd2idx(
                    [
                        fx.Int64(col.n_blk),
                        fx.Int64(k32i // fx.Int32(4)),
                        fx.Int64(k32i % fx.Int32(4)),
                        fx.Int64(col.n_intra),
                        fx.Int64(lane_div_16 * fx.Int32(4)),
                    ],
                    layout_b,
                )
            )
            # idx is a byte offset, 4-aligned (slot base is 16-aligned, ld*4
            # in {0,4,8,12}): a single 4B read inside the 16B slot.
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(w_copy_atom32, fx.slice(w_tiles1, (None, idx // fx.Int32(4))), r)
            v = fx.Vector(fx.memref_load_vec(r))
            raw.append(fx.Int32(v[0]))
        return raw

    # scale: plain u8 (E, N_OUT, G); one byte per (col, K32i); read as a dword
    # containing the byte and extract. shift: plain u8 (E, N_OUT).
    _sc_bytes = NE * N_OUT * G
    sc_tiles = _global_i32_buffer_tiles(arg_bscale, min(_sc_bytes, 0xFFFFFFFF), 1)
    sh_tiles = _global_i32_buffer_tiles(
        arg_bshift, min(NE * N_OUT, 0xFFFFFFFF), 1
    )

    def _u8_at(tiles, byte_idx):
        d = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)
        fx.copy(atom, fx.slice(tiles, (None, byte_idx // fx.Int32(4))), d)
        w0 = fx.Int32(fx.Vector(fx.memref_load_vec(d))[0])
        sh = (byte_idx % fx.Int32(4)) * fx.Int32(8)
        return (w0.shrui(sh)) & fx.Int32(0xFF)

    def load_cb(kt, col):
        """cb16 broadcast exponent offset per K32i of the tile."""
        sh = _u8_at(sh_tiles, col.c_plain)
        out = []
        for h in range_constexpr(TILE_K // 32):
            k32i = kt * (TILE_K // 32) + h
            sc = _u8_at(sc_tiles, col.c_plain * fx.Int32(G) + k32i)
            cb = ((sc - sh) << fx.Int32(3)) & fx.Int32(0xFF)
            cb16 = cb | (cb << fx.Int32(16))
            out.append(cb16)
        return out

    def load_b_tile(kt):
        """raw nibbles + cb, per ni, per gate/up, per K32i of the tile."""
        res = []
        for cols in (cols_gate, cols_up):
            per_col = []
            for c in cols:
                per_col.append((load_b_pair(kt, c), load_cb(kt, c)))
            res.append(per_col)
        return res

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

    def preload_a(slot):
        return [
            [lds_load_a(mi, ku, slot) for ku in range_constexpr(k_unroll)]
            for mi in range_constexpr(m_repeat)
        ]

    def compute_tile(b_tile, a_frags):
        # b_tile = [gate_per_ni, up_per_ni]; per ni: (raw_pairs, cb16s), one
        # per K32i (k_unroll of them laid out by load_b_pair: 2 per 64-K tile
        # -> k_unroll == 2 at tk64; generalize: pair h covers K32i kt*ku + h).
        b_g, b_u = b_tile
        for ni in range_constexpr(num_acc_n):
            rg, cg = b_g[ni]
            ru, cu = b_u[ni]
            for ku in range_constexpr(k_unroll):
                bg = _fp8_frag8(_fp4w_win_to_fp8x8(rg[ku], cg[ku]))
                bu = _fp8_frag8(_fp4w_win_to_fp8x8(ru[ku], cu[ku]))
                for mi in range_constexpr(m_repeat):
                    aa = _fp8_frag8(a_frags[mi][ku])
                    fx.gemm(mma_atom, acc_gate[mi][ni], aa, bg, acc_gate[mi][ni])
                    fx.gemm(mma_atom, acc_up[mi][ni], aa, bu, acc_up[mi][ni])

    # ---- main K loop (iter-17 software pipeline, a_stages deep) -------------
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

    # ---- epilog (verbatim iter-17): activation -> bf16 intermediate ----------
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


def gemm1_a8w4fp4_grid(BM, *, INTER, TILE_N, max_m_blocks):
    return int(max_m_blocks) * (INTER // TILE_N)


@functools.cache
def compile_gemm1_a8w4fp4(
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
    act="silu_clamp",
    waves_per_eu=None,
    rocm_arch,
):
    """a8w4-FP4W (fp8 A x native mxfp4 W1, in-register fp8 synth) MoE stage1.

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
    assert act in (
        "silu",
        "silu_clamp",
        "swiglu",
        "situv2",
    ), f"a8w4fp4 gemm1 act must be 'silu', 'silu_clamp', 'swiglu', or 'situv2', got {act!r}"
    _a_stages = max(1, min(a_stages, KLEN // TILE_K))
    lds_bytes = _a_stages * BM * TILE_K

    _name = f"gemm1_a8w4fp4_ne{NE}_i{INTER}_h{KLEN}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
    _act_tag = "" if act == "silu" else f"_{act}"
    _name += _act_tag
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if a_stages != 2:
        _name += f"_as{a_stages}"
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
        arg_bscale: fx.Int64,
        arg_bshift: fx.Int64,
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
            _gemm1_body_a8w4fp4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_bshift,
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
                rocm_arch=rocm_arch,
            )

    @flyc.jit
    def launch_gemm1(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_bshift: fx.Int64,
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
            arg_bscale,
            arg_bshift,
            arg_eids,
            arg_cumsum,
            arg_mind,
            arg_orow,
            arg_sa,
            arg_swcol,
            i32_M,
            i32_max_m_blocks,
            i32_grid,
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
