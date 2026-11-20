# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""8-wave FP8 GEMM (per-token x per-channel scales) with a fused GeLU(tanh) * up epilogue for AMD CDNA4 (gfx950).

Derived from ``gemm_a8w8_8wave``. The weight rows are interleaved so that every 256-row B tile holds 128 gate rows
followed by the 128 up rows of the same hidden units; the MFMA operands are swapped (weights as the A operand) and
the weight rows are read in a permuted order so that one lane holds 8 consecutive hidden columns of the gate in
(tile0, tile1) of the c00/c10 accumulators and of the up in c01/c11. The epilogue computes ``gelu_tanh(gate) * up``,
quantizes it per (row, 32 hidden columns) to FP8 e4m3 with a power-of-two (E8M0, rounded up) scale and writes the
FP8 activation ``[M, N/2]`` and the E8M0 scales in the ``shuffle_scale_w4`` layout (rows padded to 256, flat
``ceil(M/256)*256*N/64`` bytes) that the pre-shuffled-scale MX (1x32) down projection reads without LDS staging.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    _xcd_swizzle_any,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw


class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False, permute=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            if const_expr(permute):
                row = self.wave_idx * (self.n_tiles * 16) + ((self.lane_id % 16) // 4) * 8 + i * 4 + self.lane_id % 4
            else:
                row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)


class StoreActMX:
    """Fused epilogue: act = bf16(bf16(gelu_tanh(gate)) * up), quantized per (row, 32 hidden cols) to fp8 e4m3 with an
    e8m0 (power-of-two, rounded up) scale. Writes fp8 [M, N/2] and e8m0 bytes [M, N/64]."""

    def __init__(self, A_scale, B_scale, C, SC, c_rows, c_cols, c_idx_fn, n_tiles_a):
        self.c_rows = c_rows
        self.out_cols = c_cols // 2
        self.n_sc = c_cols // 64
        self.k1_stride = c_cols // 512  # K/256 of the down projection (K = N/2)
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_rows * (c_cols // 2))
        gSC = fx.rocdl.make_buffer_tensor(SC, max_size=False, num_records_bytes=((c_rows + 255) // 256) * 256 * (c_cols // 64))
        gSA = fx.rocdl.make_buffer_tensor(A_scale, max_size=False, num_records_bytes=c_rows * 4)
        gSB = fx.rocdl.make_buffer_tensor(B_scale, max_size=False, num_records_bytes=c_cols * 4)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.sc_div = fx.logical_divide(gSC, fx.make_layout(1, 1))
        self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
        self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(NT_STORE), fx.Int8)
        self.sc_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)
        self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.reg_i8_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Int8)
        self.reg_i8_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int8)

    def _sb_vec4(self, col):
        fx.copy(self.scale_atom_4, fx.slice(self.sb_div, (None, fx.Int32(col))), self.reg_f32_4)
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _sa_scalar(self, row):
        fx.copy(self.scale_atom_1, fx.slice(self.sa_div, (None, fx.Int32(row))), self.reg_f32_1)
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _xor_max(self, v_i32, off):
        src = (self.lane_id ^ fx.Int32(off)) * fx.Int32(4)
        r = fx.Int32(rocdl.ds_bpermute(T.i32, src, v_i32))
        return (r > v_i32).select(r, v_i32)

    def store(self, c_gate, c_up, base_row, wcol0, hid_col0, sc_k1, sc_kp, sc_l0):
        l16 = self.lane_id % 16
        sbg0 = self._sb_vec4(wcol0)
        sbg1 = self._sb_vec4(wcol0 + 4)
        sbu0 = self._sb_vec4(wcol0 + 128)
        sbu1 = self._sb_vec4(wcol0 + 132)
        es = []
        K1 = fx.Float32(-2.0 * 1.4426950408889634 * 0.7978845608028654)
        K2 = fx.Float32(-2.0 * 1.4426950408889634 * 0.7978845608028654 * 0.044715)
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + l16
            sa = Vec.filled(4, self._sa_scalar(row), fx.Float32)
            g = (Vec(c_gate[self.c_idx_fn(ti, 0)]) * (sa * sbg0)).shuffle(Vec(c_gate[self.c_idx_fn(ti, 1)]) * (sa * sbg1), list(range(8)))
            u = (Vec(c_up[self.c_idx_fn(ti, 0)]) * (sa * sbu0)).shuffle(Vec(c_up[self.c_idx_fn(ti, 1)]) * (sa * sbu1), list(range(8)))
            t = g * g
            z = g * (t * K2 + K1)
            ee = [fx.Float32(rocdl.exp2(T.f32, _raw(z[i]))) for i in range_constexpr(8)]
            d = Vec.from_elements(ee, dtype=fx.Float32) + fx.Float32(1.0)
            rr = [fx.Float32(rocdl.rcp(T.f32, _raw(d[i]))) for i in range_constexpr(8)]
            act = g * Vec.from_elements(rr, dtype=fx.Float32) * u
            m = fx.max(act[0], -act[0])
            for i in range_constexpr(1, 8):
                m = fx.max(m, fx.max(act[i], -act[i]))
            mi = Vec.filled(1, m, fx.Float32).bitcast(fx.Int32)[0]
            mi = self._xor_max(mi, 16)
            mi = self._xor_max(mi, 32)
            rb = Vec.filled(1, Vec.filled(1, mi, fx.Int32).bitcast(fx.Float32)[0] * fx.Float32(1.0 / 448.0), fx.Float32).bitcast(fx.Int32)[0]
            eb = (rb + fx.Int32(0x7FFFFF)) >> fx.Int32(23)
            eb = (eb < fx.Int32(1)).select(fx.Int32(1), eb)
            eb = (eb > fx.Int32(254)).select(fx.Int32(254), eb)
            inv = Vec.filled(1, (fx.Int32(254) - eb) << fx.Int32(23), fx.Int32).bitcast(fx.Float32)[0]
            q = act * inv
            w = []
            for p in range_constexpr(2):
                lo = rocdl.cvt_pk_fp8_f32(T.i32, q[4 * p], q[4 * p + 1], fx.Int32(0), False)
                w.append(rocdl.cvt_pk_fp8_f32(T.i32, q[4 * p + 2], q[4 * p + 3], lo, True))
            packed = fx.Vector.from_elements([fx.Int32(x) for x in w], dtype=fx.Int32).bitcast(fx.Int8)
            fx.memref_store_vec(packed, self.reg_i8_8)
            c_index = row * self.out_cols + hid_col0
            fx.copy(self.out_atom_8, self.reg_i8_8, fx.slice(self.c_div, (None, c_index)))
            es.append(eb)
        grp = self.lane_id // 16
        sel = es[0]
        for ti in range_constexpr(1, self.n_tiles_a):
            sel = (grp == fx.Int32(ti)).select(es[ti], sel)
        sel8 = Vec.filled(1, fx.Int32(sel), fx.Int32).bitcast(fx.Int8)
        fx.memref_store_vec(Vec.from_elements([sel8[0]], dtype=fx.Int8), self.reg_i8_1)
        n1 = base_row // 32 + grp // 2
        sc_i = ((n1 * self.k1_stride + sc_k1) * 64 + sc_l0 + l16) * 4 + sc_kp * 2 + (grp % 2)
        fx.copy(self.sc_atom, self.reg_i8_1, fx.slice(self.sc_div, (None, sc_i)))


class Mfma16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
        )
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _do_mma(self, a, b, c):
        a_frag = self._make_operand_frag(a)
        b_frag = self._make_operand_frag(b)
        c_frag = self._make_accum_frag(c)
        fx.gemm(self.atom, c_frag, a_frag, b_frag, c_frag)
        return c_frag.load().ir_value()

    def call(self, a, b, c, *, set_prio=True):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [
            self._make_operand_frag(a[idx]) for idx in range_constexpr(self.n_tiles_a)
        ]
        b_frags = [
            self._make_operand_frag(b[idx]) for idx in range_constexpr(self.n_tiles_b)
        ]
        c_frags = [
            self._make_accum_frag(c[idx])
            for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(self.atom, cf, b_frags[j], a_frags[i], cf)
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [
            c_frags[idx].load().ir_value()
            for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b

        return self._do_mma(a[i], b[j], c[self.idx(i, j)])


NT_STORE = 2  # buffer cache policy bit "nt": streaming stores


def compile_fp8_gemm_8w_gelu_mx(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = False,
    waves_per_eu: int = 2,
    xcd_swizzle: int = 0,
):
    BLOCK_K = 128

    assert (
        BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    )
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    # half size
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    assert b_preshuffled, "the fused epilogue needs the 16x16 pre-shuffled, gate/up-interleaved weight"
    _kname = (
        f"flydsl_gelumx_8w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_F8_"
        f"{waves_per_eu}x{xcd_swizzle}_k{K}"
    )

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(name=_kname, known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        SC: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        if const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
        )
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled
        )

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = StoreActMX(A_scale, B_scale, C, SC, c_m, c_n, mfma.idx, N_TILES_A)

        # 2x2 config of 4x2 (instead of 4x4 in 4wave) 16x16 sub-tiles
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled, permute=True)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled, permute=True)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled, permute=True)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled, permute=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        # Main loop prefetches a_next1 one step behind; issue the final
        # K_ITERS - 1 tile here, otherwise c10 / c11 read stale A1 data.
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled, permute=True)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        wcol0 = block_n * BLOCK_N + wave_n_offset + (lane_id // 16) * 8
        hid0 = block_n * (BLOCK_N // 2) + wave_n_offset + (lane_id // 16) * 8
        sck1 = block_n // 2
        sckp = block_n % 2
        scl0 = wave_n * 16
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled, permute=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, set_prio=False)
        store_c.store(c00_frag, c01_frag, base_row + 0, wcol0, hid0, sck1, sckp, scl0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, set_prio=False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        store_c.store(c10_frag, c11_frag, base_row + LDS_BLOCK_M, wcol0, hid0, sck1, sckp, scl0)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        SC: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            SC,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
