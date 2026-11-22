# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Decode-size (M <= 256) gate_up GEMM with the fused GeLU(tanh) * up + MX quantization epilogue, 8-wave version.

Same tiling as ``gemm_a8w8_gelu_mx_dec`` (BLOCK_M = 128 x HID = 160 hidden units = 320 interleaved weight rows, 2 x 64 =
128 workgroups) but on the 8-wave (2 x 4) pipeline of ``gemm_a8w8_8wave_gelu_mx``: four of the five 32-hidden MX groups of
the tile are split like the 256 x 256 kernel (wave column = group), the fifth ("extra") group is split over the M tiles:
a wave with wave_m = h owns the 16-row tile ``wave_n`` of A half ``h`` for the extra group and runs its MFMAs in the
phases that read A half ``h``. Every SIMD then carries the same MFMA count (20 per wave and K step).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave_gelu_mx import (
    Mfma16x16x128,
    S2RLoader,
    StoreActMX,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv

HID = 160  # hidden units per tile (5 MX groups of 32)
BLOCK_M = 128
BLOCK_N = 2 * HID
BLOCK_K = 128
WAVES_M = 2
WAVES_N = 4
N_WAVES = WAVES_M * WAVES_N
N_TILES_A = BLOCK_M // 64  # 16-row tiles per wave inside one LDS half
N_TILES_B = 2  # 32 hidden columns of a half per wave column
LDS_BLOCK_M = BLOCK_M // 2
LDS_BLOCK_N = BLOCK_N // 2
ROWS_PER_STEP = 8 * N_WAVES  # one 1 KiB piece per wave
N_LDS_STEPS_A = -(-LDS_BLOCK_M // ROWS_PER_STEP)
N_LDS_STEPS_B = -(-LDS_BLOCK_N // ROWS_PER_STEP)  # 160 rows = 2.5 steps: the last step loads 32 spare rows
N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
A_LDS_SIZE = N_LDS_STEPS_A * ROWS_PER_STEP * BLOCK_K
B_LDS_SIZE = N_LDS_STEPS_B * ROWS_PER_STEP * BLOCK_K
EXTRA_GROUP = WAVES_N  # index of the 5th hidden group of the tile
assert N_TILES_B * 16 * WAVES_N + 32 == HID


class StoreActMXDec(StoreActMX):
    """StoreActMX for HID = 160: the up half of the interleaved scale vector starts HID columns after the gate half and the
    E8M0 bytes are addressed by the global hidden group of the wave."""

    def prefetch(self, base_row, half_rows, wcol0):
        l16 = self.lane_id % 16
        sa = [
            [self._sa_scalar(base_row + h * half_rows + ti * 16 + l16) for ti in range_constexpr(self.n_tiles_a)]
            for h in range_constexpr(2)
        ]
        sb = [self._sb_vec4(wcol0), self._sb_vec4(wcol0 + 4), self._sb_vec4(wcol0 + HID), self._sb_vec4(wcol0 + HID + 4)]
        self.pre = (sa, sb)

    def _scale_byte(self, sel):
        sel8 = Vec.filled(1, fx.Int32(sel), fx.Int32).bitcast(fx.Int8)
        reg1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int8)
        fx.memref_store_vec(Vec.from_elements([sel8[0]], dtype=fx.Int8), reg1)
        return reg1

    def _scale_index(self, n1, group, parity):
        step = group // 4
        return ((n1 * self.k1_stride + step // 2) * 64 + (group % 4) * 16 + self.lane_id % 16) * 4 + (step % 2) * 2 + parity

    def epi_scales(self, base_row, group):
        """base_row: first row (multiple of 32) of this half's 32 rows of the wave; group: global 32-hidden group index."""
        grp = self.lane_id // 16
        es = self.es
        sel = es[0]
        for ti in range_constexpr(1, self.n_tiles_a):
            sel = (grp == fx.Int32(ti)).select(es[ti], sel)
        sc_i = self._scale_index(base_row // 32, group, grp % 2)
        # lanes of row-tile groups past n_tiles_a (grp >= 2) have no tile: send their byte out of range
        oob = ((self.c_rows + 255) // 256) * 256 * self.n_sc
        sc_i = (grp < fx.Int32(self.n_tiles_a)).select(sc_i, oob)
        fx.copy(self.sc_atom, self._scale_byte(sel), fx.slice(self.sc_div, (None, sc_i)))

    def epi_scales_one(self, base_row, group):
        """One 16-row tile (base_row a multiple of 16): only the lanes of row-tile group 0 hold its rows."""
        grp = self.lane_id // 16
        sc_i = self._scale_index(base_row // 32, group, (base_row // 16) % 2)
        oob = ((self.c_rows + 255) // 256) * 256 * self.n_sc
        sc_i = (grp == fx.Int32(0)).select(sc_i, oob)
        fx.copy(self.sc_atom, self._scale_byte(self.es[0]), fx.slice(self.sc_div, (None, sc_i)))


def mfma_extra(a, b, c):
    """The extra group's 1 x 2 scaled MFMAs (A tile = 16 rows, B = two 16-column tiles), no barrier; the helper object is
    created per call because the AST rewriter turns every object used inside a runtime ``if`` into carried state."""
    m = Mfma16x16x128(1, 2)
    return [m.one([a], b, c, 0, j) for j in range_constexpr(2)]


def _load_extra_a(a_cur0, wave_m, wave_n, lane_id):
    """A fragment (one 16-row tile) of the extra group's row tile: row tile ``wave_n`` of A half ``wave_m``; the two halves
    sit A_LDS_SIZE apart (A_lds_cur_0 / A_lds_cur_1 and A_lds_next_0 / A_lds_next_1 are adjacent)."""
    loader = S2RLoader(0, 1)
    row = wave_n * 16 + lane_id % 16
    halves = []
    for step in range_constexpr(2):
        col = (lane_id // 16) * 16 + step * 64
        row_swz, col_swz = swizzle_128(row, col)
        offset = row_swz * 128 + col_swz + wave_m * A_LDS_SIZE
        halves.append(loader.load_one(a_cur0, offset))
    return pack_i32x4_i32x8(halves[0], halves[1])


def compile_gate_up_gelu_mx_dec8(
    *,
    K: int,
    waves_per_eu: int = 2,
):
    assert K % BLOCK_K == 0 and K >= 2 * BLOCK_K
    K_ITERS = K // BLOCK_K
    N_ACCUMS = N_TILES_A * N_TILES_B

    _kname = f"flydsl_gelumx_dec8_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_F8_w{N_WAVES}_{waves_per_eu}_k{K}"

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, A_LDS_SIZE, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, A_LDS_SIZE, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, A_LDS_SIZE, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, A_LDS_SIZE, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, B_LDS_SIZE, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, B_LDS_SIZE, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, B_LDS_SIZE, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, B_LDS_SIZE, 16]

    @flyc.kernel(name=_kname, known_block_size=[N_WAVES * 64, 1, 1])
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
        wave_m = wave_id // WAVES_N
        wave_n = wave_id % WAVES_N

        # Hardware dispatches workgroup i to XCD i % 8: give each XCD a run of N tiles and keep the two M tiles that
        # share a weight tile on the same XCD.
        wgid = fx.block_idx.x
        n_tiles = c_n // BLOCK_N
        xcd = wgid % 8
        loc = wgid // 8
        per_xcd = n_tiles // 8
        m_tiles = ceildiv(c_m, BLOCK_M)
        block_m = loc % m_tiles
        block_n = xcd * per_xcd + loc // m_tiles

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = 2 * 1024
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=True)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        bx_s2r = S2RLoader(EXTRA_GROUP, N_TILES_B)  # rows 128..159 of each B half
        store_c = StoreActMXDec(A_scale, B_scale, C, SC, c_m, c_n, mfma.idx, N_TILES_A)
        store_x = StoreActMXDec(A_scale, B_scale, C, SC, c_m, c_n, lambda ti, j: ti * 2 + j, 1)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS
        cx_g = [mfma.zero_value] * 2
        cx_u = [mfma.zero_value] * 2

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
            b0_frag = b_s2r.load(b_cur0, preshuffled=True, permute=True)
            bx0_frag = bx_s2r.load(b_cur0, preshuffled=True, permute=True)
            a0_frag = a_s2r.load(a_cur0)
            ax_frag = _load_extra_a(a_cur0, wave_m, wave_n, lane_id)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            if wave_m == 0:
                cx_g = mfma_extra(ax_frag, bx0_frag, cx_g)

            b1_frag = b_s2r.load(b_cur1, preshuffled=True, permute=True)
            bx1_frag = bx_s2r.load(b_cur1, preshuffled=True, permute=True)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            if wave_m == 0:
                cx_u = mfma_extra(ax_frag, bx1_frag, cx_u)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            if wave_m == 1:
                cx_g = mfma_extra(ax_frag, bx0_frag, cx_g)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            if wave_m == 1:
                cx_u = mfma_extra(ax_frag, bx1_frag, cx_u)

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=True, permute=True)
        bx0_frag = bx_s2r.load(b_cur0, preshuffled=True, permute=True)
        a0_frag = a_s2r.load(a_cur0)
        ax_frag = _load_extra_a(a_cur0, wave_m, wave_n, lane_id)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        if wave_m == 0:
            cx_g = mfma_extra(ax_frag, bx0_frag, cx_g)

        b1_frag = b_s2r.load(b_cur1, preshuffled=True, permute=True)
        bx1_frag = bx_s2r.load(b_cur1, preshuffled=True, permute=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        if wave_m == 0:
            cx_u = mfma_extra(ax_frag, bx1_frag, cx_u)

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        if wave_m == 1:
            cx_g = mfma_extra(ax_frag, bx0_frag, cx_g)

        b0_frag = b_s2r.load(b_next0, preshuffled=True, permute=True)
        bx0_frag = bx_s2r.load(b_next0, preshuffled=True, permute=True)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        if wave_m == 1:
            cx_u = mfma_extra(ax_frag, bx1_frag, cx_u)
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        wcol0 = block_n * BLOCK_N + wave_n_offset + (lane_id // 16) * 8
        hid0 = block_n * HID + wave_n_offset + (lane_id // 16) * 8
        group = block_n * (WAVES_N + 1) + wave_n
        x_row = block_m * BLOCK_M + wave_m * LDS_BLOCK_M + wave_n * 16
        x_wcol0 = block_n * BLOCK_N + 32 * EXTRA_GROUP + (lane_id // 16) * 8
        x_hid0 = block_n * HID + 32 * EXTRA_GROUP + (lane_id // 16) * 8
        x_group = block_n * (WAVES_N + 1) + EXTRA_GROUP
        a0_frag = a_s2r.load(a_cur0)
        ax_frag = _load_extra_a(a_cur0, wave_m, wave_n, lane_id)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        if wave_m == 0:
            cx_g = mfma_extra(ax_frag, bx0_frag, cx_g)

        b1_frag = b_s2r.load(b_cur1, preshuffled=True, permute=True)
        bx1_frag = bx_s2r.load(b_cur1, preshuffled=True, permute=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        if wave_m == 0:
            cx_u = mfma_extra(ax_frag, bx1_frag, cx_u)
        store_c.prefetch(base_row, LDS_BLOCK_M, wcol0)
        store_x.prefetch(x_row, 0, x_wcol0)
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        store_c.begin(0)
        store_x.begin(0)
        c10 = list(c10_frag)
        c11 = list(c11_frag)

        def mfma_row(c, b_frag, ti):
            for j in range_constexpr(mfma.n_tiles_b):
                c[mfma.idx(ti, j)] = mfma.one(a1_frag, b_frag, c, ti, j)

        for ti in range_constexpr(N_TILES_A):
            mfma_row(c10, b0_frag, ti)
        if wave_m == 1:
            cx_g = mfma_extra(ax_frag, bx0_frag, cx_g)
        store_c.epi_one(0, c00_frag, c01_frag, 0, base_row + 0, hid0)
        for ti in range_constexpr(N_TILES_A):
            mfma_row(c11, b1_frag, ti)
        if wave_m == 1:
            cx_u = mfma_extra(ax_frag, bx1_frag, cx_u)
        for ti in range_constexpr(1, N_TILES_A):
            store_c.epi_one(ti, c00_frag, c01_frag, 0, base_row + 0, hid0)
        store_c.epi_scales(base_row + 0, group)
        rocdl.s_barrier()

        store_c.begin(1)
        for ti in range_constexpr(N_TILES_A):
            store_c.epi_one(ti, c10, c11, 1, base_row + LDS_BLOCK_M, hid0)
        store_c.epi_scales(base_row + LDS_BLOCK_M, group)

        store_x.epi_one(0, cx_g, cx_u, 0, x_row, x_hid0)
        store_x.epi_scales_one(x_row, x_group)

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
                "rocdl.flat_work_group_size": f"{N_WAVES * 64},{N_WAVES * 64}",
            },
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    return launch_gemm
