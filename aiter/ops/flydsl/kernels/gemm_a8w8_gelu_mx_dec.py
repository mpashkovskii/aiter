# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Decode-size (M <= 256) variant of the gate_up GEMM with the fused GeLU(tanh) * up + MX quantization epilogue.

``gemm_a8w8_8wave_gelu_mx`` tiles 256 x 256 (128 hidden units per tile): at M = 256 and N = 20480 that is 80 workgroups
on a 128-CU GPU. This variant tiles BLOCK_M = 128 by HID = 160 hidden units (BLOCK_N = 320 interleaved weight rows: 160
gate rows followed by the 160 up rows of the same hidden units), i.e. 2 x 64 = 128 workgroups of 10 waves (2 in M x 5 in
N, one 32-hidden MX group per wave column), one per CU. The K pipeline, the MFMA operand layout, the epilogue math and the
FP8 / E8M0 outputs are those of the 256 x 256 kernel; only the tile geometry, the scale-byte addressing (global hidden
group G = block_n * 5 + wave_n -> K step G // 4, lane block G % 4 of the ``shuffle_scale_w4`` layout) and the interleave
granularity (HID = 160) differ.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
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
WAVES_N = 5
N_WAVES = WAVES_M * WAVES_N
N_TILES_A = BLOCK_M // 64  # 16-row tiles per wave inside one LDS half
N_TILES_B = 2  # 32 hidden columns of a half per wave column
LDS_BLOCK_M = BLOCK_M // 2
LDS_BLOCK_N = BLOCK_N // 2
ROWS_PER_STEP = 8 * N_WAVES  # one 1 KiB piece per wave
N_LDS_STEPS_A = -(-LDS_BLOCK_M // ROWS_PER_STEP)
N_LDS_STEPS_B = LDS_BLOCK_N // ROWS_PER_STEP
N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
A_LDS_SIZE = N_LDS_STEPS_A * ROWS_PER_STEP * BLOCK_K
B_LDS_SIZE = N_LDS_STEPS_B * ROWS_PER_STEP * BLOCK_K
assert LDS_BLOCK_N % ROWS_PER_STEP == 0 and N_TILES_B * 16 * WAVES_N == HID


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

    def epi_scales(self, base_row, group):
        """base_row: first row of this half's 32 rows of the wave; group: global 32-hidden group index."""
        l16 = self.lane_id % 16
        grp = self.lane_id // 16
        es = self.es
        sel = es[0]
        for ti in range_constexpr(1, self.n_tiles_a):
            sel = (grp == fx.Int32(ti)).select(es[ti], sel)
        sel8 = Vec.filled(1, fx.Int32(sel), fx.Int32).bitcast(fx.Int8)
        reg1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int8)
        fx.memref_store_vec(Vec.from_elements([sel8[0]], dtype=fx.Int8), reg1)
        step = group // 4
        sc_k1 = step // 2
        sc_kp = step % 2
        sc_l0 = (group % 4) * 16
        n1 = base_row // 32
        sc_i = ((n1 * self.k1_stride + sc_k1) * 64 + sc_l0 + l16) * 4 + sc_kp * 2 + (grp % 2)
        # lanes of row-tile groups past n_tiles_a (grp >= 2) have no tile: send their byte out of range
        oob = ((self.c_rows + 255) // 256) * 256 * self.n_sc
        sc_i = (grp < fx.Int32(self.n_tiles_a)).select(sc_i, oob)
        fx.copy(self.sc_atom, reg1, fx.slice(self.sc_div, (None, sc_i)))


def compile_gate_up_gelu_mx_dec(
    *,
    K: int,
    waves_per_eu: int = 3,
):
    assert K % BLOCK_K == 0 and K >= 2 * BLOCK_K
    K_ITERS = K // BLOCK_K
    N_ACCUMS = N_TILES_A * N_TILES_B

    _kname = f"flydsl_gelumx_dec_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_F8_F8_F8_w{N_WAVES}_{waves_per_eu}_k{K}"

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
        block_m = loc % 2
        block_n = xcd * per_xcd + loc // 2

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
        store_c = StoreActMXDec(A_scale, B_scale, C, SC, c_m, c_n, mfma.idx, N_TILES_A)

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
            b0_frag = b_s2r.load(b_cur0, preshuffled=True, permute=True)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1, preshuffled=True, permute=True)
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

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=True, permute=True)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=True, permute=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0, preshuffled=True, permute=True)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
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
        group = block_n * WAVES_N + wave_n
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=True, permute=True)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        store_c.prefetch(base_row, LDS_BLOCK_M, wcol0)
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        store_c.begin(0)
        c10 = list(c10_frag)
        c11 = list(c11_frag)

        def mfma_row(c, b_frag, ti):
            for j in range_constexpr(mfma.n_tiles_b):
                c[mfma.idx(ti, j)] = mfma.one(a1_frag, b_frag, c, ti, j)

        for ti in range_constexpr(N_TILES_A):
            mfma_row(c10, b0_frag, ti)
        store_c.epi_one(0, c00_frag, c01_frag, 0, base_row + 0, hid0)
        for ti in range_constexpr(N_TILES_A):
            mfma_row(c11, b1_frag, ti)
        for ti in range_constexpr(1, N_TILES_A):
            store_c.epi_one(ti, c00_frag, c01_frag, 0, base_row + 0, hid0)
        store_c.epi_scales(base_row + 0, group)
        rocdl.s_barrier()

        store_c.begin(1)
        for ti in range_constexpr(N_TILES_A):
            store_c.epi_one(ti, c10, c11, 1, base_row + LDS_BLOCK_M, hid0)
        store_c.epi_scales(base_row + LDS_BLOCK_M, group)

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
