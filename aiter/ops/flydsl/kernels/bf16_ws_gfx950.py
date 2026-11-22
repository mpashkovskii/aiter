# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Weight-streaming bf16 GEMM for decode-size M (<= 256) on gfx950.

``Out[m, n] = bf16(sum_k X[m, k] * W[n, k])`` with X ``[M, K]`` and W ``[N, K]`` row-major bf16 (the nn.Linear layout, no
pre-shuffle), fp32 accumulation.

One workgroup computes an ``M_REP * 16`` x ``N_REP * 16`` output tile over the whole K: 8 waves, wave ``w`` owns the K range
``[w * K / 8, (w + 1) * K / 8)``, streams its operands straight from global memory into registers (no LDS pipeline, no barrier in
the main loop; ``PF`` blocks of loads stay in flight) and accumulates the tile with 16x16x32 bf16 MFMAs (the weight is the A
operand, so a lane's accumulator holds 4 consecutive output columns of one row). The 8 partial tiles are reduced through LDS in
``ROUNDS`` rounds and stored as 8-byte bf16 vectors. K is permuted inside a wave step (a lane reads ``KC`` contiguous elements
per row instead of 8) the same way for both operands, which keeps the sum and gives 4-lane-contiguous 32..64 B segments.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Constexpr, Float32, Int32, Int8, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import ceildiv

N_WAVES = 8


@flyc.jit
def launch_bf16_ws(
    arg_c: fx.Pointer,
    arg_x: fx.Pointer,
    arg_w: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    M_REP: Constexpr[int] = 4,
    N_REP: Constexpr[int] = 5,
    KC: Constexpr[int] = 16,
    PF: Constexpr[int] = 2,
    ROUNDS: Constexpr[int] = 2,
    DBG: Constexpr[int] = 0,
):
    M_TILE = M_REP * 16
    N_TILE = N_REP * 16
    KW = K // N_WAVES  # K per wave
    BLK = 4 * KC  # K per wave step
    NB = KW // BLK
    J = KC // 8
    TILES = M_REP * N_REP
    TR = TILES // ROUNDS  # tiles per reduction round
    PARTIAL_BYTES = TR * 64 * 16
    assert KW * N_WAVES == K and NB * BLK == KW, "K must split into 8 waves of whole steps"
    assert N % N_TILE == 0 and TILES % ROUNDS == 0
    n_tiles = N // N_TILE
    kernel_name = f"bf16_ws_{M_TILE}x{N_TILE}_n{N}_k{K}_kc{KC}_pf{PF}_r{ROUNDS}_d{DBG}"

    @fx.struct
    class Arena:
        data: fx.Array[Int8, N_WAVES * PARTIAL_BYTES, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[N_WAVES * 64, 1, 1])
    def kernel_ws(
        arg_c: fx.Int64,
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        i32_m: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane16 = lane % 16
        kgrp = lane // 16
        wgid = fx.Int32(fx.block_idx.x)
        m_tiles = ceildiv(i32_m, M_TILE)
        if const_expr(n_tiles % 8 == 0):
            # Hardware dispatches workgroup i to XCD i % 8: a run of N tiles per XCD, the M tiles of one weight tile together.
            xcd = wgid % 8
            loc = wgid // 8
            mt = loc % m_tiles
            nt = xcd * (n_tiles // 8) + loc // m_tiles
        else:
            mt = wgid % m_tiles
            nt = wgid // m_tiles

        def body():
            def _buffer(base, elem, align, nbytes):
                ptr_ty = fx.PointerType.get(elem, address_space=fx.AddressSpace.Global, alignment=align)
                view = fx.make_view(fx.inttoptr(ptr_ty, base), fx.make_layout(1 << 30, 1))
                buf = fx.rocdl.make_buffer_tensor(fx.Tensor(view), max_size=False, num_records_bytes=nbytes)
                return fx.logical_divide(buf, fx.make_layout(1, 1))

            x_words = _buffer(arg_x, T.i32, 16, fx.Int64(i32_m) * (K * 2))
            w_words = _buffer(arg_w, T.i32, 16, fx.Int64(N) * (K * 2))
            load16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
            atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

            def _load16(src, off_i32):
                frag = fx.make_rmem_tensor(4, Int32)
                fx.copy(load16, fx.slice(src, (None, off_i32)), frag)
                return Vec(frag.load())

            # element offsets (K elements per row), in i32 words
            k_lane = wave * KW + kgrp * KC
            w_row0 = nt * N_TILE + lane16
            x_row0 = mt * M_TILE + lane16

            def load_block(b):
                """Weight (A operand) and activation (B operand) fragments of step b: lists [rep][j] of 4 x i32."""
                ws = []
                xs = []
                koff = k_lane + b * BLK
                for ni in range_constexpr(N_REP):
                    row = w_row0 + ni * 16
                    if const_expr(DBG == 3):
                        row = x_row0
                    if const_expr(DBG in (4, 5, 7, 8)):
                        ws.append([_load16(w_words, (nt * (N_TILE * K) + ((wave * NB + b) * N_REP * J + ni * J + j) * 512 + lane * 8) // 2 * 2 // 2 * 1 // 1) for j in range_constexpr(J)])
                    else:
                        ws.append([_load16(w_words, (row * K + koff + j * 8) // 2) for j in range_constexpr(J)])
                for mi in range_constexpr(M_REP):
                    row = x_row0 + mi * 16
                    if const_expr(DBG == 2):
                        row = w_row0
                    if const_expr(DBG in (4, 6, 7, 8)):
                        xs.append([_load16(x_words, (mt * (M_TILE * K) + ((wave * NB + b) * M_REP * J + mi * J + j) * 512 + lane * 8) // 2 * 2 // 2 * 1 // 1) for j in range_constexpr(J)])
                    else:
                        xs.append([_load16(x_words, (row * K + koff + j * 8) // 2) for j in range_constexpr(J)])
                return ws, xs

            accs = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(TILES)]
            for acc in accs:
                acc.store(Vec.filled(4, 0.0, Float32))

            def as_bf16(v):
                frag = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                frag.store(v.bitcast(BFloat16))
                return frag

            dummy = Vec.filled(4, 0, Int32)
            q = [load_block(b) for b in range_constexpr(min(PF, NB))]
            for b in range_constexpr(NB):
                if const_expr(b + PF < NB):
                    q.append(load_block(b + PF))
                ws, xs = q.pop(0)
                rocdl.sched_barrier(0)
                if const_expr(DBG in (1, 7)):
                    for j in range_constexpr(J):
                        for ni in range_constexpr(N_REP):
                            dummy = dummy + ws[ni][j]
                        for mi in range_constexpr(M_REP):
                            dummy = dummy + xs[mi][j]
                    continue
                for j in range_constexpr(J):
                    wf = [as_bf16(ws[ni][j]) for ni in range_constexpr(N_REP)]
                    for mi in range_constexpr(M_REP):
                        xf = as_bf16(xs[mi][j])
                        for ni in range_constexpr(N_REP):
                            acc = accs[mi * N_REP + ni]
                            fx.gemm(atom, acc, wf[ni], xf, acc)

            if const_expr(DBG in (1, 7)):
                accs[0].store(dummy.bitcast(Float32))
            # ---- reduction of the 8 K-split partials through LDS, ROUNDS rounds of TR tiles, then bf16 stores ----
            lds = fx.SharedAllocator().allocate(Arena).peek()
            lds_i32 = fx.recast_iter(Int32, lds.data.ptr)
            rw16 = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
            out_words = _buffer(arg_c, T.i32, 8, fx.Int64(i32_m) * (N * 2))
            st64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), 32)
            rows_left = i32_m - mt * M_TILE
            if const_expr(DBG == 8):
                for f in range_constexpr(TILES):
                    mi = f // N_REP
                    ni = f % N_REP
                    col = nt * N_TILE + ni * 16 + kgrp * 4
                    row = mt * M_TILE + mi * 16 + lane16
                    pk = fx.make_rmem_tensor(2, Int32)
                    pk.store(Vec(accs[f].load()).to(BFloat16).bitcast(Int32))
                    fx.copy(st64, pk, fx.slice(out_words, (None, (row * N + col) // 2)))
            for rnd in range_constexpr(0 if DBG == 8 else ROUNDS):
                gpu.barrier()
                for f in range_constexpr(TR):
                    frag = fx.make_rmem_tensor(4, Int32)
                    frag.store(Vec(accs[rnd * TR + f].load()).bitcast(Int32))
                    dst = fx.make_view(fx.add_offset(lds_i32, (wave * PARTIAL_BYTES + (f * 64 + lane) * 16) // 4), fx.make_layout(4, 1))
                    fx.copy(rw16, frag, dst)
                gpu.barrier()
                for j in range_constexpr((TR + N_WAVES - 1) // N_WAVES):
                    f_idx = wave + j * N_WAVES
                    if f_idx < TR:
                        tot = Vec.filled(4, 0.0, Float32)
                        for sw in range_constexpr(N_WAVES):
                            frag = fx.make_rmem_tensor(4, Int32)
                            src = fx.make_view(fx.add_offset(lds_i32, (sw * PARTIAL_BYTES + (f_idx * 64 + lane) * 16) // 4), fx.make_layout(4, 1))
                            fx.copy(rw16, src, frag)
                            tot = tot + Vec(frag.load()).bitcast(Float32)
                        tile = rnd * TR + f_idx
                        mi = tile // N_REP
                        ni = tile % N_REP
                        col = nt * N_TILE + ni * 16 + kgrp * 4
                        row = mt * M_TILE + mi * 16 + lane16
                        if row < i32_m:
                            vals = tot.to(BFloat16)
                            pk = fx.make_rmem_tensor(2, Int32)
                            pk.store(vals.bitcast(Int32))
                            fx.copy(st64, pk, fx.slice(out_words, (None, (row * N + col) // 2)))

        if mt * M_TILE < i32_m:
            body()

    grid = (ceildiv(i32_m, M_TILE) * n_tiles, 1, 1)
    kernel_ws(
        fx.Int64(fx.ptrtoint(arg_c)),
        fx.Int64(fx.ptrtoint(arg_x)),
        fx.Int64(fx.ptrtoint(arg_w)),
        i32_m,
    ).launch(grid=grid, block=(N_WAVES * 64, 1, 1), stream=stream)
