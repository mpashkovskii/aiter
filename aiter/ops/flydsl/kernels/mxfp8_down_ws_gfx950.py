# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Weight-streaming MXFP8 down projection for decode-size M (<= 256) on gfx950.

``Out[m, n] = bf16(chan[n] * sum_k 2**(xs[m, k // 32] - 127) * X[m, k] * W[n, k])`` with X fp8 in the MFMA tile layout (see
below), W fp8 16x16 pre-shuffled, xs row-major E8M0 bytes ``[M, K / 32]``, chan fp32 ``[N]``.

One workgroup computes a 64 x 80 output tile over the whole K: 8 waves, wave ``w`` owns K steps ``w, w + 8, ..., w + 72`` (of 128),
streams its operands straight from global memory into registers (no LDS pipeline, no barrier in the main loop), accumulates
4 x 5 16x16 tiles with the scaled MFMA, and the 8 partial tiles are reduced through LDS (in two rounds of 10 tiles; wave ``w``
sums tiles ``w`` and ``w + 8`` of a round) and stored as bf16. 128 workgroups = one per CU for N = 2560, M = 256.

Activation scales: E8M0 bytes in the "mq" layout: for row ``r`` (row group ``r // 64``, row tile ``mi = r % 64 // 16``, ``r16 = r % 16``)
and hidden group ``G`` (K step ``ks = G // 4``, ``g = G % 4``) at ``(((r // 64 * K_STEPS + ks) * 4 + g) * 16 + r16) * 4 + mi``: the
lane (``r16``, ``g``) reads one dword per K step holding the scales of the 4 row tiles and the MFMA of row tile ``mi`` picks
byte ``mi`` with its opsel.

X tile layout (what the decode gate_up epilogue writes with ``act_layout="tiles"``): row tile ``r // 16``, K step ``c // 128``,
a 2 KiB block per (row tile, K step); inside it lane ``l`` = ``16 g + r % 16`` owns the 16 bytes at ``l * 16`` (K columns
``16 g .. 16 g + 15`` of the step) and the 16 bytes at ``1024 + l * 16`` (columns ``64 + 16 g ..``), i.e. exactly the MFMA
operand fragment of the lane in two 16-byte pieces.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Constexpr, Float32, Int8, Int32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import ceildiv

N_WAVES = 8
M_TILE = 64
N_TILE = 80
STEPS_PER_WAVE = 10
K_STEPS = N_WAVES * STEPS_PER_WAVE  # 80 steps of 128
K = K_STEPS * 128
N_REP = N_TILE // 16  # 5
M_REP = M_TILE // 16  # 4
HALF_TILES = (M_REP * N_REP) // 2  # accumulator tiles reduced per round
assert N_WAVES * STEPS_PER_WAVE * 128 == 10240
PARTIAL_BYTES = HALF_TILES * 64 * 16  # one wave's partial of one round


@flyc.jit
def launch_mx_down_ws(
    arg_c: fx.Pointer,
    arg_x: fx.Pointer,
    arg_w: fx.Pointer,
    arg_xs: fx.Pointer,
    arg_chan: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    prefetch_b: Constexpr[int] = 3,
):
    assert N % N_TILE == 0
    n_tiles = N // N_TILE
    kernel_name = f"mx_down_ws_{M_TILE}x{N_TILE}_k{K}_n{N}_pf{prefetch_b}"

    @fx.struct
    class Arena:
        data: fx.Array[Int8, N_WAVES * PARTIAL_BYTES, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[N_WAVES * 64, 1, 1])
    def kernel_down(
        arg_c: fx.Int64,
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_xs: fx.Int64,
        arg_chan: fx.Int64,
        i32_m: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane16 = lane % 16
        kgrp = lane // 16
        wgid = fx.Int32(fx.block_idx.x)
        m_tiles = ceildiv(i32_m, M_TILE)
        # Hardware dispatches workgroup i to XCD i % 8: a run of N tiles per XCD, the M tiles of one weight tile together.
        xcd = wgid % 8
        loc = wgid // 8
        per_xcd = n_tiles // 8
        mt = loc % m_tiles
        nt = xcd * per_xcd + loc // m_tiles

        def body():
            def _buffer(base, elem, align, nbytes):
                ptr_ty = fx.PointerType.get(elem, address_space=fx.AddressSpace.Global, alignment=align)
                view = fx.make_view(fx.inttoptr(ptr_ty, base), fx.make_layout(1 << 30, 1))
                buf = fx.rocdl.make_buffer_tensor(fx.Tensor(view), max_size=False, num_records_bytes=nbytes)
                return fx.logical_divide(buf, fx.make_layout(1, 1))

            row_tiles = (i32_m + 15) // 16
            x_words = _buffer(arg_x, T.i32, 16, fx.Int64(row_tiles) * (K_STEPS * 2048))
            w_words = _buffer(arg_w, T.i32, 16, fx.Int64(N) * K)
            xs_words = _buffer(arg_xs, T.i32, 4, fx.Int64(ceildiv(i32_m, M_TILE)) * M_TILE * (K // 32))
            load16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
            load32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
            atoms = [
                fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, opsel_a=osa, opsel_b=0)
                )
                for osa in range(4)
            ]
            unit = fx.Int32(0x7F7F7F7F)

            def _load16(src, off_i32):
                frag = fx.make_rmem_tensor(4, Int32)
                fx.copy(load16, fx.slice(src, (None, off_i32)), frag)
                return Vec(frag.load())

            def load_a(s):
                """The 4 A fragments (row tiles of this workgroup) and the scale dword (4 row tiles) of local step s."""
                ks = s * N_WAVES + wave
                frags = []
                for mi in range_constexpr(M_REP):
                    rt = mt * M_REP + mi
                    base = ((rt * K_STEPS + ks) * 2048 + lane * 16) // 4
                    lo = _load16(x_words, base)
                    hi = _load16(x_words, base + 256)
                    frag = fx.make_rmem_tensor(8, Int32)
                    frag.store(lo.shuffle(hi, list(range(8))))
                    frags.append(frag)
                sfrag = fx.make_rmem_tensor(1, Int32)
                fx.copy(load32, fx.slice(xs_words, (None, ((mt * K_STEPS + ks) * 4 + kgrp) * 16 + lane16)), sfrag)
                return frags, fx.Int32(Vec(sfrag.load())[0])

            def load_b(s, t):
                ks = s * N_WAVES + wave
                base = (((nt * N_REP + t) * K_STEPS + ks) * 2048 + lane * 16) // 4
                lo = _load16(w_words, base)
                hi = _load16(w_words, base + 256)
                frag = fx.make_rmem_tensor(8, Int32)
                frag.store(lo.shuffle(hi, list(range(8))))
                return frag

            accs = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(M_REP * N_REP)]
            for acc in accs:
                acc.store(Vec.filled(4, 0.0, Float32))

            units = STEPS_PER_WAVE * N_REP
            a_next = load_a(0)
            b_q = [load_b(u // N_REP, u % N_REP) for u in range_constexpr(min(prefetch_b, units))]
            a_cur = a_next
            for u in range_constexpr(units):
                s = u // N_REP
                t = u % N_REP
                if const_expr(t == 0):
                    a_cur = a_next
                nxt = u + prefetch_b
                if const_expr(nxt < units):
                    b_q.append(load_b(nxt // N_REP, nxt % N_REP))
                if const_expr(t == N_REP - 2 and s + 1 < STEPS_PER_WAVE):
                    a_next = load_a(s + 1)
                b = b_q.pop(0)
                rocdl.sched_barrier(0)  # keep this unit's prefetch loads ahead of its MFMAs
                for mi in range_constexpr(M_REP):
                    acc = accs[mi * N_REP + t]
                    fx.gemm(atoms[mi], acc, a_cur[0][mi], b, acc, scale_a=a_cur[1], scale_b=unit)

            # ---- reduction of the 8 K-split partials through LDS, two rounds of HALF_TILES tiles, then bf16 stores ----
            lds = fx.SharedAllocator().allocate(Arena).peek()
            lds_i32 = fx.recast_iter(Int32, lds.data.ptr)
            rw16 = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
            chan_ptr = fx.inttoptr(fx.PointerType.get(T.f32, fx.AddressSpace.Global, 4), arg_chan)
            out_ptr = fx.inttoptr(fx.PointerType.get(BFloat16.ir_type, fx.AddressSpace.Global, 2), arg_c)
            rows_left = i32_m - mt * M_TILE
            for rnd in range_constexpr(2):
                gpu.barrier()
                for f in range_constexpr(HALF_TILES):
                    frag = fx.make_rmem_tensor(4, Int32)
                    frag.store(Vec(accs[rnd * HALF_TILES + f].load()).bitcast(Int32))
                    dst = fx.make_view(fx.add_offset(lds_i32, (wave * PARTIAL_BYTES + (f * 64 + lane) * 16) // 4), fx.make_layout(4, 1))
                    fx.copy(rw16, frag, dst)
                gpu.barrier()
                # wave w sums tiles w and w + 8 (if < 10) of this round over the 8 partials
                for j in range_constexpr(2):
                    f_idx = wave + j * N_WAVES
                    if f_idx < HALF_TILES:
                        tot = Vec.filled(4, 0.0, Float32)
                        for sw in range_constexpr(N_WAVES):
                            frag = fx.make_rmem_tensor(4, Int32)
                            src = fx.make_view(fx.add_offset(lds_i32, (sw * PARTIAL_BYTES + (f_idx * 64 + lane) * 16) // 4), fx.make_layout(4, 1))
                            fx.copy(rw16, src, frag)
                            tot = tot + Vec(frag.load()).bitcast(Float32)
                        tile = rnd * HALF_TILES + f_idx
                        mi = tile // N_REP
                        ni = tile % N_REP
                        col = nt * N_TILE + ni * 16 + lane % 16
                        sw_v = fx.Float32(fx.ptr_load(fx.add_offset(chan_ptr, fx.Int64(col))))
                        vals = (tot * Vec.filled(4, sw_v, Float32)).to(BFloat16)
                        for ii in range_constexpr(4):
                            row_l = mi * 16 + (lane // 16) * 4 + ii
                            if row_l < rows_left:
                                fx.ptr_store(vals[ii], fx.add_offset(out_ptr, fx.Int64(mt * M_TILE + row_l) * N + col))

        if mt * M_TILE < i32_m:
            body()

    grid = (ceildiv(i32_m, M_TILE) * n_tiles, 1, 1)
    kernel_down(
        fx.Int64(fx.ptrtoint(arg_c)),
        fx.Int64(fx.ptrtoint(arg_x)),
        fx.Int64(fx.ptrtoint(arg_w)),
        fx.Int64(fx.ptrtoint(arg_xs)),
        fx.Int64(fx.ptrtoint(arg_chan)),
        i32_m,
    ).launch(grid=grid, block=(N_WAVES * 64, 1, 1), stream=stream)
