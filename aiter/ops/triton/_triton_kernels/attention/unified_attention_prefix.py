# Shared-prefix (cascade) attention for short queries (speculative-decode verify): the
# group detection on the block table and the merge of the two sets of split-KV partials.
# The host side is in aiter/ops/triton/attention/unified_attention.py (PrefixGroups,
# build_prefix_groups, the prefix path of unified_attention).
#
# Groups are found on the GPU, into fixed buffers, with fixed grids and no host sync, so
# the metadata can be built for a CUDA-graph replay:
#   leader  each sequence's leader is the lowest-indexed eligible sequence with the same
#           first block; run = the leading blocks it shares with the leader, capped so the
#           sequence keeps its query tokens (and their keys) in its private tail
#   score   a group takes the shared length S that maximizes members(run >= S) x S, so a
#           short new prompt joining a group stays private instead of collapsing the prefix
#   member  members (run >= S) add their query tokens to the group's count
#   group   groups of at least MIN_GROUP members get a slot (at most MAX_GROUPS) and a
#           range of virtual tokens
#   fill    per sequence: its virtual tokens, the block table of its private tail, and the
#           leader's table for the group
import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)

@triton.jit
def _pg_leader_kernel(
    bt_ptr, bt_stride, seq_lens_ptr, cu_ptr, num_seqs, num_tokens, max_tokens,
    leader_ptr, run_ptr, best_ptr, gtok_ptr, fill_ptr, vtok_src_ptr, tok_vtok_ptr,
    BLOCK_SIZE: tl.constexpr, MIN_BLOCKS: tl.constexpr, N_PAD: tl.constexpr,
    W_PAD: tl.constexpr, TOK_CHUNK: tl.constexpr,
):
    i = tl.program_id(0)
    # reset the per-step scratch and token maps (this program's share of them)
    tl.store(best_ptr + i, 0)
    tl.store(gtok_ptr + i, 0)
    tl.store(fill_ptr + i, 0)
    offs_t = i * TOK_CHUNK + tl.arange(0, TOK_CHUNK)
    tl.store(vtok_src_ptr + offs_t, 0, mask=offs_t < max_tokens)
    tl.store(tok_vtok_ptr + offs_t, -1, mask=offs_t < max_tokens)
    if i >= num_seqs:
        tl.store(leader_ptr + i, -1)
        tl.store(run_ptr + i, 0)
        return
    offs = tl.arange(0, N_PAD)
    valid = offs < num_seqs
    L = tl.load(seq_lens_ptr + offs, mask=valid, other=0)
    q0 = tl.load(cu_ptr + offs, mask=valid, other=0)
    q1 = tl.load(cu_ptr + offs + 1, mask=valid, other=0)
    cap = (L - (q1 - q0)) // BLOCK_SIZE
    elig = valid & (cap >= MIN_BLOCKS) & (q1 > q0)
    first = tl.load(bt_ptr + offs.to(tl.int64) * bt_stride, mask=elig, other=-1)
    cap_i = tl.sum(tl.where(offs == i, cap, 0))
    elig_i = tl.sum(tl.where((offs == i) & elig, 1, 0)) > 0
    f_i = tl.sum(tl.where(offs == i, first, 0))
    if not elig_i:
        tl.store(leader_ptr + i, -1)
        tl.store(run_ptr + i, 0)
        return
    ld = tl.min(tl.where(elig & (first == f_i), offs, N_PAD))
    cap_l = tl.sum(tl.where(offs == ld, cap, 0))
    lim = tl.minimum(cap_i, cap_l)
    k = tl.arange(0, W_PAD)
    m = k < lim
    ri = tl.load(bt_ptr + i.to(tl.int64) * bt_stride + k, mask=m, other=0)
    rl = tl.load(bt_ptr + ld.to(tl.int64) * bt_stride + k, mask=m, other=0)
    run = tl.min(tl.where(m & (ri != rl), k, lim))
    tl.store(leader_ptr + i, ld)
    tl.store(run_ptr + i, run)


@triton.jit
def _pg_score_kernel(
    leader_ptr, run_ptr, best_ptr, num_seqs,
    MIN_GROUP: tl.constexpr, MIN_BLOCKS: tl.constexpr, N_PAD: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= num_seqs:
        return
    li = tl.load(leader_ptr + i)
    ri = tl.load(run_ptr + i)
    if (li < 0) | (ri < MIN_BLOCKS):
        return
    offs = tl.arange(0, N_PAD)
    ld = tl.load(leader_ptr + offs, mask=offs < num_seqs, other=-1)
    rn = tl.load(run_ptr + offs, mask=offs < num_seqs, other=0)
    cnt = tl.sum(tl.where((ld == li) & (rn >= ri), 1, 0))
    if cnt >= MIN_GROUP:
        key = (cnt.to(tl.int64) * ri.to(tl.int64)) * 65536 + ri.to(tl.int64)
        tl.atomic_max(best_ptr + li, key)


@triton.jit
def _pg_member_kernel(leader_ptr, run_ptr, best_ptr, gtok_ptr, cu_ptr, num_seqs):
    i = tl.program_id(0)
    if i >= num_seqs:
        return
    li = tl.load(leader_ptr + i)
    if li < 0:
        return
    b = tl.load(best_ptr + li)
    S = (b % 65536).to(tl.int32)
    if (b > 0) & (tl.load(run_ptr + i) >= S):
        ql = tl.load(cu_ptr + i + 1) - tl.load(cu_ptr + i)
        tl.atomic_add(gtok_ptr + li, ql)


@triton.jit
def _pg_group_kernel(
    best_ptr, gtok_ptr, gid_ptr, gbase_ptr, grp_cu_ptr, grp_len_ptr, num_seqs,
    BLOCK_SIZE: tl.constexpr, MAX_GROUPS: tl.constexpr, N_PAD: tl.constexpr,
    G_PAD: tl.constexpr,
):
    offs = tl.arange(0, N_PAD)
    valid = offs < num_seqs
    b = tl.load(best_ptr + offs, mask=valid, other=0)
    is_g = b > 0
    gid = tl.cumsum(is_g.to(tl.int32), 0) - 1
    is_g = is_g & (gid < MAX_GROUPS)
    tok = tl.where(is_g, tl.load(gtok_ptr + offs, mask=valid, other=0), 0)
    base = tl.cumsum(tok, 0) - tok
    tl.store(gid_ptr + offs, tl.where(is_g, gid, -1), mask=valid)
    tl.store(gbase_ptr + offs, base, mask=valid)
    g = tl.arange(0, G_PAD)
    # grp_cu[g] = tokens of the groups before g; grp_len[g] = shared tokens of group g
    before = is_g[None, :] & (gid[None, :] < g[:, None])
    cu = tl.sum(tl.where(before, tok[None, :], 0), 1)
    tl.store(grp_cu_ptr + g, cu, mask=g <= MAX_GROUPS)
    S = (b % 65536).to(tl.int32) * BLOCK_SIZE
    mine = is_g[None, :] & (gid[None, :] == g[:, None])
    tl.store(grp_len_ptr + g, tl.sum(tl.where(mine, S[None, :], 0), 1), mask=g < MAX_GROUPS)


@triton.jit
def _pg_fill_kernel(
    bt_ptr, bt_stride, seq_lens_ptr, cu_ptr, num_seqs, W,
    leader_ptr, run_ptr, best_ptr, fill_ptr, gid_ptr, gbase_ptr,
    seq_shift_ptr, seq_priv_ptr, bt_priv_ptr, bt_priv_stride, bt_grp_ptr, bt_grp_stride,
    vtok_src_ptr, tok_vtok_ptr,
    BLOCK_SIZE: tl.constexpr, W_PAD: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= num_seqs:
        return
    L = tl.load(seq_lens_ptr + i)
    q0 = tl.load(cu_ptr + i)
    ql = tl.load(cu_ptr + i + 1) - q0
    li = tl.load(leader_ptr + i)
    S = ql * 0
    g = ql * 0 - 1
    if li >= 0:
        b = tl.load(best_ptr + li)
        g = tl.load(gid_ptr + li)
        Sb = (b % 65536).to(tl.int32)
        if (b > 0) & (g >= 0) & (tl.load(run_ptr + i) >= Sb):
            S = Sb
    k = tl.arange(0, W_PAD)
    row = tl.load(bt_ptr + i.to(tl.int64) * bt_stride + S + k, mask=k + S < W, other=0)
    tl.store(bt_priv_ptr + i.to(tl.int64) * bt_priv_stride + k, row, mask=k < W)
    tl.store(seq_shift_ptr + i, S)
    tl.store(seq_priv_ptr + i, L - S * BLOCK_SIZE)
    if S > 0:
        slot = tl.atomic_add(fill_ptr + li, ql)
        vt = tl.load(gbase_ptr + li) + slot
        for j in range(0, ql):
            tl.store(tok_vtok_ptr + q0 + j, vt + j)
            tl.store(vtok_src_ptr + vt + j, q0 + j)
        if li == i:
            full = tl.load(bt_ptr + i.to(tl.int64) * bt_stride + k, mask=k < S, other=0)
            tl.store(bt_grp_ptr + g.to(tl.int64) * bt_grp_stride + k, full, mask=k < W)


@triton.jit
def _find_seq(cu_ptr, t, num_seqs):
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(cu_ptr + mid) <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@triton.jit
def _pg_merge_kernel(
    out_ptr, out_s0, out_s1, out_scale_ptr,
    p_acc, p_m, p_l, s_acc, s_m, s_l,
    cu_ptr, num_seqs, seq_priv_ptr, seq_shift_ptr, tok_vtok_ptr,
    NH: tl.constexpr, D: tl.constexpr, P: tl.constexpr, SH: tl.constexpr,
    TILE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min, FP8_MAX: tl.constexpr = float8_info.max,
):
    """out[t, h] from the private (P) and shared (SH) split-KV partials of token t.

    Each part's live segments follow the producer's carve (as reduce_segments): a split
    of TILE-token tiles over its own KV length, the private tail or the group's prefix.
    """
    t = tl.program_id(0)
    h = tl.program_id(1)
    seq = _find_seq(cu_ptr, t, num_seqs)
    offs_d = tl.arange(0, D)
    L = tl.load(seq_priv_ptr + seq)
    tps = tl.maximum(tl.cdiv(L, P * TILE), 1)
    act_p = tl.cdiv(L, tps * TILE)
    offs_p = tl.arange(0, P)
    base_p = (t.to(tl.int64) * NH + h) * P
    mp = tl.load(p_m + base_p + offs_p, mask=offs_p < act_p, other=float("-inf"))
    lp = tl.load(p_l + base_p + offs_p, mask=offs_p < act_p, other=0.0)
    ap = tl.load(p_acc + (base_p + offs_p)[:, None] * D + offs_d[None, :],
                 mask=(offs_p < act_p)[:, None], other=0.0)
    vt = tl.load(tok_vtok_ptr + t)
    S = tl.load(seq_shift_ptr + seq) * BLOCK_SIZE
    offs_s = tl.arange(0, SH)
    tss = tl.maximum(tl.cdiv(S, SH * TILE), 1)
    act_s = tl.where(vt >= 0, tl.cdiv(S, tss * TILE), 0)
    base_s = (tl.maximum(vt, 0).to(tl.int64) * NH + h) * SH
    ms = tl.load(s_m + base_s + offs_s, mask=offs_s < act_s, other=float("-inf"))
    ls = tl.load(s_l + base_s + offs_s, mask=offs_s < act_s, other=0.0)
    as_ = tl.load(s_acc + (base_s + offs_s)[:, None] * D + offs_d[None, :],
                  mask=(offs_s < act_s)[:, None], other=0.0)
    gmax = tl.maximum(tl.max(mp), tl.max(ms))
    gmax = tl.where(gmax > float("-inf"), gmax, 0.0)
    wp = tl.where(mp > float("-inf"), tl.math.exp2(mp - gmax), 0.0)
    ws = tl.where(ms > float("-inf"), tl.math.exp2(ms - gmax), 0.0)
    den = tl.sum(lp * wp) + tl.sum(ls * ws)
    num = tl.sum(ap * wp[:, None], 0) + tl.sum(as_ * ws[:, None], 0)
    o = num / tl.where(den > 0.0, den, 1.0)
    if out_scale_ptr is not None:
        o = o / tl.load(out_scale_ptr)
        o = tl.minimum(tl.maximum(o, FP8_MIN), FP8_MAX)
    tl.store(out_ptr + t.to(tl.int64) * out_s0 + h * out_s1 + offs_d,
             o.to(out_ptr.dtype.element_ty))
