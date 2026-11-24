# unified_attention(prefix_groups=...) against the same call without groups and an
# fp32 reference, on batches with shared leading blocks (gfx950 gluon backend).
#
# Batches: num_seqs requests, each with qlen query tokens; groups of requests share their
# first `shared` physical blocks (a prefix-cache hit), the rest of each table is private.
# A "turn change" group has one member that shares a single block only (it must stay
# private, not shorten the group's prefix). Mixed batches append prompt rows (long
# queries) after the short rows, as vLLM orders them, and build the groups from the short
# rows only (num_seqs / num_tokens / max_query_len set as the vLLM builder does).
import random

import pytest
import torch

import aiter.ops.triton.attention.unified_attention as ua

FP8 = torch.float8_e4m3fn
HD = 256
NUM_BLOCKS = 4096  # raised per batch in build()


def build(num_seqs, qlen, block_size, groups, shared, ctx, seed, prompt_q=()):
    global NUM_BLOCKS
    rnd = random.Random(seed)
    need = (num_seqs + len(prompt_q)) * ((ctx + max(prompt_q, default=0)) // block_size + 8 + shared)
    NUM_BLOCKS = max(1024, need)
    free = list(range(NUM_BLOCKS))
    rnd.shuffle(free)
    rows, lens, qls = [], [], []
    members = []
    for g in range(groups):
        members += [g] * (num_seqs // groups)
    members += [-1] * (num_seqs - len(members))
    rnd.shuffle(members)
    prefix = {g: [free.pop() for _ in range(shared)] for g in range(groups)}
    outlier = {}
    for i, g in enumerate(members):
        L = ctx + rnd.randrange(0, 4 * block_size) + qlen
        nb = (L + block_size - 1) // block_size
        if g >= 0 and g == 0 and g not in outlier:
            outlier[g] = i  # shares one block only (a short new prompt joining)
            row = prefix[g][:1] + [free.pop() for _ in range(nb - 1)]
        elif g >= 0:
            row = prefix[g] + [free.pop() for _ in range(nb - shared)]
        else:
            row = [free.pop() for _ in range(nb)]
        rows.append(row)
        lens.append(L)
        qls.append(qlen)
    for pq in prompt_q:  # prompt rows last, sharing group 1's prefix
        L = shared * block_size + pq + rnd.randrange(0, block_size)
        nb = (L + block_size - 1) // block_size
        rows.append(prefix[min(1, groups - 1)] + [free.pop() for _ in range(nb - shared)])
        lens.append(L)
        qls.append(pq)
    width = max(len(r) for r in rows) + 2
    bt = torch.zeros(len(rows), width, dtype=torch.int32)
    for i, r in enumerate(rows):
        bt[i, : len(r)] = torch.tensor(r, dtype=torch.int32)
    cu = torch.tensor([0] + list(torch.tensor(qls).cumsum(0)), dtype=torch.int32)
    return bt.cuda(), torch.tensor(lens, dtype=torch.int32).cuda(), cu.cuda(), qls


def reference(q, k, v, bt, lens, cu, qls, block_size, scale):
    nq, nkv = q.shape[1], k.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for i, ql in enumerate(qls):
        L = int(lens[i])
        nb = (L + block_size - 1) // block_size
        blocks = bt[i, :nb].long()
        kk = k[blocks].reshape(-1, nkv, HD)[:L].float()
        vv = v[blocks].reshape(-1, nkv, HD)[:L].float()
        q0 = int(cu[i])
        qq = q[q0 : q0 + ql].float()
        pos = torch.arange(L, device=q.device)[None, :]
        qpos = (L - ql + torch.arange(ql, device=q.device))[:, None]
        for h in range(nq):
            kh = h // (nq // nkv)
            s = (qq[:, h] @ kk[:, kh].T) * scale
            s = s.masked_fill(pos > qpos, float("-inf"))
            out[q0 : q0 + ql, h] = torch.softmax(s, -1) @ vv[:, kh]
    return out


@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("num_heads", [(24, 4), (8, 1)])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize(
    "num_seqs,qlen,groups,shared,ctx,prompt_q",
    [
        (64, 2, 4, 40, 3000, ()),  # verify, 4 groups
        (64, 3, 2, 60, 4000, ()),  # 3-token verify
        (48, 2, 0, 0, 2000, ()),  # no sharing: groups stay empty
        (17, 2, 1, 20, 1500, ()),  # one group just above the minimum batch
        (8, 2, 1, 20, 1500, ()),  # below the minimum batch: plain path
        (48, 2, 3, 30, 2500, (200, 37)),  # mixed: short rows lead, prompt rows last
        # iter-29: rows of 5 tokens after the lead are mixed for vLLM (groups over the lead
        # only) but still fit the short launch (30 rows at 24/4): the groups must not be
        # applied to the whole call
        (48, 2, 3, 30, 2500, (5,)),
        (48, 2, 3, 30, 2500, (5, 5, 5)),
        (17, 2, 1, 20, 1500, (5,)),
        (16, 2, 1, 20, 1500, (5,)),
    ],
)
def test_prefix_groups(dtype, num_heads, block_size, num_seqs, qlen, groups, shared, ctx, prompt_q):
    torch.manual_seed(0)
    nq, nkv = num_heads
    bt, lens, cu, qls = build(num_seqs, qlen, block_size, groups, shared, ctx, 1, prompt_q)
    T = int(cu[-1])
    raw = torch.randn(NUM_BLOCKS, block_size, nkv, 2 * HD, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(T, nq, HD, device="cuda", dtype=torch.bfloat16) * 0.5
    one = None
    if dtype == "fp8":
        raw, q = raw.to(FP8), q.to(FP8)
        one = torch.ones((), dtype=torch.float32, device="cuda")
    k, v = raw[..., :HD], raw[..., HD:]
    scale = HD ** -0.5
    pg = ua.alloc_prefix_groups(512, 8192, bt.shape[1] + 4, 32, "cuda")
    ns = num_seqs
    # stale buffers, as in the server (the previous step's groups): a build over the whole
    # batch first, so rows past the lead hold another batch's tails
    ua.build_prefix_groups(pg, bt, lens, cu, block_size)
    ua.build_prefix_groups(pg, bt[:ns], lens[:ns], cu[: ns + 1], block_size)
    if prompt_q:
        pg = pg._replace(num_seqs=ns, num_tokens=int(cu[ns]), max_query_len=qlen)
    outs = []
    for groups_arg in (None, pg):
        out = torch.zeros(T, nq, HD, device="cuda", dtype=torch.bfloat16)
        ua.unified_attention(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=max(qls), seqused_k=lens,
            max_seqlen_k=int(lens.max()), softmax_scale=scale, causal=True, window_size=(-1, -1),
            block_table=bt, softcap=0, q_descale=one, k_descale=one, v_descale=one,
            backend="gluon", prefix_groups=groups_arg,
        )
        outs.append(out.float())
    ref = reference(q, k, v, bt, lens, cu, qls, block_size, scale)
    err_plain = (outs[0] - ref).abs().max().item()
    err_prefix = (outs[1] - ref).abs().max().item()
    diff = (outs[1] - outs[0]).abs().max().item()
    covered = int(pg.seq_shift[:ns].long().sum())
    print(f"groups found {int((pg.grp_len > 0).sum())} shared blocks {covered}  "
          f"err plain {err_plain:.2e} prefix {err_prefix:.2e} diff {diff:.2e}")
    if groups and num_seqs >= ua.PREFIX_GROUPS_MIN_SEQS:
        assert covered > 0, "the shared prefixes were not found"
        # the outlier of group 0 keeps its one shared block private
    assert err_prefix <= max(2 * err_plain, 2e-2), (err_plain, err_prefix)
    assert diff <= max(2 * err_plain, 2e-2)
