"""Two-pass CK 2-stage MoE stage2 (gemm2) for per-token fp8 on gfx942.

The production CK stage2 (``ck_moe_stage2_fwd``) uses a bf16 atomic-add
epilogue that accumulates the top-k partials directly into the shared
[token, model_dim] output, with the dequant scales (a2_scale, w2_scale) and
routing weights applied in the same epilogue.  On short-K shapes
(K = inter_dim = 384 here) that epilogue is the bottleneck: measured
1222.8 us at T=16384, E=128, N=6144, K=384, topk=4 in isolation (20.4% of
the FP8 peak; the dense CK GEMM at the same tile geometry reaches 44.9%),
i.e. ~11.5% of the mixed decode/prefill step in vLLM serving.

This module replaces that call with:

  pass0  gather stage1's [token*topk, K] fp8 activations (+ row scales and
         routing weights folded into one per-row scalar) into sorted order,
         and the token->(expert, sorted-row) maps (2 tiny Triton kernels);
  pass1  the very same ``ck_moe_stage2_fwd`` GEMM, but with IDENTITY sorted
         ids and NO scales in the epilogue, storing every partial to a
         unique row of a [rows_pad, N] bf16 workspace (Set-store equivalent:
         no atomic traffic, no epilogue scale work);
  pass2  one Triton reduction: out[t, :] = sum_k c[row(t,k)] *
         w2_scale[e(row(t,k)), :] * ws[row(t,k), :], reading ws once at
         ~4 TB/s.

Measured in isolation at the in-server shapes (MiniMax-M3, gfx942):
T=2048: 217.5 vs 246.4 us (1.13x); T=8192: 519.0 vs 651.9 (1.26x);
T=16384: 947.9 vs 1222.8 us (1.29x), with rel. error vs the atomic
reference <= 1e-2 (bf16 output rounding dominates; partials sum in fp32).
Below ~2k tokens the overhead loses, so the caller gates on token count.

No host syncs: num_valid_ids is read on-device inside the kernels, so this
is CUDA-graph safe.  All buffers come from the torch caching allocator.
"""
import os

import torch
import triton
import triton.language as tl

import aiter
from aiter import QuantType

_ENABLED = os.environ.get("AITER_CK_STAGE2_TWOPASS", "1") == "1"
_MIN_TOKENS = int(os.environ.get("AITER_CK_STAGE2_TWOPASS_MIN_TOKENS", "2048"))

# Runtime control file for same-start A/B/A experiments: if the file named by
# AITER_CK_STAGE2_TWOPASS_CTRL (default /tmp/aiter_stage2_twopass.enabled)
# exists, its content ("1"/"0") wins over the env default.  Reloaded on mtime
# change only, so steady-state cost is one stat() per MoE call.
_CTRL = os.environ.get(
    "AITER_CK_STAGE2_TWOPASS_CTRL", "/tmp/aiter_stage2_twopass.enabled"
)
_ctrl_state = {"mtime": None, "on": True}


def _enabled() -> bool:
    try:
        m = os.stat(_CTRL).st_mtime_ns
    except OSError:
        return _ENABLED
    if m != _ctrl_state["mtime"]:
        try:
            with open(_CTRL) as f:
                _ctrl_state["on"] = f.read().strip().lower() not in (
                    "0", "off", "false", "")
        except OSError:
            _ctrl_state["on"] = _ENABLED
        _ctrl_state["mtime"] = m
    return _ctrl_state["on"]

_IDENTITY_CACHE: dict = {}


def _identity(n: int, device) -> torch.Tensor:
    key = (n, device.index)
    t = _IDENTITY_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device, dtype=torch.int32)
        _IDENTITY_CACHE[key] = t
    return t


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_bn(n: int) -> int:
    for bn in (2048, 1536, 1024, 768, 512, 256, 128, 64):
        if n % bn == 0:
            return bn
    return 0


@triton.jit
def _twopass_gather(A2, SCAL, SW, FUSED, NVP, DST, DSTS, TTOK,
                    K32: tl.constexpr, KPOW: tl.constexpr, TOPK: tl.constexpr):
    """Reorder stage1 fp8 rows into sorted order + fold scales into one scalar.

    A2:   [T*TOPK, K] fp8 viewed as int32 rows of K32 = K/4 elements.
    SCAL: [T*TOPK] fp32 per-row activation scale (unsorted order).
    SW:   [rows_pad] fp32 routing weight in sorted order (sorted_weights).
    FUSED:[rows_pad] int32 fused sorted ids (token | (slot << 24)).
    DST:  [rows_pad, K] fp8 as int32 rows, sorted order (zeros beyond valid).
    DSTS: [rows_pad] fp32 combined scalar: act_scale(src(r)) * SW[r].
    Padding/invalid rows write zero rows and scale 0.
    """
    nv = tl.load(NVP)
    pid = tl.program_id(0)
    rows = pid * 8 + tl.arange(0, 8)
    offs = tl.arange(0, KPOW)
    km = offs < K32
    ok = rows < nv
    f = tl.load(FUSED + rows, mask=ok, other=-1)
    tok = f & 0xFFFFFF
    slot = f >> 24
    ok = ok & (tok < TTOK) & (slot < TOPK)
    src = (tok * TOPK + slot) * ok
    m = ok[:, None] & km[None, :]
    v = tl.load(A2 + src[:, None] * K32 + offs[None, :], mask=m, other=0)
    # write every destination row: gathered content, or zeros for padding
    tl.store(DST + rows[:, None] * K32 + offs[None, :], v, mask=km[None, :])
    s = tl.load(SCAL + src, mask=ok, other=0.0)
    w = tl.load(SW + rows, mask=rows < nv, other=0.0)
    tl.store(DSTS + rows, s * w)


@triton.jit
def _twopass_inv_epr(FUSED, NVP, EXP_IDS, INV, EPR, TTOK,
                     BM: tl.constexpr, TOPK: tl.constexpr):
    """inv[token*TOPK + slot] = sorted-row; epr[r] = expert of sorted-row r."""
    nv = tl.load(NVP)
    pid = tl.program_id(0)
    ok = pid < nv
    f = tl.load(FUSED + pid, mask=ok, other=-1)
    tok = f & 0xFFFFFF
    slot = f >> 24
    real = ok & (tok < TTOK) & (slot < TOPK)
    tl.store(INV + tok * TOPK + slot, pid, mask=real)
    e = tl.load(EXP_IDS + pid // BM, mask=ok, other=0)
    tl.store(EPR + pid, e, mask=ok)


@triton.jit
def _twopass_fold(WS, INV, C, EPR, W2S, OUT,
                  N: tl.constexpr, BN: tl.constexpr, TOPK: tl.constexpr):
    """out[t] = sum_k C[r] * W2S[EPR[r], :] * WS[r, :] where r = INV[t, k]."""
    t = tl.program_id(0)
    nb = tl.program_id(1)
    offs = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        r = tl.load(INV + t * TOPK + k).to(tl.int64)
        e = tl.load(EPR + r).to(tl.int64)
        c = tl.load(C + r).to(tl.float32)
        w = tl.load(W2S + e * N + offs)
        acc += c * w * tl.load(WS + r * N + offs).to(tl.float32)
    tl.store(OUT + t * N + offs, acc.to(tl.bfloat16))


def available(stage2_callable, expert_mask, token_num, model_dim,
              sorted_weights, a2, w2) -> bool:
    """Gate the two-pass path: CK stage2, per-token fp8, no EP, large enough."""
    if not _enabled():
        return False
    func = getattr(stage2_callable, "func", stage2_callable)
    if func is not aiter.ck_moe_stage2_fwd:
        return False
    if getattr(stage2_callable, "keywords", {}).get("quant_type",
                                                    QuantType.per_Token) != QuantType.per_Token:
        return False
    if expert_mask is not None or sorted_weights is None:
        return False
    # stage2 operands must be 8-bit (fp8): the gather moves raw int32 lanes
    if a2 is None or a2.element_size() != 1 or a2.shape[-1] % 4 != 0:
        return False
    if w2.element_size() != 1:
        return False
    if token_num < _MIN_TOKENS:
        return False
    if _pick_bn(model_dim) == 0:
        return False
    return True


def ck_stage2_twopass(
    a2,  # [token_num, topk, inter_dim] fp8
    w1,
    w2,  # [expert, model_dim, inter_dim] fp8
    sorted_ids,  # fused ids [rows_pad]
    sorted_expert_ids,  # [rows_pad // block_m]
    num_valid_ids,
    moe_out,  # [token_num, model_dim] bf16, fully overwritten
    topk,
    kernelName=None,
    w2_scale=None,  # [expert, model_dim, 1] fp32
    a2_scale=None,  # [token_num * topk] fp32 flattened
    block_m=32,
    sorted_weights=None,  # [rows_pad] fp32
    quant_type=QuantType.No,
    activation=0,
    **_unused,
):
    device = a2.device
    token_num, topk, inter_dim = a2.shape
    model_dim = moe_out.shape[1]
    rows_pad = sorted_ids.numel()
    E = w2.shape[0]

    k32 = inter_dim // 4
    ws = torch.empty(rows_pad, model_dim, dtype=moe_out.dtype, device=device)
    a2s = torch.empty(rows_pad, inter_dim, dtype=a2.dtype, device=device)
    cbuf = torch.empty(rows_pad, dtype=torch.float32, device=device)
    inv = torch.empty(token_num * topk, dtype=torch.int32, device=device)
    epr = torch.empty(rows_pad, dtype=torch.int32, device=device)

    a2v = a2.view(-1, inter_dim)
    a2scl = a2_scale.contiguous().view(-1)
    _twopass_gather[((rows_pad + 7) // 8,)](
        a2v.view(torch.int32), a2scl, sorted_weights, sorted_ids,
        num_valid_ids, a2s.view(torch.int32), cbuf, token_num,
        K32=k32, KPOW=_next_pow2(k32), TOPK=topk, num_warps=2,
    )
    _twopass_inv_epr[(rows_pad,)](
        sorted_ids, num_valid_ids, sorted_expert_ids, inv, epr, token_num,
        BM=block_m, TOPK=topk, num_warps=1,
    )
    aiter.ck_moe_stage2_fwd(
        a2s, w1, w2,
        _identity(rows_pad, device), sorted_expert_ids, num_valid_ids,
        ws, 1,
        kernelName=kernelName,
        w2_scale=None,
        a2_scale=None,
        block_m=block_m,
        sorted_weights=None,
        quant_type=quant_type,
        activation=activation,
    )
    bn = _pick_bn(model_dim)
    _twopass_fold[(token_num, model_dim // bn)](
        ws, inv, cbuf, epr, w2_scale.view(E, model_dim), moe_out,
        N=model_dim, BN=bn, TOPK=topk, num_warps=8,
    )
    return moe_out
