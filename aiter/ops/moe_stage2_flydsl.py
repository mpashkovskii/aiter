"""FlyDSL prefill gemm2 + pure-sum reduce as a drop-in CK 2-stage stage2
(gemm2) for per-token fp8 a8w8 on gfx942 (iter-14, MiniMax-M3.1-Flash).

The production CK stage2 (``aiter.ck_moe_stage2_fwd``) accumulates the top-k
partials into the shared [token, model_dim] output with a bf16 atomic-add
epilogue that also applies a2_scale, w2_scale and the routing weights.  On
the short K=384 shape (E=128/rank, N=6144) that epilogue is the bottleneck:
1222.8 us at T=16384 in isolation (~19% of the FP8 peak), ~12% of the mixed
serving step.

This module replaces that call with two kernels from the gfx9_fp8 whole-graph
MoE family (``aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8``):

  gemm2   ``moe_2stage_down_prefill_1x4``: full scale work INSIDE the
          epilogue (a2_scale per-token x routing weight x w2_scale
          per-channel), plain Set store of each sorted block's partials to a
          [num_blocks * block_m, model_dim] bf16 workspace (zero atomic
          traffic), at 583.7 TF at T=16384 -- above the 557 TF dense CK
          ceiling measured in iter-11;
  reduce  ``invert_sorted_ids`` + ``sorted_sum``: pure per-token topk sum
          (no scales/weights: they are already applied), at ~4.1 TB/s.

Isolated at the in-server shapes (hybrid_bench.py, production CK stage1 kept):
T=2048: 488.7 vs 537.3 us full-fwd (0.91x); T=4096: 636.9 vs 715.7 (0.89x);
T=8192: 1200.9 vs 1255.6 (0.96x); T=16384: 1920.5 vs 2251.7 (0.85x) with
stage2-equivalent 794 us vs 1223 us.  Numerics vs the atomic reference:
max rel err 0.0075 (bf16 out rounding; partials sum in fp32, at least as
accurate as 4x bf16 atomic adds).  Below ~2k tokens the workspace+reduce
overhead loses (1.13-1.6x), so the caller gates on token count.

The stage2 input contract is identical to ``ck_moe_stage2_fwd``: token-major
[token, topk, inter_dim] fp8 activations with per-row fp32 scales, preshuffled
[E, model_dim, inter_dim] fp8 weights, per-channel fp32 weight scales, and the
moe_sorting buffers (block_m threaded through unchanged).  num_valid_ids is
read on device inside the kernels: no host sync, CUDA-graph safe.  Enabled by
default on gfx942; AITER_CK_STAGE2_FLYDSL=0 disables, and
AITER_CK_STAGE2_FLYDSL_CTRL (default /tmp/aiter_ck_stage2_flydsl.enabled)
toggles per call from a control file ("0"/"1") for same-start interleaved
A/B/A measurement.
"""
import functools
import os

import torch

import aiter

_CTRL_DEFAULT = "/tmp/aiter_ck_stage2_flydsl.enabled"
_check_src = None
_ctrl_cache = {"path": None, "mtime": None, "value": True}


def _enabled():
    """Master env AITER_CK_STAGE2_FLYDSL (default 1) with a per-call control
    file override for same-start interleaved A/B/A (iter-11 pattern)."""
    if os.environ.get("AITER_CK_STAGE2_FLYDSL", "1") not in ("1", "true", "True"):
        return False
    path = os.environ.get("AITER_CK_STAGE2_FLYDSL_CTRL", _CTRL_DEFAULT)
    try:
        st = os.stat(path)
    except OSError:
        _ctrl_cache.update(path=path, mtime=None, value=True)
        return True
    if _ctrl_cache["path"] == path and _ctrl_cache["mtime"] == st.st_mtime_ns:
        return _ctrl_cache["value"]
    try:
        with open(path) as f:
            value = f.read().strip() not in ("", "0", "false", "False")
    except OSError:
        value = True
    _ctrl_cache.update(path=path, mtime=st.st_mtime_ns, value=value)
    return value


@functools.lru_cache(maxsize=64)
def _gemm2_kernel(E, N, K, topk, block_m):
    from aiter.ops.flydsl.fused_moe_gfx942 import _get_compiled_kernel

    return _get_compiled_kernel(
        N=N,
        K=K,
        weight_dtype_str="fp8",
        quant_type_str="ptpc",
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        BLOCK_TILE_SIZE_N=128,
        BLOCK_TILE_SIZE_K=None,
        stage="down",
        alg="prefill_1x4",
        E=E,
        act_quant_type_str="ptpc",
    )


@functools.lru_cache(maxsize=8)
def _invert(topk):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import invert_sorted_ids

    return invert_sorted_ids(topk)


@functools.lru_cache(maxsize=8)
def _sorted_sum(topk, model_dim):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import sorted_sum

    return sorted_sum(topk, model_dim)


def _supported_operands(inter_states, w2, sorted_weights, w2_scale, a2_scale):
    return (
        inter_states.dtype == torch.float8_e4m3fnuz
        and w2.dtype == torch.float8_e4m3fnuz
        and getattr(w2, "is_shuffled", False)
        and sorted_weights is not None
        and w2_scale is not None
        and w2_scale.dtype == torch.float32
        and a2_scale is not None
    )


def dispatch_block_m(
    token, q_type, q_dtype_a, q_dtype_w, dtype, run_1stage, is_ep, block_m
):
    """iter-14: for the FlyDSL stage2 contract (a8w8 fp8 per_Token on gfx942,
    EP-free, 2-stage) the prefill_1x4 gemm2 is occupancy-limited at
    block_m=128 when token is in [2048, 4096) (64 expert blocks << 304 CUs);
    block_m=64 wins the whole pipeline (measured full-fwd at T=2048: ~440 vs
    ~520 us; CK stage1 is block-agnostic at that size: 212.4 vs 217.0 us).
    Only touches the sorting/stage1 block choice for the contract this module
    replaces; AITER_CK_STAGE2_FLYDSL=0 keeps the stock heuristic."""
    if (
        _enabled()
        and not run_1stage
        and not is_ep
        and 2048 <= token < 4096
        and block_m == 128
        and q_type == aiter.QuantType.per_Token
        and q_dtype_a == aiter.dtypes.fp8
        and q_dtype_w == aiter.dtypes.fp8
        and dtype == aiter.dtypes.bf16
    ):
        from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() == "gfx942":
            return 64
    return block_m


def available(stage2_partial, expert_mask, token_num, inter_states, w2, kwargs):
    """Dispatch-time gate. Only replaces the production ck_moe_stage2_fwd call
    (including its tuned-CSV instance replacements) for the contract we have
    measured; every other path is untouched."""
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx942":
        return False
    if expert_mask is not None or token_num < 2048:
        return False
    func = getattr(stage2_partial, "func", stage2_partial)
    if func is not aiter.ck_moe_stage2_fwd:
        return False
    return _supported_operands(
        inter_states, w2, kwargs.get("sorted_weights"), kwargs.get("w2_scale"),
        kwargs.get("a2_scale"),
    )


def stage2(metadata_stage2, stage2_args, stage2_kwargs):
    """Runtime control-file toggle: fall through to the production call when
    disabled (same-start interleaved A/B/A without a server restart)."""
    if not _enabled():
        return metadata_stage2(*stage2_args, **stage2_kwargs)

    (
        inter_states,
        _w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
    ) = stage2_args
    w2_scale = stage2_kwargs["w2_scale"]
    a2_scale = stage2_kwargs["a2_scale"]
    block_m = int(stage2_kwargs["block_m"])
    sorted_weights = stage2_kwargs["sorted_weights"]

    token_num, _topk, inter_dim = inter_states.shape
    E, model_dim, _K = w2.shape

    n_blocks = int(sorted_expert_ids.shape[0])
    gemm2_out = torch.empty(
        (n_blocks * block_m, model_dim), dtype=out.dtype, device=out.device
    )
    kernel = _gemm2_kernel(E, model_dim, inter_dim, topk, block_m)

    from aiter.ops.flydsl.fused_moe_gfx942 import _launch

    _launch(
        kernel,
        inter_states,
        w2,
        gemm2_out,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale if w2_scale is not None else torch.empty(0, device=out.device),
        a2_scale.view(token_num, topk).contiguous()
        if a2_scale is not None
        else torch.empty((token_num, topk), dtype=torch.float32, device=out.device),
        token_num,
        n_blocks,
    )

    loc_ids = torch.empty(
        (token_num, topk), dtype=torch.int32, device=out.device
    )
    _invert(topk)(
        sorted_token_ids, loc_ids, num_valid_ids, sorted_token_ids.shape[0], token_num
    )
    _sorted_sum(topk, model_dim)(loc_ids, gemm2_out, out, token_num)
    return out
