"""HIP fp8 prefill kernel for gfx950 at head size 512 (no sliding window, GQA 4, 64- or 128-token
pages, no softcap, sinks or alibi).

The source (ua_hd512_prefill.cu) is compiled with hipcc on first use into a per-user cache and
launched through ctypes on the current torch stream. Switch: AITER_UA_HD512_HIP (default 1);
AITER_UA_HIP_DEBUG=1 prints why a call falls back to the gluon kernel.
"""
import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess

import torch

_SRC = os.environ.get("AITER_UA512_SRC_EXPERIMENT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "ua_hd512_prefill.cu")  # EXPERIMENT-ONLY override
_FLAGS = ["-O3", "-w", "--offload-arch=gfx950", "-shared", "-fPIC"]
_BT_MAX_TILES = 3424
_lib = None
_logged = set()


class _Args(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p), ("kc", ctypes.c_void_p), ("vc", ctypes.c_void_p), ("out", ctypes.c_void_p),
        ("block_table", ctypes.c_void_p), ("seq_lens", ctypes.c_void_p), ("cu_q", ctypes.c_void_p),
        ("q_descale", ctypes.c_void_p), ("k_descale", ctypes.c_void_p), ("v_descale", ctypes.c_void_p),
        ("q_s0", ctypes.c_longlong), ("q_s1", ctypes.c_longlong), ("o_s0", ctypes.c_longlong), ("o_s1", ctypes.c_longlong),
        ("kv_s_blk", ctypes.c_longlong), ("kv_s_tok", ctypes.c_longlong), ("kv_s_head", ctypes.c_longlong),
        ("bt_stride", ctypes.c_int), ("num_seqs", ctypes.c_int), ("scale", ctypes.c_float), ("blk_shift", ctypes.c_int),
    ]


def _build():
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src + " ".join(_FLAGS).encode()).hexdigest()[:16]
    cache = os.environ.get("AITER_UA512_CACHE", os.path.expanduser("~/.cache/aiter_ua512"))
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f"ua_hd512_{tag}.so")
    if os.path.exists(so):
        return so
    with open(os.path.join(cache, f"{tag}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(so):
            hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.run([hipcc, *_FLAGS, "-o", tmp, _SRC], check=True, capture_output=True)
            os.replace(tmp, so)
    return so


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_build())
        lib.ua512_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.ua512_launch.restype = ctypes.c_int
        _lib = lib
    return _lib


def supported(q, k, v, out, num_kv_heads, block_size, head_size, softcap, sinks, alibi_slopes, qq_bias,
              output_scale, sliding_window, max_seqlen_k) -> bool:
    if head_size != 512 or os.environ.get("AITER_UA_HD512_HIP", "1") == "0":
        return False
    checks = {
        "window": sliding_window == 0,
        "gqa4": q.shape[1] == 4 * num_kv_heads,
        "page": block_size in (64, 128),
        "plain": not softcap and sinks is None and alibi_slopes is None and qq_bias is None and output_scale is None,
        "dtypes": q.dtype == k.dtype == v.dtype and q.dtype.itemsize == 1 and out.dtype == torch.bfloat16,
        "strides": q.stride(2) == 1 and out.stride(2) == 1 and k.stride() == v.stride() and k.stride(3) == 1,
        "align": (k.data_ptr() & 15) == 0 and (v.data_ptr() & 15) == 0,
        "seqlen": int(max_seqlen_k) <= _BT_MAX_TILES * 64,
    }
    ok = all(checks.values())
    if not ok and os.environ.get("AITER_UA_HIP_DEBUG") and "miss" not in _logged:
        _logged.add("miss")
        print(f"[aiter] ua_hip_prefill: not used for head size 512: {[n for n, c in checks.items() if not c]} "
              f"window={sliding_window} kstride={k.stride()} k={tuple(k.shape)}", flush=True)
    return ok


def run(q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, q_descale, k_descale, v_descale):
    lib = _load()
    if "use" not in _logged:
        _logged.add("use")
        print("[aiter] ua_hip_prefill: HIP fp8 prefill kernel in use for head size 512", flush=True)
    ones = None
    keep = []
    ptrs = []
    for d in (q_descale, k_descale, v_descale):
        if d is None:
            if ones is None:
                ones = torch.ones(1, dtype=torch.float32, device=q.device)
            d = ones
        d = d.reshape(-1)
        if d.dtype != torch.float32:
            d = d.float()
        keep.append(d)
        ptrs.append(d.data_ptr())
    a = _Args()
    a.q = q.data_ptr(); a.kc = k.data_ptr(); a.vc = v.data_ptr(); a.out = out.data_ptr()
    a.block_table = block_table.data_ptr(); a.seq_lens = seqused_k.data_ptr(); a.cu_q = cu_seqlens_q.data_ptr()
    a.q_descale, a.k_descale, a.v_descale = ptrs
    a.q_s0, a.q_s1 = q.stride(0), q.stride(1)
    a.o_s0, a.o_s1 = out.stride(0), out.stride(1)
    a.kv_s_blk, a.kv_s_tok, a.kv_s_head = k.stride(0), k.stride(1), k.stride(2)
    a.bt_stride = block_table.stride(0)
    num_seqs = seqused_k.shape[0]
    a.num_seqs = num_seqs
    a.scale = float(softmax_scale)
    a.blk_shift = (k.shape[1] // 64).bit_length() - 1
    nqb = q.shape[0] // 32 + num_seqs
    r = lib.ua512_launch(ctypes.byref(a), nqb, k.shape[2], ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), 1)
    if r != 0:
        raise RuntimeError(f"ua512_launch failed: {r}")
    return out
