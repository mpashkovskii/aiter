"""HIP fp8 prefill kernels for gfx950 (GQA 4, 64-token pages, no softcap, sinks or alibi):
head size 512 without a window and head size 256 with a 512-token sliding window.

The sources (ua_hd512_prefill.cu, ua_hd256_w512_prefill.cu, both generated from one template) are
compiled with hipcc on first use into a per-user cache and launched through ctypes on the current
torch stream. Switches: AITER_UA_HD512_HIP and AITER_UA_HD256_HIP (default 1).
"""
import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = {512: os.path.join(_DIR, "ua_hd512_prefill.cu"), 256: os.path.join(_DIR, "ua_hd256_w512_prefill.cu")}
_ENV = {512: "AITER_UA_HD512_HIP", 256: "AITER_UA_HD256_HIP"}
_WINDOW = {512: 0, 256: 512}
_FLAGS = ["-O3", "-w", "--offload-arch=gfx950", "-shared", "-fPIC"]
_BT_MAX_TILES = 3424
_libs = {}
_seen = set()


class _Args(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p), ("kc", ctypes.c_void_p), ("vc", ctypes.c_void_p), ("out", ctypes.c_void_p),
        ("block_table", ctypes.c_void_p), ("seq_lens", ctypes.c_void_p), ("cu_q", ctypes.c_void_p),
        ("q_descale", ctypes.c_void_p), ("k_descale", ctypes.c_void_p), ("v_descale", ctypes.c_void_p),
        ("q_s0", ctypes.c_longlong), ("q_s1", ctypes.c_longlong), ("o_s0", ctypes.c_longlong), ("o_s1", ctypes.c_longlong),
        ("kv_s_blk", ctypes.c_longlong), ("kv_s_tok", ctypes.c_longlong), ("kv_s_head", ctypes.c_longlong),
        ("bt_stride", ctypes.c_int), ("num_seqs", ctypes.c_int), ("scale", ctypes.c_float),
    ]


def enabled(head_size) -> bool:
    return os.environ.get(_ENV[head_size], "1") != "0"


def _build(head_size):
    src_path = _SRC[head_size]
    src = open(src_path, "rb").read()
    tag = hashlib.sha1(src + " ".join(_FLAGS).encode()).hexdigest()[:16]
    cache = os.path.join(os.environ.get("AITER_UA512_CACHE", os.path.expanduser("~/.cache/aiter_ua512")))
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f"ua_hd{head_size}_{tag}.so")
    if os.path.exists(so):
        return so
    with open(os.path.join(cache, f"{tag}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(so):
            hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
            tmp = f"{so}.{os.getpid()}.tmp"
            subprocess.run([hipcc, *_FLAGS, "-o", tmp, src_path], check=True, capture_output=True)
            os.replace(tmp, so)
    return so


def _load(head_size):
    lib = _libs.get(head_size)
    if lib is None:
        lib = ctypes.CDLL(_build(head_size))
        lib.ua512_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.ua512_launch.restype = ctypes.c_int
        _libs[head_size] = lib
    return lib


def supported(q, k, v, out, num_kv_heads, block_size, head_size, softcap, sinks, alibi_slopes, qq_bias,
              output_scale, sliding_window, max_seqlen_k) -> bool:
    return (
        head_size in _SRC
        and enabled(head_size)
        and sliding_window == _WINDOW[head_size]
        and q.shape[1] == 4 * num_kv_heads
        and block_size == 64
        and not softcap
        and sinks is None
        and alibi_slopes is None
        and qq_bias is None
        and output_scale is None
        and q.dtype == k.dtype == v.dtype
        and q.dtype.itemsize == 1
        and out.dtype == torch.bfloat16
        and q.stride(2) == 1 and out.stride(2) == 1
        and k.stride() == v.stride() and k.stride(3) == 1
        and (k.data_ptr() & 15) == 0 and (v.data_ptr() & 15) == 0
        and int(max_seqlen_k) <= _BT_MAX_TILES * 64
    )


def run(q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, q_descale, k_descale, v_descale):
    lib = _load(q.shape[2])
    if q.shape[2] not in _seen:
        _seen.add(q.shape[2])
        print(f"[aiter] ua_hip_prefill: HIP fp8 prefill kernel in use for head size {q.shape[2]}", flush=True)
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
    nqb = q.shape[0] // 32 + num_seqs
    r = lib.ua512_launch(ctypes.byref(a), nqb, k.shape[2], ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), 1)
    if r != 0:
        raise RuntimeError(f"ua512_launch failed: {r}")
    return out
