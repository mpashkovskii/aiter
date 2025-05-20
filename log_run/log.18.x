[aiter] WARNING: NUMA balancing is enabled, which may cause errors. It is recommended to disable NUMA balancing by running "sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'" for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing
[W527 04:47:30.915885614 collection.cpp:1100] Warning: ROCTracer produced duplicate flow start: 4 (function operator())
[aiter] type hints mismatch, override to --> wvSpltK(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: int, arg4: int) -> None
[aiter] [perf] dim: (4, 32, 8192)        dtype: torch.float16, torch avg: 10.33    us, B avg: 7.80     us, uplift: 32.4%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
[aiter] [perf] dim: (4, 32, 8192)        dtype: torch.bfloat16, torch avg: 10.09    us, B avg: 11.72    us, uplift: -13.9%[checkAllclose atol=0.01 rtol=0.01 [32mpassed~[0m]
