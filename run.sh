# export HIP_VISIBLE_DEVICES=0
# export HIP_VISIBLE_DEVICES=3
export HIP_VISIBLE_DEVICES=6
# export HIP_VISIBLE_DEVICES=7


# AITER_LOG_MORE=1 python3 ./op_tests/test_gemm.py > ./log_run/log.18.alm.x 2>&1
python3 ./op_tests/test_gemm.py > ./log_run/log.18.x 2>&1
# python3 ./op_tests/test_gemm.py > ./log_run/log.51.d6.wvSplitK_hf_sml_.v7 2>&1

# AITER_LOG_MORE=1 python3 ./op_tests/test_gemm.py > ./log_run/log.x 2>&1
# HIPCC_VERBOSE=1 python3 ./op_tests/test_gemm.py
# rocprof -d /home/yanguahe/heyanguang/code/aiter/res_rocprof --stats python3 ./op_tests/test_gemm.py
# rocprof --stats python3 ./op_tests/test_gemm.py
# rocprof -i ./prof.txt python3 ./op_tests/test_gemm.py


# python3 ./op_tests/test_gemm.py > ./log_run/log.x 2>&1
# python3 ./op_tests/test_gemm.py > ./log_run/log.d0.modi.v1 2>&1
# python3 ./op_tests/test_gemm.py > ./log_run/log.d0.modi.v2 2>&1


# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s

# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s -k "False-True-dtype1-8-128-2048-512-127-64"
# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s -k "False-True-dtype1-8-128-2050-512-127-64"
# pytest ./op_tests/triton_tests/test_mla_decode_rope.py::test_op_fwd_rope_integration -v -s -k "False-True-dtype1-8-128-2050-512-128-64"


# export root_dir=/mnt/raid0/heyanguang/code/
# rocprofv2 -d $root_dir/ttv_dir/ -i $root_dir/att.txt --plugin att auto --mode file,csv python3 ./op_tests/test_gemm.py
