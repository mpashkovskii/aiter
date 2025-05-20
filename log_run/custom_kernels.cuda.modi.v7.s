	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.protected	_Z20matrixMultiplySharedPfS_S_iiiiii ; -- Begin function _Z20matrixMultiplySharedPfS_S_iiiiii
	.globl	_Z20matrixMultiplySharedPfS_S_iiiiii
	.p2align	8
	.type	_Z20matrixMultiplySharedPfS_S_iiiiii,@function
_Z20matrixMultiplySharedPfS_S_iiiiii:   ; @_Z20matrixMultiplySharedPfS_S_iiiiii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_add_u32 s0, s0, 48
	s_addc_u32 s1, s1, 0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_u32 s15, s17
	s_cselect_b32 s17, 14, 20
	v_mov_b32_e32 v1, s17
	s_cmp_lt_u32 s14, s16
	s_cselect_b32 s16, 12, 18
	global_load_ushort v2, v1, s[0:1]
	v_mov_b32_e32 v1, s16
	global_load_ushort v4, v1, s[0:1]
	v_mov_b32_e32 v1, 0
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v6, 0x3ff, v0
	v_lshlrev_b32_e32 v0, 2, v6
	v_lshlrev_b32_e32 v7, 7, v3
	v_mul_lo_u32 v5, v3, s11
	v_add_u32_e32 v9, v7, v0
	v_or_b32_e32 v10, 0x1000, v0
	v_add_u32_e32 v11, v10, v7
	v_add_u32_e32 v12, 0x400, v10
	v_add_u32_e32 v13, 0x800, v10
	v_add_u32_e32 v14, 0xc00, v10
	s_add_i32 s0, s9, -1
	s_lshl_b32 s20, s11, 5
	s_lshr_b32 s0, s0, 5
	s_add_i32 s21, s0, 1
	s_waitcnt vmcnt(1)
	v_mul_lo_u32 v0, s15, v2
	v_add_u32_e32 v8, v0, v3
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v0, s14, v4
	v_add_u32_e32 v2, v0, v6
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_mul_lo_u32 v15, v8, s9
	v_add3_u32 v4, v6, v5, v0
	v_cmp_gt_i32_e64 s[0:1], s11, v2
	v_mov_b32_e32 v16, 0
	s_branch .LBB0_3
.LBB0_1:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b32 v11, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[34:35], v10 offset1:32
	ds_read_b128 v[18:21], v7
	ds_read_b128 v[22:25], v7 offset:16
	ds_read2_b32 v[36:37], v10 offset0:64 offset1:96
	ds_read_b128 v[26:29], v7 offset:32
	ds_read_b128 v[30:33], v7 offset:48
	s_waitcnt lgkmcnt(4)
	v_fmac_f32_e32 v16, v18, v34
	ds_read2_b32 v[38:39], v10 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v19, v35
	s_waitcnt lgkmcnt(3)
	v_fmac_f32_e32 v16, v20, v36
	ds_read2_b32 v[18:19], v10 offset0:192 offset1:224
	v_fmac_f32_e32 v16, v21, v37
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v22, v38
	ds_read2_b32 v[20:21], v12 offset1:32
	v_fmac_f32_e32 v16, v23, v39
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v24, v18
	ds_read2_b32 v[22:23], v12 offset0:64 offset1:96
	v_fmac_f32_e32 v16, v25, v19
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v26, v20
	ds_read2_b32 v[18:19], v12 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v27, v21
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v28, v22
	v_fmac_f32_e32 v16, v29, v23
	ds_read2_b32 v[22:23], v12 offset0:192 offset1:224
	s_waitcnt lgkmcnt(1)
	v_fmac_f32_e32 v16, v30, v18
	v_fmac_f32_e32 v16, v31, v19
	ds_read2_b32 v[26:27], v13 offset1:32
	ds_read_b128 v[18:21], v7 offset:64
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v32, v22
	v_fmac_f32_e32 v16, v33, v23
	ds_read2_b32 v[28:29], v13 offset0:64 offset1:96
	ds_read_b128 v[22:25], v7 offset:80
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v18, v26
	ds_read2_b32 v[30:31], v13 offset0:128 offset1:160
	v_fmac_f32_e32 v16, v19, v27
	s_waitcnt lgkmcnt(2)
	v_fmac_f32_e32 v16, v20, v28
	ds_read2_b32 v[18:19], v13 offset0:192 offset1:224
	v_fmac_f32_e32 v16, v21, v29
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[20:21], v[22:23], v[30:31]
	s_nop 0
	v_add_f32_e32 v0, v16, v20
	v_add_f32_e32 v0, v0, v21
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[20:21], v[24:25], v[18:19]
	ds_read2_b32 v[24:25], v14 offset1:32
	ds_read_b128 v[16:19], v7 offset:96
	v_add_f32_e32 v0, v0, v20
	v_add_f32_e32 v0, v0, v21
	ds_read2_b32 v[26:27], v14 offset0:64 offset1:96
	ds_read_b128 v[20:23], v7 offset:112
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[16:17], v[16:17], v[24:25]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v0, v0, v17
	ds_read2_b32 v[16:17], v14 offset0:128 offset1:160
	s_waitcnt lgkmcnt(2)
	v_pk_mul_f32 v[18:19], v[18:19], v[26:27]
	s_nop 0
	v_add_f32_e32 v0, v0, v18
	ds_read2_b32 v[24:25], v14 offset0:192 offset1:224
	v_add_f32_e32 v0, v0, v19
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f32 v[16:17], v[20:21], v[16:17]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v0, v0, v17
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[16:17], v[22:23], v[24:25]
	s_nop 0
	v_add_f32_e32 v0, v0, v16
	v_add_f32_e32 v16, v0, v17
	s_add_i32 s21, s21, -1
	v_add_u32_e32 v6, 32, v6
	v_add_u32_e32 v4, s20, v4
	s_cmp_eq_u32 s21, 0
	v_add_u32_e32 v3, 32, v3
	s_cbranch_scc1 .LBB0_10
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_7
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_gt_u32_e64 s[14:15], s9, v6
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[18:19], s[14:15]
	s_cbranch_execz .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	v_add_u32_e32 v0, v15, v6
	v_lshl_add_u64 v[18:19], v[0:1], 2, s[2:3]
	global_load_dword v0, v[18:19], off
.LBB0_6:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[16:17]
	s_waitcnt vmcnt(0)
	ds_write_b32 v9, v0
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_gt_u32_e64 s[14:15], s10, v3
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[18:19], s[14:15]
	s_cbranch_execz .LBB0_1
; %bb.9:                                ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v5, v1
	v_lshl_add_u64 v[18:19], v[4:5], 2, s[4:5]
	global_load_dword v0, v[18:19], off
	s_branch .LBB0_1
.LBB0_10:
	v_cmp_gt_i32_e32 vcc, s12, v8
	v_cmp_gt_i32_e64 s[0:1], s13, v2
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_mad_u64_u32 v[0:1], s[0:1], v8, s13, v[2:3]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	global_store_dword v[0:1], v16, off
.LBB0_12:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20matrixMultiplySharedPfS_S_iiiiii
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  12
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 22
		.amdhsa_accum_offset 40
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z20matrixMultiplySharedPfS_S_iiiiii, .Lfunc_end0-_Z20matrixMultiplySharedPfS_S_iiiiii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 908
; NumSgprs: 28
; NumVgprs: 40
; NumAgprs: 0
; TotalNumVgprs: 40
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 8192 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 40
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v96, 3, v2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB1_6
; %bb.1:
	s_lshl_b32 s0, s2, 2
	s_min_i32 s20, s0, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v96
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB1_5
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v5
	v_cmp_gt_u32_e32 vcc, s20, v0
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s20
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[16:17], s[16:17], s[22:23]
	s_branch .LBB1_2
.LBB1_5:
	s_or_b64 exec, exec, s[0:1]
.LBB1_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_64
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v98, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v98
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_64
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s33, s11, s10
	v_mad_u64_u32 v[100:101], s[6:7], s2, v98, v[96:97]
	s_mul_i32 s40, s33, s2
	v_lshlrev_b32_e32 v97, 4, v2
	s_mov_b64 s[12:13], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v103, 0
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
                                        ; implicit-def: $vgpr50_vgpr51
                                        ; implicit-def: $vgpr54_vgpr55
                                        ; implicit-def: $vgpr58_vgpr59
                                        ; implicit-def: $vgpr62_vgpr63
                                        ; implicit-def: $vgpr66_vgpr67
                                        ; implicit-def: $vgpr70_vgpr71
                                        ; implicit-def: $vgpr74_vgpr75
                                        ; implicit-def: $vgpr78_vgpr79
                                        ; implicit-def: $vgpr82_vgpr83
                                        ; implicit-def: $vgpr86_vgpr87
                                        ; implicit-def: $vgpr90_vgpr91
                                        ; implicit-def: $vgpr94_vgpr95
	s_branch .LBB1_10
.LBB1_9:                                ;   in Loop: Header=BB1_10 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v98, s33, v98
	v_cmp_le_u32_e32 vcc, s3, v98
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v100, s40, v100
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB1_64
.LBB1_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_27 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB1_12
; %bb.11:                               ;   in Loop: Header=BB1_10 Depth=1
	s_mov_b32 s41, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v101, v97
	s_branch .LBB1_27
.LBB1_12:                               ;   in Loop: Header=BB1_10 Depth=1
	v_mov_b32_e32 v99, v103
.LBB1_13:                               ;   in Loop: Header=BB1_10 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	v_mov_b32_e32 v101, v103
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v103, v103 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB1_9
; %bb.14:                               ;   in Loop: Header=BB1_10 Depth=1
	v_cvt_f16_f32_e32 v102, v99
	v_mov_b32_e32 v99, v103
	v_cvt_f16_f32_e32 v101, v101
	v_lshl_add_u64 v[104:105], v[98:99], 1, s[8:9]
	global_store_short v[104:105], v102, off
	v_add_u32_e32 v102, s3, v98
	v_lshl_add_u64 v[104:105], v[102:103], 1, s[8:9]
	global_store_short v[104:105], v101, off
	v_add_u32_e32 v102, s3, v102
	v_lshl_add_u64 v[104:105], v[102:103], 1, s[8:9]
	global_store_short v[104:105], v101, off
	v_add_u32_e32 v102, s3, v102
	v_lshl_add_u64 v[104:105], v[102:103], 1, s[8:9]
	global_store_short v[104:105], v101, off
	s_branch .LBB1_9
.LBB1_15:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[36:37]
.LBB1_16:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[34:35]
.LBB1_17:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB1_18:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB1_19:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB1_20:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB1_21:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB1_22:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB1_23:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_24:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB1_25:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB1_26:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s41, 0x1800
	s_cmp_ge_u32 s41, s2
	v_add_u32_e32 v101, 0x3000, v101
	s_cbranch_scc1 .LBB1_13
.LBB1_27:                               ;   Parent Loop BB1_10 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v104, s41, v96
	v_cmp_gt_u32_e32 vcc, s2, v104
	v_add_u32_e32 v105, 0x200, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_51
; %bb.28:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v102, s41, v100
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[44:45], v[102:103], 1, s[4:5]
	global_load_dwordx4 v[44:47], v[44:45], off nt
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[92:95], v101
	v_cmp_gt_u32_e64 s[10:11], s2, v105
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB1_50
; %bb.29:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v40, 0x200, v102
	v_mov_b32_e32 v41, v103
	v_lshl_add_u64 v[40:41], v[40:41], 1, s[4:5]
	global_load_dwordx4 v[40:43], v[40:41], off nt
	ds_read_b128 v[88:91], v101 offset:1024
	v_add_u32_e32 v106, 0x400, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB1_49
; %bb.30:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v36, 0x400, v102
	v_mov_b32_e32 v37, v103
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[4:5]
	global_load_dwordx4 v[36:39], v[36:37], off nt
	ds_read_b128 v[84:87], v101 offset:2048
	v_add_u32_e32 v106, 0x600, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB1_48
; %bb.31:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v32, 0x600, v102
	v_mov_b32_e32 v33, v103
	v_lshl_add_u64 v[32:33], v[32:33], 1, s[4:5]
	global_load_dwordx4 v[32:35], v[32:33], off nt
	ds_read_b128 v[80:83], v101 offset:3072
	v_add_u32_e32 v106, 0x800, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_cbranch_execz .LBB1_47
; %bb.32:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v28, 0x800, v102
	v_mov_b32_e32 v29, v103
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[4:5]
	global_load_dwordx4 v[28:31], v[28:29], off nt
	ds_read_b128 v[76:79], v101 offset:4096
	v_add_u32_e32 v106, 0xa00, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[24:25], s[10:11]
	s_cbranch_execz .LBB1_46
; %bb.33:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v24, 0xa00, v102
	v_mov_b32_e32 v25, v103
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[4:5]
	global_load_dwordx4 v[24:27], v[24:25], off nt
	ds_read_b128 v[72:75], v101 offset:5120
	v_add_u32_e32 v106, 0xc00, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[26:27], s[10:11]
	s_cbranch_execz .LBB1_45
; %bb.34:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v20, 0xc00, v102
	v_mov_b32_e32 v21, v103
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[4:5]
	global_load_dwordx4 v[20:23], v[20:21], off nt
	ds_read_b128 v[68:71], v101 offset:6144
	v_add_u32_e32 v106, 0xe00, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[28:29], s[10:11]
	s_cbranch_execz .LBB1_44
; %bb.35:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v16, 0xe00, v102
	v_mov_b32_e32 v17, v103
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[4:5]
	global_load_dwordx4 v[16:19], v[16:17], off nt
	ds_read_b128 v[64:67], v101 offset:7168
	v_add_u32_e32 v106, 0x1000, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[30:31], s[10:11]
	s_cbranch_execz .LBB1_43
; %bb.36:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v12, 0x1000, v102
	v_mov_b32_e32 v13, v103
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	ds_read_b128 v[60:63], v101 offset:8192
	v_add_u32_e32 v106, 0x1200, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[34:35], s[10:11]
	s_cbranch_execz .LBB1_42
; %bb.37:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v8, 0x1200, v102
	v_mov_b32_e32 v9, v103
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	ds_read_b128 v[56:59], v101 offset:9216
	v_add_u32_e32 v106, 0x1400, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[36:37], s[10:11]
	s_cbranch_execz .LBB1_41
; %bb.38:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v4, 0x1400, v102
	v_mov_b32_e32 v5, v103
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	ds_read_b128 v[52:55], v101 offset:10240
	v_add_u32_e32 v106, 0x1600, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[38:39], s[10:11]
	s_cbranch_execz .LBB1_40
; %bb.39:                               ;   in Loop: Header=BB1_27 Depth=2
	v_add_u32_e32 v102, 0x1600, v102
	v_lshl_add_u64 v[0:1], v[102:103], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	ds_read_b128 v[48:51], v101 offset:11264
.LBB1_40:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB1_41:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[36:37]
.LBB1_42:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[34:35]
.LBB1_43:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB1_44:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB1_45:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB1_46:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB1_47:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB1_48:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB1_49:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB1_50:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB1_51:                               ;   in Loop: Header=BB1_27 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB1_26
; %bb.52:                               ;   in Loop: Header=BB1_27 Depth=2
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_sdwa v107, v44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v44
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_sdwa v109, v92 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v92
	v_cvt_f32_f16_sdwa v111, v93 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v93
	v_cvt_f32_f16_sdwa v113, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v45
	v_cvt_f32_f16_sdwa v115, v94 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v94
	v_cvt_f32_f16_sdwa v117, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v46
	v_cvt_f32_f16_sdwa v119, v95 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v95
	v_cvt_f32_f16_sdwa v121, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v47
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_cmp_gt_u32_e32 vcc, s2, v105
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_25
; %bb.53:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v40 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v40
	v_cvt_f32_f16_sdwa v109, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v88
	v_cvt_f32_f16_sdwa v111, v89 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v89
	v_cvt_f32_f16_sdwa v113, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v41
	v_cvt_f32_f16_sdwa v115, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v90
	v_cvt_f32_f16_sdwa v117, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v42
	v_cvt_f32_f16_sdwa v119, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v91
	v_cvt_f32_f16_sdwa v121, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v43
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x400, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB1_24
; %bb.54:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v36
	v_cvt_f32_f16_sdwa v109, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v84
	v_cvt_f32_f16_sdwa v111, v85 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v85
	v_cvt_f32_f16_sdwa v113, v37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v37
	v_cvt_f32_f16_sdwa v115, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v86
	v_cvt_f32_f16_sdwa v117, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v38
	v_cvt_f32_f16_sdwa v119, v87 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v87
	v_cvt_f32_f16_sdwa v121, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v39
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x600, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB1_23
; %bb.55:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v32
	v_cvt_f32_f16_sdwa v109, v80 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v80
	v_cvt_f32_f16_sdwa v111, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v81
	v_cvt_f32_f16_sdwa v113, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v33
	v_cvt_f32_f16_sdwa v115, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v82
	v_cvt_f32_f16_sdwa v117, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v34
	v_cvt_f32_f16_sdwa v119, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v83
	v_cvt_f32_f16_sdwa v121, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v35
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x800, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB1_22
; %bb.56:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v28
	v_cvt_f32_f16_sdwa v109, v76 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v76
	v_cvt_f32_f16_sdwa v111, v77 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v77
	v_cvt_f32_f16_sdwa v113, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v29
	v_cvt_f32_f16_sdwa v115, v78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v78
	v_cvt_f32_f16_sdwa v117, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v30
	v_cvt_f32_f16_sdwa v119, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v79
	v_cvt_f32_f16_sdwa v121, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v31
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0xa00, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB1_21
; %bb.57:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v24
	v_cvt_f32_f16_sdwa v109, v72 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v72
	v_cvt_f32_f16_sdwa v111, v73 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v73
	v_cvt_f32_f16_sdwa v113, v25 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v25
	v_cvt_f32_f16_sdwa v115, v74 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v74
	v_cvt_f32_f16_sdwa v117, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v26
	v_cvt_f32_f16_sdwa v119, v75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v75
	v_cvt_f32_f16_sdwa v121, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v27
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0xc00, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB1_20
; %bb.58:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v20
	v_cvt_f32_f16_sdwa v109, v68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v68
	v_cvt_f32_f16_sdwa v111, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v69
	v_cvt_f32_f16_sdwa v113, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v21
	v_cvt_f32_f16_sdwa v115, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v70
	v_cvt_f32_f16_sdwa v117, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v22
	v_cvt_f32_f16_sdwa v119, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v71
	v_cvt_f32_f16_sdwa v121, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v23
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0xe00, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB1_19
; %bb.59:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v16
	v_cvt_f32_f16_sdwa v109, v64 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v64
	v_cvt_f32_f16_sdwa v111, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v65
	v_cvt_f32_f16_sdwa v113, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v17
	v_cvt_f32_f16_sdwa v115, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v66
	v_cvt_f32_f16_sdwa v117, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v18
	v_cvt_f32_f16_sdwa v119, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v67
	v_cvt_f32_f16_sdwa v121, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v19
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1000, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB1_18
; %bb.60:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v12
	v_cvt_f32_f16_sdwa v109, v60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v60
	v_cvt_f32_f16_sdwa v111, v61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v61
	v_cvt_f32_f16_sdwa v113, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v13
	v_cvt_f32_f16_sdwa v115, v62 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v62
	v_cvt_f32_f16_sdwa v117, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v14
	v_cvt_f32_f16_sdwa v119, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v63
	v_cvt_f32_f16_sdwa v121, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v15
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1200, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB1_17
; %bb.61:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v8
	v_cvt_f32_f16_sdwa v109, v56 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v56
	v_cvt_f32_f16_sdwa v111, v57 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v57
	v_cvt_f32_f16_sdwa v113, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v9
	v_cvt_f32_f16_sdwa v115, v58 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v58
	v_cvt_f32_f16_sdwa v117, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v10
	v_cvt_f32_f16_sdwa v119, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v59
	v_cvt_f32_f16_sdwa v121, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v11
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1400, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB1_16
; %bb.62:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v107, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v4
	v_cvt_f32_f16_sdwa v109, v52 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v52
	v_cvt_f32_f16_sdwa v111, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v53
	v_cvt_f32_f16_sdwa v113, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v5
	v_cvt_f32_f16_sdwa v115, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v54
	v_cvt_f32_f16_sdwa v117, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v6
	v_cvt_f32_f16_sdwa v119, v55 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v55
	v_cvt_f32_f16_sdwa v121, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v120, v7
	v_pk_mul_f32 v[106:107], v[108:109], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[110:111], v[112:113], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[114:115], v[116:117], v[106:107]
	s_nop 0
	v_pk_fma_f32 v[106:107], v[118:119], v[120:121], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1600, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB1_15
; %bb.63:                               ;   in Loop: Header=BB1_27 Depth=2
	v_cvt_f32_f16_sdwa v105, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v104, v0
	v_cvt_f32_f16_sdwa v107, v48 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v106, v48
	v_cvt_f32_f16_sdwa v109, v49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v108, v49
	v_cvt_f32_f16_sdwa v111, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v110, v1
	v_cvt_f32_f16_sdwa v113, v50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v112, v50
	v_cvt_f32_f16_sdwa v115, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v114, v2
	v_cvt_f32_f16_sdwa v117, v51 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v116, v51
	v_cvt_f32_f16_sdwa v119, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
	v_cvt_f32_f16_e32 v118, v3
	v_pk_mul_f32 v[104:105], v[106:107], v[104:105]
	s_nop 0
	v_pk_fma_f32 v[104:105], v[108:109], v[110:111], v[104:105]
	s_nop 0
	v_pk_fma_f32 v[104:105], v[112:113], v[114:115], v[104:105]
	s_nop 0
	v_pk_fma_f32 v[104:105], v[116:117], v[118:119], v[104:105]
	s_nop 0
	v_add_f32_e32 v102, v104, v105
	v_add_f32_e32 v99, v99, v102
	s_branch .LBB1_15
.LBB1_64:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 122
		.amdhsa_next_free_sgpr 42
		.amdhsa_accum_offset 124
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end1:
	.size	_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end1-_Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3564
; NumSgprs: 48
; NumVgprs: 122
; NumAgprs: 0
; TotalNumVgprs: 122
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 122
; AccumOffset: 124
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 30
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,comdat
	.protected	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii ; -- Begin function _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
	.globl	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
	.p2align	8
	.type	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,@function
_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii: ; @_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
	s_trap 2 ; Kernarg preload header. Trap with incompatible firmware that doesn't support preloading kernel arguments.
	.fill 63, 4, 0xbf800000 ; s_nop 0
; %bb.0:
	v_bfe_u32 v3, v0, 10, 10
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v96, 3, v2
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[14:15], -1, 0
	s_cmp_eq_u32 s2, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB2_6
; %bb.1:
	s_lshl_b32 s0, s2, 2
	s_min_i32 s20, s0, 0x8000
	v_lshlrev_b32_e32 v0, 4, v2
	v_lshl_add_u32 v4, v3, 10, v0
	v_lshl_add_u32 v5, v3, 9, v96
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v1, 0
                                        ; implicit-def: $sgpr16_sgpr17
	s_branch .LBB2_3
.LBB2_2:                                ;   in Loop: Header=BB2_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], exec, s[16:17]
	s_or_b64 s[0:1], s[18:19], s[0:1]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_5
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v0, s13, v5
	v_cmp_gt_u32_e32 vcc, s20, v0
	s_or_b64 s[16:17], s[16:17], exec
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	v_lshl_add_u64 v[6:7], v[0:1], 1, s[6:7]
	global_load_dwordx4 v[6:9], v[6:7], off
	s_addk_i32 s13, 0x2000
	s_cmp_ge_u32 s13, s20
	s_cselect_b64 s[22:23], -1, 0
	s_andn2_b64 s[16:17], s[16:17], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[6:9]
	v_add_u32_e32 v4, 0x4000, v4
	s_or_b64 s[16:17], s[16:17], s[22:23]
	s_branch .LBB2_2
.LBB2_5:
	s_or_b64 exec, exec, s[0:1]
.LBB2_6:
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_cmp_gt_u32_e32 vcc, s10, v3
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_80
; %bb.7:
	s_mul_i32 s12, s12, s10
	v_add_u32_e32 v98, s12, v3
	v_cmp_gt_u32_e32 vcc, s3, v98
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_80
; %bb.8:
	v_cmp_eq_u32_e64 s[0:1], 63, v2
	s_mul_i32 s33, s11, s10
	v_mad_u64_u32 v[100:101], s[6:7], s2, v98, v[96:97]
	s_mul_i32 s40, s33, s2
	v_lshlrev_b32_e32 v97, 4, v2
	s_mov_b64 s[12:13], 0
	v_cndmask_b32_e64 v0, 0, 1, s[14:15]
	v_cmp_ne_u32_e64 s[6:7], 1, v0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s41, 0x7f800000
	s_movk_i32 s42, 0x7fff
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
                                        ; implicit-def: $vgpr51
                                        ; implicit-def: $vgpr55
                                        ; implicit-def: $vgpr59
                                        ; implicit-def: $vgpr63
                                        ; implicit-def: $vgpr67
                                        ; implicit-def: $vgpr71
                                        ; implicit-def: $vgpr75
                                        ; implicit-def: $vgpr79
                                        ; implicit-def: $vgpr83
                                        ; implicit-def: $vgpr87
                                        ; implicit-def: $vgpr91
                                        ; implicit-def: $vgpr95
	s_branch .LBB2_11
.LBB2_9:                                ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v102, s3, v102
	v_lshl_add_u64 v[104:105], v[102:103], 1, s[8:9]
	global_store_short_d16_hi v[104:105], v99, off
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[14:15]
	v_add_u32_e32 v98, s33, v98
	v_cmp_le_u32_e32 vcc, s3, v98
	s_or_b64 s[12:13], vcc, s[12:13]
	v_add_u32_e32 v100, s40, v100
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execz .LBB2_80
.LBB2_11:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_43 Depth 2
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB2_13
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=1
	s_mov_b32 s43, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v101, v97
	s_branch .LBB2_43
.LBB2_13:                               ;   in Loop: Header=BB2_11 Depth=1
	v_mov_b32_e32 v99, v103
.LBB2_14:                               ;   in Loop: Header=BB2_11 Depth=1
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v99, v99, v99 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	v_mov_b32_e32 v101, v103
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v103, v103 row_shr:8 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_shr:4 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_shr:2 bound_ctrl:0 
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 wave_shr:1 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_bcast:15 bound_ctrl:0
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_nop 0
	v_add_f32 v101, v101, v101 row_bcast:31 bound_ctrl:0
	;;#ASMEND
	s_and_saveexec_b64 s[14:15], s[0:1]
	s_cbranch_execz .LBB2_10
; %bb.15:                               ;   in Loop: Header=BB2_11 Depth=1
	v_and_b32_e32 v102, 0x7f800000, v99
	v_cmp_ne_u32_e32 vcc, s41, v102
                                        ; implicit-def: $vgpr102
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
; %bb.16:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v102, v99, 16, 1
	v_add3_u32 v102, v99, v102, s42
                                        ; implicit-def: $vgpr99
; %bb.17:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[10:11], s[10:11]
; %bb.18:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v102, 0x10000, v99
	v_cmp_eq_u32_sdwa vcc, v99, v103 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v102, v102, v99, vcc
; %bb.19:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v99, v103
	v_lshl_add_u64 v[104:105], v[98:99], 1, s[8:9]
	global_store_short_d16_hi v[104:105], v102, off
	v_and_b32_e32 v99, 0x7f800000, v101
	v_cmp_ne_u32_e32 vcc, s41, v99
                                        ; implicit-def: $vgpr99
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
; %bb.20:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v99, v101, 16, 1
	v_add3_u32 v99, v101, v99, s42
; %bb.21:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[16:17], s[10:11]
; %bb.22:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v99, 0x10000, v101
	v_cmp_eq_u32_sdwa s[10:11], v101, v103 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e64 v99, v99, v101, s[10:11]
; %bb.23:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_add_u32_e32 v102, s3, v98
	v_lshl_add_u64 v[104:105], v[102:103], 1, s[8:9]
	global_store_short_d16_hi v[104:105], v99, off
                                        ; implicit-def: $vgpr99
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
; %bb.24:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v99, v101, 16, 1
	v_add3_u32 v99, v101, v99, s42
; %bb.25:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[16:17], s[10:11]
; %bb.26:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v99, 0x10000, v101
	v_cmp_eq_u32_sdwa s[10:11], v101, v103 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e64 v99, v99, v101, s[10:11]
; %bb.27:                               ;   in Loop: Header=BB2_11 Depth=1
	s_or_b64 exec, exec, s[16:17]
	v_add_u32_e32 v102, s3, v102
	v_lshl_add_u64 v[104:105], v[102:103], 1, s[8:9]
	global_store_short_d16_hi v[104:105], v99, off
                                        ; implicit-def: $vgpr99
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
; %bb.28:                               ;   in Loop: Header=BB2_11 Depth=1
	v_bfe_u32 v99, v101, 16, 1
	v_add3_u32 v99, v101, v99, s42
                                        ; implicit-def: $vgpr101
; %bb.29:                               ;   in Loop: Header=BB2_11 Depth=1
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB2_9
; %bb.30:                               ;   in Loop: Header=BB2_11 Depth=1
	v_or_b32_e32 v99, 0x10000, v101
	v_cmp_eq_u32_sdwa vcc, v101, v103 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v99, v99, v101, vcc
	s_branch .LBB2_9
.LBB2_31:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[36:37]
.LBB2_32:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[34:35]
.LBB2_33:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB2_34:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB2_35:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB2_36:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB2_37:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB2_38:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_39:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_40:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB2_41:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[14:15]
.LBB2_42:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[10:11]
	s_addk_i32 s43, 0x1800
	s_cmp_ge_u32 s43, s2
	v_add_u32_e32 v101, 0x3000, v101
	s_cbranch_scc1 .LBB2_14
.LBB2_43:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v104, s43, v96
	v_cmp_gt_u32_e32 vcc, s2, v104
	v_add_u32_e32 v105, 0x200, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_67
; %bb.44:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v102, s43, v100
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[44:45], v[102:103], 1, s[4:5]
	global_load_dwordx4 v[44:47], v[44:45], off nt
	s_waitcnt lgkmcnt(0)
	ds_read_b128 v[92:95], v101
	v_cmp_gt_u32_e64 s[10:11], s2, v105
	s_and_saveexec_b64 s[16:17], s[10:11]
	s_cbranch_execz .LBB2_66
; %bb.45:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v40, 0x200, v102
	v_mov_b32_e32 v41, v103
	v_lshl_add_u64 v[40:41], v[40:41], 1, s[4:5]
	global_load_dwordx4 v[40:43], v[40:41], off nt
	ds_read_b128 v[88:91], v101 offset:1024
	v_add_u32_e32 v106, 0x400, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[18:19], s[10:11]
	s_cbranch_execz .LBB2_65
; %bb.46:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v36, 0x400, v102
	v_mov_b32_e32 v37, v103
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[4:5]
	global_load_dwordx4 v[36:39], v[36:37], off nt
	ds_read_b128 v[84:87], v101 offset:2048
	v_add_u32_e32 v106, 0x600, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[20:21], s[10:11]
	s_cbranch_execz .LBB2_64
; %bb.47:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v32, 0x600, v102
	v_mov_b32_e32 v33, v103
	v_lshl_add_u64 v[32:33], v[32:33], 1, s[4:5]
	global_load_dwordx4 v[32:35], v[32:33], off nt
	ds_read_b128 v[80:83], v101 offset:3072
	v_add_u32_e32 v106, 0x800, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[22:23], s[10:11]
	s_cbranch_execz .LBB2_63
; %bb.48:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v28, 0x800, v102
	v_mov_b32_e32 v29, v103
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[4:5]
	global_load_dwordx4 v[28:31], v[28:29], off nt
	ds_read_b128 v[76:79], v101 offset:4096
	v_add_u32_e32 v106, 0xa00, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[24:25], s[10:11]
	s_cbranch_execz .LBB2_62
; %bb.49:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v24, 0xa00, v102
	v_mov_b32_e32 v25, v103
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[4:5]
	global_load_dwordx4 v[24:27], v[24:25], off nt
	ds_read_b128 v[72:75], v101 offset:5120
	v_add_u32_e32 v106, 0xc00, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[26:27], s[10:11]
	s_cbranch_execz .LBB2_61
; %bb.50:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v20, 0xc00, v102
	v_mov_b32_e32 v21, v103
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[4:5]
	global_load_dwordx4 v[20:23], v[20:21], off nt
	ds_read_b128 v[68:71], v101 offset:6144
	v_add_u32_e32 v106, 0xe00, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[28:29], s[10:11]
	s_cbranch_execz .LBB2_60
; %bb.51:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v16, 0xe00, v102
	v_mov_b32_e32 v17, v103
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[4:5]
	global_load_dwordx4 v[16:19], v[16:17], off nt
	ds_read_b128 v[64:67], v101 offset:7168
	v_add_u32_e32 v106, 0x1000, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[30:31], s[10:11]
	s_cbranch_execz .LBB2_59
; %bb.52:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v12, 0x1000, v102
	v_mov_b32_e32 v13, v103
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[4:5]
	global_load_dwordx4 v[12:15], v[12:13], off nt
	ds_read_b128 v[60:63], v101 offset:8192
	v_add_u32_e32 v106, 0x1200, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[34:35], s[10:11]
	s_cbranch_execz .LBB2_58
; %bb.53:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v8, 0x1200, v102
	v_mov_b32_e32 v9, v103
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	global_load_dwordx4 v[8:11], v[8:9], off nt
	ds_read_b128 v[56:59], v101 offset:9216
	v_add_u32_e32 v106, 0x1400, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[36:37], s[10:11]
	s_cbranch_execz .LBB2_57
; %bb.54:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v4, 0x1400, v102
	v_mov_b32_e32 v5, v103
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off nt
	ds_read_b128 v[52:55], v101 offset:10240
	v_add_u32_e32 v106, 0x1600, v104
	v_cmp_gt_u32_e64 s[10:11], s2, v106
	s_and_saveexec_b64 s[38:39], s[10:11]
	s_cbranch_execz .LBB2_56
; %bb.55:                               ;   in Loop: Header=BB2_43 Depth=2
	v_add_u32_e32 v102, 0x1600, v102
	v_lshl_add_u64 v[0:1], v[102:103], 1, s[4:5]
	global_load_dwordx4 v[0:3], v[0:1], off nt
	ds_read_b128 v[48:51], v101 offset:11264
.LBB2_56:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[38:39]
.LBB2_57:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[36:37]
.LBB2_58:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[34:35]
.LBB2_59:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[30:31]
.LBB2_60:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[28:29]
.LBB2_61:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[26:27]
.LBB2_62:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[24:25]
.LBB2_63:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[22:23]
.LBB2_64:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[20:21]
.LBB2_65:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[18:19]
.LBB2_66:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[16:17]
.LBB2_67:                               ;   in Loop: Header=BB2_43 Depth=2
	s_or_b64 exec, exec, s[14:15]
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_42
; %bb.68:                               ;   in Loop: Header=BB2_43 Depth=2
	s_waitcnt lgkmcnt(0)
	v_and_b32_e32 v107, 0xffff0000, v92
	v_lshlrev_b32_e32 v106, 16, v92
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v109, 0xffff0000, v44
	v_lshlrev_b32_e32 v108, 16, v44
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v93
	v_lshlrev_b32_e32 v108, 16, v93
	v_and_b32_e32 v111, 0xffff0000, v45
	v_lshlrev_b32_e32 v110, 16, v45
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v94
	v_lshlrev_b32_e32 v108, 16, v94
	v_and_b32_e32 v111, 0xffff0000, v46
	v_lshlrev_b32_e32 v110, 16, v46
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v95
	v_lshlrev_b32_e32 v108, 16, v95
	v_and_b32_e32 v111, 0xffff0000, v47
	v_lshlrev_b32_e32 v110, 16, v47
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_cmp_gt_u32_e32 vcc, s2, v105
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB2_41
; %bb.69:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v88
	v_lshlrev_b32_e32 v106, 16, v88
	v_and_b32_e32 v109, 0xffff0000, v40
	v_lshlrev_b32_e32 v108, 16, v40
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v89
	v_lshlrev_b32_e32 v108, 16, v89
	v_and_b32_e32 v111, 0xffff0000, v41
	v_lshlrev_b32_e32 v110, 16, v41
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v90
	v_lshlrev_b32_e32 v108, 16, v90
	v_and_b32_e32 v111, 0xffff0000, v42
	v_lshlrev_b32_e32 v110, 16, v42
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v91
	v_lshlrev_b32_e32 v108, 16, v91
	v_and_b32_e32 v111, 0xffff0000, v43
	v_lshlrev_b32_e32 v110, 16, v43
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x400, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB2_40
; %bb.70:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v84
	v_lshlrev_b32_e32 v106, 16, v84
	v_and_b32_e32 v109, 0xffff0000, v36
	v_lshlrev_b32_e32 v108, 16, v36
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v85
	v_lshlrev_b32_e32 v108, 16, v85
	v_and_b32_e32 v111, 0xffff0000, v37
	v_lshlrev_b32_e32 v110, 16, v37
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v86
	v_lshlrev_b32_e32 v108, 16, v86
	v_and_b32_e32 v111, 0xffff0000, v38
	v_lshlrev_b32_e32 v110, 16, v38
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v87
	v_lshlrev_b32_e32 v108, 16, v87
	v_and_b32_e32 v111, 0xffff0000, v39
	v_lshlrev_b32_e32 v110, 16, v39
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x600, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB2_39
; %bb.71:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v80
	v_lshlrev_b32_e32 v106, 16, v80
	v_and_b32_e32 v109, 0xffff0000, v32
	v_lshlrev_b32_e32 v108, 16, v32
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v81
	v_lshlrev_b32_e32 v108, 16, v81
	v_and_b32_e32 v111, 0xffff0000, v33
	v_lshlrev_b32_e32 v110, 16, v33
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v82
	v_lshlrev_b32_e32 v108, 16, v82
	v_and_b32_e32 v111, 0xffff0000, v34
	v_lshlrev_b32_e32 v110, 16, v34
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v83
	v_lshlrev_b32_e32 v108, 16, v83
	v_and_b32_e32 v111, 0xffff0000, v35
	v_lshlrev_b32_e32 v110, 16, v35
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x800, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB2_38
; %bb.72:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v76
	v_lshlrev_b32_e32 v106, 16, v76
	v_and_b32_e32 v109, 0xffff0000, v28
	v_lshlrev_b32_e32 v108, 16, v28
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v77
	v_lshlrev_b32_e32 v108, 16, v77
	v_and_b32_e32 v111, 0xffff0000, v29
	v_lshlrev_b32_e32 v110, 16, v29
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v78
	v_lshlrev_b32_e32 v108, 16, v78
	v_and_b32_e32 v111, 0xffff0000, v30
	v_lshlrev_b32_e32 v110, 16, v30
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v79
	v_lshlrev_b32_e32 v108, 16, v79
	v_and_b32_e32 v111, 0xffff0000, v31
	v_lshlrev_b32_e32 v110, 16, v31
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0xa00, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB2_37
; %bb.73:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v72
	v_lshlrev_b32_e32 v106, 16, v72
	v_and_b32_e32 v109, 0xffff0000, v24
	v_lshlrev_b32_e32 v108, 16, v24
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v73
	v_lshlrev_b32_e32 v108, 16, v73
	v_and_b32_e32 v111, 0xffff0000, v25
	v_lshlrev_b32_e32 v110, 16, v25
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v74
	v_lshlrev_b32_e32 v108, 16, v74
	v_and_b32_e32 v111, 0xffff0000, v26
	v_lshlrev_b32_e32 v110, 16, v26
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v75
	v_lshlrev_b32_e32 v108, 16, v75
	v_and_b32_e32 v111, 0xffff0000, v27
	v_lshlrev_b32_e32 v110, 16, v27
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0xc00, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB2_36
; %bb.74:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v68
	v_lshlrev_b32_e32 v106, 16, v68
	v_and_b32_e32 v109, 0xffff0000, v20
	v_lshlrev_b32_e32 v108, 16, v20
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v69
	v_lshlrev_b32_e32 v108, 16, v69
	v_and_b32_e32 v111, 0xffff0000, v21
	v_lshlrev_b32_e32 v110, 16, v21
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v70
	v_lshlrev_b32_e32 v108, 16, v70
	v_and_b32_e32 v111, 0xffff0000, v22
	v_lshlrev_b32_e32 v110, 16, v22
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v71
	v_lshlrev_b32_e32 v108, 16, v71
	v_and_b32_e32 v111, 0xffff0000, v23
	v_lshlrev_b32_e32 v110, 16, v23
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0xe00, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB2_35
; %bb.75:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v64
	v_lshlrev_b32_e32 v106, 16, v64
	v_and_b32_e32 v109, 0xffff0000, v16
	v_lshlrev_b32_e32 v108, 16, v16
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v65
	v_lshlrev_b32_e32 v108, 16, v65
	v_and_b32_e32 v111, 0xffff0000, v17
	v_lshlrev_b32_e32 v110, 16, v17
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v66
	v_lshlrev_b32_e32 v108, 16, v66
	v_and_b32_e32 v111, 0xffff0000, v18
	v_lshlrev_b32_e32 v110, 16, v18
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v67
	v_lshlrev_b32_e32 v108, 16, v67
	v_and_b32_e32 v111, 0xffff0000, v19
	v_lshlrev_b32_e32 v110, 16, v19
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1000, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB2_34
; %bb.76:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v60
	v_lshlrev_b32_e32 v106, 16, v60
	v_and_b32_e32 v109, 0xffff0000, v12
	v_lshlrev_b32_e32 v108, 16, v12
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v61
	v_lshlrev_b32_e32 v108, 16, v61
	v_and_b32_e32 v111, 0xffff0000, v13
	v_lshlrev_b32_e32 v110, 16, v13
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v62
	v_lshlrev_b32_e32 v108, 16, v62
	v_and_b32_e32 v111, 0xffff0000, v14
	v_lshlrev_b32_e32 v110, 16, v14
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v63
	v_lshlrev_b32_e32 v108, 16, v63
	v_and_b32_e32 v111, 0xffff0000, v15
	v_lshlrev_b32_e32 v110, 16, v15
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1200, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB2_33
; %bb.77:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v56
	v_lshlrev_b32_e32 v106, 16, v56
	v_and_b32_e32 v109, 0xffff0000, v8
	v_lshlrev_b32_e32 v108, 16, v8
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v57
	v_lshlrev_b32_e32 v108, 16, v57
	v_and_b32_e32 v111, 0xffff0000, v9
	v_lshlrev_b32_e32 v110, 16, v9
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v58
	v_lshlrev_b32_e32 v108, 16, v58
	v_and_b32_e32 v111, 0xffff0000, v10
	v_lshlrev_b32_e32 v110, 16, v10
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v59
	v_lshlrev_b32_e32 v108, 16, v59
	v_and_b32_e32 v111, 0xffff0000, v11
	v_lshlrev_b32_e32 v110, 16, v11
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1400, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB2_32
; %bb.78:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v107, 0xffff0000, v52
	v_lshlrev_b32_e32 v106, 16, v52
	v_and_b32_e32 v109, 0xffff0000, v4
	v_lshlrev_b32_e32 v108, 16, v4
	v_pk_mul_f32 v[106:107], v[106:107], v[108:109]
	v_and_b32_e32 v109, 0xffff0000, v53
	v_lshlrev_b32_e32 v108, 16, v53
	v_and_b32_e32 v111, 0xffff0000, v5
	v_lshlrev_b32_e32 v110, 16, v5
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v54
	v_lshlrev_b32_e32 v108, 16, v54
	v_and_b32_e32 v111, 0xffff0000, v6
	v_lshlrev_b32_e32 v110, 16, v6
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	v_and_b32_e32 v109, 0xffff0000, v55
	v_lshlrev_b32_e32 v108, 16, v55
	v_and_b32_e32 v111, 0xffff0000, v7
	v_lshlrev_b32_e32 v110, 16, v7
	v_pk_fma_f32 v[106:107], v[108:109], v[110:111], v[106:107]
	s_nop 0
	v_add_f32_e32 v102, v106, v107
	v_add_f32_e32 v99, v99, v102
	v_add_u32_e32 v102, 0x1600, v104
	v_cmp_gt_u32_e32 vcc, s2, v102
	s_and_saveexec_b64 s[36:37], vcc
	s_cbranch_execz .LBB2_31
; %bb.79:                               ;   in Loop: Header=BB2_43 Depth=2
	v_and_b32_e32 v105, 0xffff0000, v48
	v_lshlrev_b32_e32 v104, 16, v48
	v_and_b32_e32 v107, 0xffff0000, v0
	v_lshlrev_b32_e32 v106, 16, v0
	v_pk_mul_f32 v[104:105], v[104:105], v[106:107]
	v_and_b32_e32 v107, 0xffff0000, v49
	v_lshlrev_b32_e32 v106, 16, v49
	v_and_b32_e32 v109, 0xffff0000, v1
	v_lshlrev_b32_e32 v108, 16, v1
	v_pk_fma_f32 v[104:105], v[106:107], v[108:109], v[104:105]
	v_and_b32_e32 v107, 0xffff0000, v50
	v_lshlrev_b32_e32 v106, 16, v50
	v_and_b32_e32 v109, 0xffff0000, v2
	v_lshlrev_b32_e32 v108, 16, v2
	v_pk_fma_f32 v[104:105], v[106:107], v[108:109], v[104:105]
	v_and_b32_e32 v107, 0xffff0000, v51
	v_lshlrev_b32_e32 v106, 16, v51
	v_and_b32_e32 v109, 0xffff0000, v3
	v_lshlrev_b32_e32 v108, 16, v3
	v_pk_fma_f32 v[104:105], v[106:107], v[108:109], v[104:105]
	s_nop 0
	v_add_f32_e32 v102, v104, v105
	v_add_f32_e32 v99, v99, v102
	s_branch .LBB2_31
.LBB2_80:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  10
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 112
		.amdhsa_next_free_sgpr 44
		.amdhsa_accum_offset 112
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,"axG",@progbits,_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii,comdat
.Lfunc_end2:
	.size	_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii, .Lfunc_end2-_Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3684
; NumSgprs: 50
; NumVgprs: 112
; NumAgprs: 0
; TotalNumVgprs: 112
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 50
; NumVGPRsForWavesPerEU: 112
; AccumOffset: 112
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 27
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_40ad3caca8a73c87,@object ; @__hip_cuid_40ad3caca8a73c87
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_40ad3caca8a73c87
__hip_cuid_40ad3caca8a73c87:
	.byte	0                               ; 0x0
	.size	__hip_cuid_40ad3caca8a73c87, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_40ad3caca8a73c87
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20matrixMultiplySharedPfS_S_iiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         _Z20matrixMultiplySharedPfS_S_iiiiii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     40
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I6__halfLi64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     122
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         _Z16wvSplitK_hf_sml_I14__hip_bfloat16Li64ELi1ELi16ELi8ELi12ELi4EEviiPKT_S3_PS1_ii.kd
    .uses_dynamic_stack: false
    .vgpr_count:     112
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
