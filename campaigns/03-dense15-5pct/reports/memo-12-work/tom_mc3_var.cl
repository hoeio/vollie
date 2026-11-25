// Insert before "#undef MC_COL_Q40" in upstream gemv_noshuffle_q4_0_f32.cl (620d2225c) to get the variant used by tom_mc3t.cpp.
// probe-only variant (not upstream): mc3 at a compile-time 2 columns, constant K stride (N_SIMDGROUP), local memory sized
// for 4 subgroups of float4 (4 KB, against mc3's 16 KB float8 x 8 subgroups). Same arithmetic order per column as mc3.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void var_mc_fixed2(
        __read_only  image1d_buffer_t src0_q, global half2 * src0_d, __read_only image1d_buffer_t src1,
        global float * dst, ulong offsetd, int ne00, int ne01, int n_cols)
{
    uint groupId = get_local_id(1);
    uint gid     = get_global_id(0);
    ushort slid  = get_sub_group_local_id();
    uint K = ne00, M = ne01;
    uint LINE_STRIDE_A = M / 2;
    uint gid_s = min(gid, LINE_STRIDE_A - 1);
    uint BLOCK_STRIDE_A = N_SIMDGROUP * M;
    uint COL_STRIDE = K / 4;
    __private uint4  regA_hi, regA_lo;
    __private half2  regS;
    __private float8 regB;
    __private float2 ts0 = (float2)(0.0f), ts1 = (float2)(0.0f);
    for (uint k = groupId; k < (K / QK4_0); k += N_SIMDGROUP) {
        regS = src0_d[gid_s + k * LINE_STRIDE_A];
        regA_hi.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
        regA_hi.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
        regA_hi.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
        regA_hi.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
        regA_lo.s0 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4)).x;
        regA_lo.s1 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5)).x;
        regA_lo.s2 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6)).x;
        regA_lo.s3 = read_imageui(src0_q, (gid_s + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7)).x;
        MC_COL_Q40(ts0, 0);
        MC_COL_Q40(ts1, 1);
    }
    __local float4 reduceLM4[SIMDGROUP_WIDTH * 4];
    float4 acc = (float4)(ts0.s0, ts0.s1, ts1.s0, ts1.s1);
    reduceLM4[groupId * SIMDGROUP_WIDTH + slid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) {
        for (uint g = 1; g < N_SIMDGROUP; g++) acc += reduceLM4[g * SIMDGROUP_WIDTH + slid];
        dst = (global float*)((global char*)dst + offsetd);
        if (gid < LINE_STRIDE_A) {
            vstore2((float2)(acc.s0, acc.s1), 0, &(dst[0 * M + gid * 2]));
            vstore2((float2)(acc.s2, acc.s3), 0, &(dst[1 * M + gid * 2]));
        }
    }
}
