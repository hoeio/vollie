// memo-12 round 9 (Bonnie): time KleidiAI's scalar Q4_0 weight pack (the routine ggml's KleidiAI buffer type calls at
// every load) on the host. *host*, one thread, relative only. Build (from the repo root):
//   K=lab/llama.cpp-c1h6/build-kai/_deps/kleidiai-src
//   cc -O3 -I$K bonnie_pack_time.c $K/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c -o pack_time
// Shapes: 11008 x 4096 and 4096 x 11008 (n rows, k inputs), Q4_0 source (18 bytes per 32 weights), nr 4, kr 8, sr 2, bl 32
// (the dotprod gemm kernel's values, kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c lines 40 to 42).
// "cold" walks a ring of sources larger than the caches, as a load does; the check compares one packed block with a
// direct re-derivation (scale copied, nibble bytes XOR 0x88).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    const size_t shapes[2][2] = { { 11008, 4096 }, { 4096, 11008 } }; const size_t nr = 4, kr = 8, sr = 2, bl = 32;
    struct kai_rhs_pack_qs4cxs1s0_param p = { 1, 8 };
    for (int s = 0; s < 2; s++) {
        const size_t n = shapes[s][0], k = shapes[s][1], src_b = n * (k / 32) * 18;
        const size_t dst_b = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);
        const int copies = 24;                                   // 24 x 25 MB = 0.6 GB of source
        unsigned char * src = malloc(src_b * copies), * dst = malloc(dst_b * copies);
        if (!src || !dst) return 1;
        unsigned x = 12345; for (size_t i = 0; i < src_b * copies; i++) { x = x * 1664525u + 1013904223u; src[i] = x >> 24; }
        memset(dst, 0, dst_b * copies);                          // fault the destination in before timing
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            const double t0 = now();
            for (int c = 0; c < copies; c++)
                kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(1, n, k, nr, kr, sr, bl, src + c * src_b, NULL, dst + c * dst_b, 0, &p);
            const double t = now() - t0; if (t < best) best = t;
        }
        // check block 0 of row group 0: 4 half scales, then 16 bytes per row interleaved 4 bytes at a time
        int ok = dst_b == src_b; const size_t rs = (k / 32) * 18;
        for (int r = 0; r < 4 && ok; r++) {
            ok = !memcmp(dst + 2 * r, src + r * rs, 2);
            for (int seg = 0; seg < 4 && ok; seg++) for (int b = 0; b < 4 && ok; b++)
                ok = dst[8 + seg * 16 + r * 4 + b] == (unsigned char)(src[r * rs + 2 + seg * 4 + b] ^ 0x88);
        }
        printf("pack n=%zu k=%zu src_mb=%.1f dst_mb=%.1f copies=%d best_s=%.3f gbs=%.2f check=%d\n", n, k, src_b / 1048576.0, dst_b / 1048576.0, copies,
               best, src_b * copies / best / 1e9, ok);
        free(src); free(dst);
    }
    return 0;
}
