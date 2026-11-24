// Self-check of patches/llama.cpp-c3/0006-x3-emu.patch (ggml/src/ggml-cpu/vollie-x3emu.h), host only, arm64 with fp16.
//   cc -O2 -march=armv8.2-a+dotprod+fp16 -I<tree>/ggml/src -I<tree>/ggml/src/ggml-cpu -I<tree>/ggml/include tools/c3_x3emu_check.c -lm
// 1. the NEON tile path equals the scalar reference bit for bit (Q4_0 and Q6_K, modes 1 and 2, 1 and 5 columns), and
//    the Q6_K weight unpack equals ggml dequantize_row_q6_K's order (its loop copied below);
// 2. the reference replays R74's x3 identity lines (same generator, shapes and float64 reference as patch 0005) and
//    prints max_rel_err / max_abs beside the device values. Exit 0 only if 1 holds.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "vollie-x3emu.h"
#include <stdio.h>
#include <stdlib.h>

static uint64_t rs = 88172645463325252ull; // patch 0001's generator and seed
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) rs; }
static void fill(uint8_t * p, size_t n) { for (size_t i = 0; i + 4 <= n; i += 4) { uint32_t v = rnd(); memcpy(p + i, &v, 4); } }
static float * rand_f(size_t n) {
    float * f = malloc(n * 4);
    for (size_t i = 0; i < n; i++) { float s = 0; for (int k = 0; k < 4; k++) s += (rnd() & 0xffff) / 65535.f; f[i] = (s - 2.f) * 1.732f; }
    return f;
}
static block_q4_0 * make_w(int M, int K) { // patch 0005 make_hmat: scale first (left operand first), then 16 bytes
    block_q4_0 * w = malloc(sizeof(block_q4_0) * (size_t) M * (K / 32));
    for (size_t i = 0; i < (size_t) M * (K / 32); i++) {
        const float a = (0.002f + 0.02f * (rnd() & 1023) / 1023.f); const float s = (rnd() & 1) ? 1.f : -1.f;
        __fp16 h = (__fp16)(a * s); memcpy(&w[i].d, &h, 2); fill(w[i].qs, 16);
    }
    return w;
}

int main(void) {
    int fails = 0;
    // 1. tile == reference, 37 rows (two tiles + 5 tail rows done by the reference), K = 4096 + 96 (a KC tail)
    { const int M = 32, K = 4192, nb = K / 32, NC = 5;
      block_q4_0 * w = make_w(M, K); float * x = rand_f((size_t) K * NC);
      w[3].d = 0x0001; w[7].d = 0x03ff; // subnormal scales
      float buf[16 * 32 * VOLLIE_X3EMU_KC]; float32x4_t acc[16 * NC]; float out[16 * NC];
      for (int mode = 1; mode <= 2; mode++) for (int nc = 1; nc <= NC; nc += NC - 1) for (int t = 0; t < M; t += 16) {
          vollie_x3emu_tile16((const char *)(w + (size_t) t * nb), sizeof(block_q4_0) * nb, nb, x, K, nc, out, 16, mode, buf, acc, 0);
          for (int j = 0; j < nc; j++) for (int r = 0; r < 16; r++) {
              const float ref = vollie_x3emu_row_ref(w + (size_t)(t + r) * nb, x + (size_t) j * K, nb, mode);
              if (memcmp(&ref, &out[j * 16 + r], 4)) { fails++; if (fails < 5) printf("mismatch mode %d col %d row %d: %a vs %a\n", mode, j, t + r, ref, out[j * 16 + r]); }
          }
      }
      printf("tile_vs_ref mismatches=%d (modes 1,2; cols 1,5; 32 rows; K=%d)\n", fails, K);
      free(w); free(x); rs = 88172645463325252ull; }
    // 1b. Q6_K: unpack order against ggml's dequantize_row_q6_K loop, then tile == reference
    { const int M = 32, K = 4096, nb = K / 32, NS = K / 256, NC = 5; int bad = 0;
      block_q6_K * w = malloc(sizeof(block_q6_K) * M * NS); fill((uint8_t *) w, sizeof(block_q6_K) * M * NS);
      for (int i = 0; i < M * NS; i++) { __fp16 h = (__fp16)(0.001f + 0.01f * (rnd() & 1023) / 1023.f); memcpy(&w[i].d, &h, 2); }
      for (int sb = 0; sb < NS; sb++) { // ggml-quants.c dequantize_row_q6_K, one super-block of row 0
          const block_q6_K * B = w + sb; const uint8_t * ql = B->ql, * qh = B->qh; const int8_t * sc = B->scales; int q[256]; int s[256];
          for (int n = 0; n < 256; n += 128) { for (int l = 0; l < 32; ++l) { int is = l / 16;
              q[n + l] = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32; s[n + l] = sc[is + 0];
              q[n + l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32; s[n + l + 32] = sc[is + 2];
              q[n + l + 64] = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32; s[n + l + 64] = sc[is + 4];
              q[n + l + 96] = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32; s[n + l + 96] = sc[is + 6]; }
            ql += 64; qh += 32; sc += 8; }
          uint16_t dh; memcpy(&dh, &B->d, 2);
          for (int e = 0; e < 256; e++) { const float want = (float)(q[e] * s[e]) * vollie_x3emu_h2f(dh);
              const float got = vollie_x3emu_q6k_w(w, sb * 8 + e / 32, e % 32); if (memcmp(&want, &got, 4)) bad++; }
      }
      float * x = rand_f((size_t) K * NC); float buf[16 * 32 * VOLLIE_X3EMU_KC]; float32x4_t acc[16 * NC]; float out[16 * NC];
      for (int mode = 1; mode <= 2; mode++) for (int nc = 1; nc <= NC; nc += NC - 1) for (int t = 0; t < M; t += 16) {
          vollie_x3emu_tile16((const char *)(w + (size_t) t * NS), sizeof(block_q6_K) * NS, nb, x, K, nc, out, 16, mode, buf, acc, 1);
          for (int j = 0; j < nc; j++) for (int r = 0; r < 16; r++) {
              const float ref = vollie_x3emu_row_ref_t(w + (size_t)(t + r) * NS, x + (size_t) j * K, nb, mode, 1);
              if (memcmp(&ref, &out[j * 16 + r], 4)) bad++; }
      }
      printf("q6k unpack_vs_ggml_and_tile_vs_ref mismatches=%d\n", bad); fails += bad;
      free(w); free(x); rs = 88172645463325252ull; }
    // 2. replay of x3b: x4, x11, then six identity matrices in order
    float * x4 = rand_f(4096); float * x11 = rand_f(11008);
    const struct { int M, K; const char * id; double dev_rel, dev_abs; } c[6] = {
        { 512, 4096, "general", 6.336e-04, 2.367e-03 }, { 2816, 4096, "general", 6.687e-04, 2.593e-03 },
        { 11008, 4096, "general", 7.634e-04, 3.005e-03 }, { 4096, 4096, "s4096_1_4096", 6.910e-04, 2.718e-03 },
        { 4096, 11008, "s4096_1_11008", 6.233e-04, 4.010e-03 }, { 11008, 4096, "s11008_1_4096", 6.882e-04, 2.713e-03 } };
    for (int s = 0; s < 6; s++) {
        const int M = c[s].M, K = c[s].K, nb = K / 32; const float * x = K == 4096 ? x4 : x11;
        block_q4_0 * w = make_w(M, K);
        double * ref = malloc(sizeof(double) * M), rms = 0;
        for (int i = 0; i < M; i++) {
            double acc = 0;
            for (int b = 0; b < nb; b++) { const block_q4_0 * k = w + (size_t) i * nb + b; double t = 0;
                for (int e = 0; e < 16; e++) t += ((k->qs[e] & 15) - 8) * (double) x[b * 32 + e] + ((k->qs[e] >> 4) - 8) * (double) x[b * 32 + e + 16];
                uint16_t dh; memcpy(&dh, &k->d, 2); acc += (double) vollie_x3emu_h2f(dh) * t; }
            ref[i] = acc; rms += acc * acc;
        }
        rms = sqrt(rms / M);
        for (int mode = 1; mode <= 2; mode++) {
            double mr = 0, ma = 0;
            for (int i = 0; i < M; i++) { const double a = fabs(vollie_x3emu_row_ref(w + (size_t) i * nb, x, nb, mode) - ref[i]); if (a > ma) ma = a; if (a / rms > mr) mr = a / rms; }
            printf("identity kern=%s m=%d k=%d mode=%d emu max_rel_err=%.3e max_abs=%.3e rms_ref=%.3e | device (R74) %.3e %.3e\n",
                   c[s].id, M, K, mode, mr, ma, rms, c[s].dev_rel, c[s].dev_abs);
        }
        free(w); free(ref);
    }
    printf(fails ? "FAIL\n" : "ok\n");
    return fails != 0;
}
