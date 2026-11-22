// Ebony, memo-10 review scratch: W-vector Q4_0 4x4 gemv, weights unpacked once per block, stock per-column op order.
// host: identity vs stock gemv per position (memcmp) + hot ns per weight by W. android -S: A78 pipe count.
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { __fp16 d[4]; int8_t qs[64]; } bq4x4;
typedef struct { __fp16 d; int8_t qs[32]; } bq8;
__attribute__((noinline)) void stock(int nb, float *s, const bq4x4 *b_ptr, const bq8 *x, int nc) {
    for (int c = 0; c < nc; c += 4) {
        const bq8 *a_ptr = x; float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8(b_ptr->qs), b1 = vld1q_s8(b_ptr->qs + 16), b2 = vld1q_s8(b_ptr->qs + 32), b3 = vld1q_s8(b_ptr->qs + 48);
            float16x4_t bd = vld1_f16(b_ptr->d);
            int8x16_t a0 = vld1q_s8(a_ptr->qs), a1 = vld1q_s8(a_ptr->qs + 16);
            float16x4_t ad = vld1_dup_f16(&a_ptr->d);
            int32x4_t ret = vdupq_n_s32(0);
            ret = vdotq_laneq_s32(ret, b0 << 4, a0, 0); ret = vdotq_laneq_s32(ret, b1 << 4, a0, 1);
            ret = vdotq_laneq_s32(ret, b2 << 4, a0, 2); ret = vdotq_laneq_s32(ret, b3 << 4, a0, 3);
            ret = vdotq_laneq_s32(ret, b0 & 0xf0U, a1, 0); ret = vdotq_laneq_s32(ret, b1 & 0xf0U, a1, 1);
            ret = vdotq_laneq_s32(ret, b2 & 0xf0U, a1, 2); ret = vdotq_laneq_s32(ret, b3 & 0xf0U, a1, 3);
            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++; b_ptr++;
        }
        vst1q_f32(s, acc); s += 4;
    }
}
// x: W position vectors, each nb blocks; adf: their scales as f32 (converted once per call, exact); s: W outputs of nc
#define GW(W) __attribute__((noinline)) void gw##W(int nb, float *s, const bq4x4 *b_ptr, const bq8 *x, const float *adf, int nc) { \
    const int8x16_t m = vdupq_n_s8((int8_t)0xf0); \
    for (int c = 0; c < nc; c += 4) { \
        float32x4_t acc[W]; for (int p = 0; p < W; p++) acc[p] = vdupq_n_f32(0); \
        for (int b = 0; b < nb; b++) { \
            int8x16_t b0 = vld1q_s8(b_ptr->qs), b1 = vld1q_s8(b_ptr->qs + 16), b2 = vld1q_s8(b_ptr->qs + 32), b3 = vld1q_s8(b_ptr->qs + 48); \
            const int8x16_t h0 = vshlq_n_s8(b0, 4), h1 = vshlq_n_s8(b1, 4), h2 = vshlq_n_s8(b2, 4), h3 = vshlq_n_s8(b3, 4); \
            b0 = vandq_s8(b0, m); b1 = vandq_s8(b1, m); b2 = vandq_s8(b2, m); b3 = vandq_s8(b3, m); \
            const float32x4_t bdf = vcvt_f32_f16(vld1_f16(b_ptr->d)); \
            _Pragma("clang loop unroll(full)") for (int p = 0; p < W; p++) { \
                const bq8 *a = x + p * nb + b; int8x16_t a0 = vld1q_s8(a->qs), a1 = vld1q_s8(a->qs + 16); \
                int32x4_t ret = vdupq_n_s32(0); \
                ret = vdotq_laneq_s32(ret, h0, a0, 0); ret = vdotq_laneq_s32(ret, h1, a0, 1); \
                ret = vdotq_laneq_s32(ret, h2, a0, 2); ret = vdotq_laneq_s32(ret, h3, a0, 3); \
                ret = vdotq_laneq_s32(ret, b0, a1, 0); ret = vdotq_laneq_s32(ret, b1, a1, 1); \
                ret = vdotq_laneq_s32(ret, b2, a1, 2); ret = vdotq_laneq_s32(ret, b3, a1, 3); \
                acc[p] = vfmaq_f32(acc[p], vcvtq_n_f32_s32(ret, 4), vmulq_f32(vld1q_dup_f32(adf + p * nb + b), bdf)); } \
            b_ptr++; } \
        for (int p = 0; p < W; p++) vst1q_f32(s + p * nc + c, acc[p]); } }
GW(1) GW(2) GW(3) GW(4) GW(6) GW(8)
#ifndef ASM_ONLY
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(void) {
    enum { N = 4096, NB = N / 32, NC = 256, WM = 8 };   // 256 rows x 4096: 0.6 MB of weights, hot in L2
    bq4x4 *w = malloc(sizeof(bq4x4) * NB * NC / 4); bq8 *x = malloc(sizeof(bq8) * NB * WM);
    float *adf = malloc(4 * NB * WM), *y = malloc(4 * NC * WM), *r = malloc(4 * NC * WM);
    srand(1); for (size_t i = 0; i < sizeof(bq4x4) * NB * NC / 4; i++) ((uint8_t *)w)[i] = rand();
    for (int i = 0; i < NB * NC / 4; i++) for (int k = 0; k < 4; k++) w[i].d[k] = (rand() % 2000 - 1000) / 7919.0f;
    for (int i = 0; i < NB * WM; i++) { for (int k = 0; k < 32; k++) x[i].qs[k] = rand() % 255 - 127; x[i].d = (rand() % 1000 + 1) / 3571.0f; adf[i] = (float)x[i].d; }
    for (int p = 0; p < WM; p++) stock(NB, r + p * NC, w, x + p * NB, NC);
    void (*f[])(int, float *, const bq4x4 *, const bq8 *, const float *, int) = { gw1, gw2, gw3, gw4, gw6, gw8 }; int Ws[] = { 1, 2, 3, 4, 6, 8 };
    double t0 = now(); int it = 4000; for (int i = 0; i < it; i++) stock(NB, y, w, x, NC);
    double ts = (now() - t0) / it / (N * NC) * 1e9; printf("stock %.4f ns/weight\n", ts);
    for (int k = 0; k < 6; k++) { int W = Ws[k]; double best = 1e9;
        for (int rep = 0; rep < 5; rep++) { t0 = now(); for (int i = 0; i < it; i++) f[k](NB, y, w, x, adf, NC); double t = (now() - t0) / it / (N * NC) * 1e9; if (t < best) best = t; }
        int same = 1; for (int p = 0; p < W; p++) same &= !memcmp(y + p * NC, r + p * NC, 4 * NC);
        printf("W=%d identity=%d  %.4f ns/weight  = %.2f x stock  marginal %.2f\n", W, same, best, best / ts, W > 1 ? (best / ts - 1) / (W - 1) : 0); }
    return 0; }
#endif
