// host microbench (campaign 3 memo-01 H6): exact int4 column kernel against ggml's Q4_0 4x4 repacked gemv.
// The stock kernel is ggml_gemv_q4_0_4x4_q8_0 (arch/arm/repack.cpp at 620d2225c, NEON + dotprod path) copied verbatim
// with its repack (make_block_q4_0x4, interleave 4). The column kernel keeps the file's Q4_0 blocks (32 inputs, one fp16
// scale per row and block) but stores them input-major: per tile of TR rows and per block, the TR scales, then 32
// columns of TR nibbles. A dropped input skips its column (TR/2 contiguous bytes). Per block the kept columns are summed
// exactly in int16 (|w| <= 8, |x| <= 127, 32 terms < 2^15), then acc = fma(acc, (float)isum, dx*dw) in block order,
// which is the stock kernel's float step, so the result is bit-identical to the stock kernel with the dropped inputs'
// q8 values set to 0. Checked below at density 1.0 and at each density.
// build: clang -O3 -march=armv8.2-a+dotprod+fp16 -o colkern tools/c3_colkern_bench.c
// usage: ./colkern [N=14336] [K=4096] [reps=50]
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define QK 32
typedef struct { __fp16 d; uint8_t qs[16]; } block_q4_0;
typedef struct { __fp16 d; int8_t qs[32]; } block_q8_0;
typedef struct { __fp16 d[4]; uint8_t qs[64]; } block_q4_0x4;
#ifndef TR
#define TR 1024   // rows per tile; -DTR=... to change
#endif

static block_q4_0x4 make_block_q4_0x4(const block_q4_0 * in) {   // ggml, blck_size_interleave == 4
    block_q4_0x4 out;
    for (int i = 0; i < 4; i++) out.d[i] = in[i].d;
    for (int i = 0; i < 16; ++i) {
        uint32_t e; memcpy(&e, &in[i % 4].qs[(i / 4) * 4], 4); e ^= 0x88888888u; memcpy(&out.qs[i * 4], &e, 4);
    }
    return out;
}
static void gemv_stock(int n, float * s, const void * vx, const void * vy, int nc) {   // ggml_gemv_q4_0_4x4_q8_0, verbatim math
    const int nb = n / QK; const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;
    for (int c = 0; c < nc; c += 4) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy; float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs), b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32), b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);
            int8x16_t a0 = vld1q_s8(a_ptr->qs), a1 = vld1q_s8(a_ptr->qs + 16);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);
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
// column layout: [tile][block] { __fp16 d[TR]; uint8_t col[32][TR/2] }, byte i of a column = row i (low) | row i+TR/2 (high)
static size_t col_block_bytes(void) { return TR * 2 + 32 * (TR / 2); }
static void to_columns(const block_q4_0 * w, int N, int K, uint8_t * out) {
    const int nb = K / QK;
    for (int t = 0; t < N / TR; t++) for (int b = 0; b < nb; b++) {
        uint8_t * p = out + ((size_t) t * nb + b) * col_block_bytes(); __fp16 * d = (__fp16 *) p; uint8_t * col = p + TR * 2;
        for (int r = 0; r < TR; r++) d[r] = w[(size_t)(t * TR + r) * nb + b].d;
        for (int j = 0; j < 32; j++) for (int i = 0; i < TR / 2; i++) {
            const block_q4_0 * lo = &w[(size_t)(t * TR + i) * nb + b], * hi = &w[(size_t)(t * TR + i + TR / 2) * nb + b];
            uint8_t ql = j < 16 ? lo->qs[j] & 15 : lo->qs[j - 16] >> 4, qh = j < 16 ? hi->qs[j] & 15 : hi->qs[j - 16] >> 4;
            col[j * (TR / 2) + i] = ql | (qh << 4);
        }
    }
}
// kept[b] = list of kept input indices (0..31) of block b, nk[b] their count
static void gemv_col(int N, int K, float * s, const uint8_t * wc, const block_q8_0 * x, const uint8_t (*kept)[32], const int * nk) {
    const int nb = K / QK; int16_t acc16[TR] __attribute__((aligned(64))); float accf[TR] __attribute__((aligned(64)));
    const int8x16_t eight = vdupq_n_s8(8); const uint8x16_t m4 = vdupq_n_u8(15);
    for (int t = 0; t < N / TR; t++) {
        memset(accf, 0, sizeof accf);
        for (int b = 0; b < nb; b++) {
            const uint8_t * p = wc + ((size_t) t * nb + b) * col_block_bytes(); const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + TR * 2;
            const float dx = (float) x[b].d;
            if (nk[b] == 0) {   // stock: isum = 0, fma(acc, 0, dx*dw) = acc (no inf/nan in finite data)
                continue;
            }
            memset(acc16, 0, sizeof acc16);   // columns outer: one axpy of TR int16 sums per kept input (L1-resident)
            for (int q = 0; q < nk[b]; q++) {
                const int j = kept[b][q]; const int8x8_t xv = vdup_n_s8(x[b].qs[j]); const int8x16_t xv2 = vcombine_s8(xv, xv); const uint8_t * c = col + j * (TR / 2);
                for (int i = 0; i < TR / 2; i += 16) {
                    uint8x16_t v = vld1q_u8(c + i);
                    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), eight), hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
                    int16_t * a = acc16 + i, * h = acc16 + TR / 2 + i;
                    vst1q_s16(a, vmlal_s8(vld1q_s16(a), vget_low_s8(lo), xv)); vst1q_s16(a + 8, vmlal_high_s8(vld1q_s16(a + 8), lo, xv2));
                    vst1q_s16(h, vmlal_s8(vld1q_s16(h), vget_low_s8(hi), xv)); vst1q_s16(h + 8, vmlal_high_s8(vld1q_s16(h + 8), hi, xv2));
                }
            }
            const float32x4_t vdx = vdupq_n_f32(dx);
            for (int r = 0; r < TR; r += 8) {
                int16x8_t a = vld1q_s16(acc16 + r); float16x8_t dw = vld1q_f16(d + r);
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), f1 = vcvtq_f32_s32(vmovl_high_s16(a));
                vst1q_f32(accf + r, vfmaq_f32(vld1q_f32(accf + r), f0, vmulq_f32(vdx, vcvt_f32_f16(vget_low_f16(dw)))));
                vst1q_f32(accf + r + 4, vfmaq_f32(vld1q_f32(accf + r + 4), f1, vmulq_f32(vdx, vcvt_high_f32_f16(dw))));
            }
        }
        memcpy(s + t * TR, accf, sizeof accf);
    }
}
// H6b: the stock 4x4 kernel with a skip list: only the kept input blocks (kb[0..nkb)) are visited. Same float step in
// the same block order, so it equals the stock kernel fed x with the dropped blocks zeroed (isum 0 adds nothing).
static void gemv_stock_skip(int n, float * s, const void * vx, const void * vy, int nc, const int * kb, int nkb) {
    const int nb = n / QK; const block_q4_0x4 * base = (const block_q4_0x4 *) vx; const block_q8_0 * xa = (const block_q8_0 *) vy;
    for (int c = 0; c < nc; c += 4) {
        const block_q4_0x4 * row = base + (size_t)(c / 4) * nb; float32x4_t acc = vdupq_n_f32(0);
        for (int q = 0; q < nkb; q++) {
            const int b = kb[q]; const block_q4_0x4 * b_ptr = row + b; const block_q8_0 * a_ptr = xa + b;
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs), b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32), b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);
            int8x16_t a0 = vld1q_s8(a_ptr->qs), a1 = vld1q_s8(a_ptr->qs + 16);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);
            int32x4_t ret = vdupq_n_s32(0);
            ret = vdotq_laneq_s32(ret, b0 << 4, a0, 0); ret = vdotq_laneq_s32(ret, b1 << 4, a0, 1);
            ret = vdotq_laneq_s32(ret, b2 << 4, a0, 2); ret = vdotq_laneq_s32(ret, b3 << 4, a0, 3);
            ret = vdotq_laneq_s32(ret, b0 & 0xf0U, a1, 0); ret = vdotq_laneq_s32(ret, b1 & 0xf0U, a1, 1);
            ret = vdotq_laneq_s32(ret, b2 & 0xf0U, a1, 2); ret = vdotq_laneq_s32(ret, b3 & 0xf0U, a1, 3);
            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
        }
        vst1q_f32(s, acc); s += 4;
    }
}
// H6b block-major: the same 4x4 groups stored [block][row group], so one dropped input block is N/4 * 72 contiguous bytes.
// Blocks outer (kept only), row groups inner, one float accumulator per row in L1/L2: per row the blocks are still
// added in increasing order with the stock float step, so the result is the stock kernel's to the bit.
static void gemv_blockmajor(int n, float * s, const block_q4_0x4 * wb, const block_q8_0 * xa, int nc, const int * kb, int nkb) {
    const int ng = nc / 4; (void) n;
    for (int g = 0; g < ng; g++) vst1q_f32(s + 4 * g, vdupq_n_f32(0));
    for (int q = 0; q < nkb; q++) {
        const int b = kb[q]; const block_q4_0x4 * col = wb + (size_t) b * ng; const block_q8_0 * a_ptr = xa + b;
        const int8x16_t a0 = vld1q_s8(a_ptr->qs), a1 = vld1q_s8(a_ptr->qs + 16); const float32x4_t ad = vcvt_f32_f16(vld1_dup_f16((const __fp16 *) &a_ptr->d));
        for (int g = 0; g < ng; g++) {
            const block_q4_0x4 * b_ptr = col + g;
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs), b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32), b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            int32x4_t ret = vdupq_n_s32(0);
            ret = vdotq_laneq_s32(ret, b0 << 4, a0, 0); ret = vdotq_laneq_s32(ret, b1 << 4, a0, 1);
            ret = vdotq_laneq_s32(ret, b2 << 4, a0, 2); ret = vdotq_laneq_s32(ret, b3 << 4, a0, 3);
            ret = vdotq_laneq_s32(ret, b0 & 0xf0U, a1, 0); ret = vdotq_laneq_s32(ret, b1 & 0xf0U, a1, 1);
            ret = vdotq_laneq_s32(ret, b2 & 0xf0U, a1, 2); ret = vdotq_laneq_s32(ret, b3 & 0xf0U, a1, 3);
            vst1q_f32(s + 4 * g, vfmaq_f32(vld1q_f32(s + 4 * g), vcvtq_n_f32_s32(ret, 4), vmulq_f32(ad, vcvt_f32_f16(vld1_f16((const __fp16 *) b_ptr->d)))));
        }
    }
}
// distinct 64-byte lines touched by the kept blocks, row-group-major (stock order) or block-major
static double lines_touched(int nb, int ng, const int * kb, int nkb, int blockmajor) {
    const size_t tot = ((size_t) nb * ng * sizeof(block_q4_0x4) + 63) / 64; unsigned char * seen = calloc(tot, 1); size_t cnt = 0;
    for (int g = 0; g < ng; g++) for (int q = 0; q < nkb; q++) {
        const size_t off = (blockmajor ? (size_t) kb[q] * ng + g : (size_t) g * nb + kb[q]) * sizeof(block_q4_0x4);
        for (size_t l = off / 64; l <= (off + sizeof(block_q4_0x4) - 1) / 64; l++) if (!seen[l]) { seen[l] = 1; cnt++; }
    }
    free(seen); return (double) cnt / tot;
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 14336, K = argc > 2 ? atoi(argv[2]) : 4096, reps = argc > 3 ? atoi(argv[3]) : 50, nb = K / QK;
    srand(1);
    block_q4_0 * w = malloc(sizeof(block_q4_0) * (size_t) N * nb);
    for (size_t i = 0; i < (size_t) N * nb; i++) { w[i].d = (__fp16)(0.001f + 0.01f * rand() / RAND_MAX); for (int j = 0; j < 16; j++) w[i].qs[j] = rand(); }
    block_q4_0x4 * w4 = malloc(sizeof(block_q4_0x4) * (size_t) N * nb / 4);
    for (int r = 0; r < N; r += 4) for (int b = 0; b < nb; b++) {   // ggml repack order: rows of 4, then blocks
        block_q4_0 in[4]; for (int k = 0; k < 4; k++) in[k] = w[(size_t)(r + k) * nb + b];
        w4[(size_t)(r / 4) * nb + b] = make_block_q4_0x4(in);
    }
    uint8_t * wc = malloc(col_block_bytes() * (size_t)(N / TR) * nb); to_columns(w, N, K, wc);
    block_q8_0 * x = malloc(sizeof(block_q8_0) * nb), * xz = malloc(sizeof(block_q8_0) * nb);
    for (int b = 0; b < nb; b++) { x[b].d = (__fp16)(0.01f + 0.02f * rand() / RAND_MAX); for (int j = 0; j < 32; j++) x[b].qs[j] = (int8_t)(rand() % 255 - 127); }
    float * ys = malloc(4 * N), * yc = malloc(4 * N); uint8_t (*kept)[32] = malloc(32 * nb); int * nk = malloc(sizeof(int) * nb);
    const double dens[] = { 1.0, 0.8, 0.6, 0.5, 0.4 }; double t_stock = 0;
    printf("N %d K %d TR %d; stock 4x4 bytes %.1f MB, column bytes %.1f MB; single thread, host, relative only\n", N, K, TR,
           sizeof(block_q4_0x4) * (double) N * nb / 4 / 1e6, col_block_bytes() * (double)(N / TR) * nb / 1e6);
    for (int di = 0; di < 5; di++) {
        memcpy(xz, x, sizeof(block_q8_0) * nb);
        for (int b = 0; b < nb; b++) { nk[b] = 0; for (int j = 0; j < 32; j++) if ((double) rand() / RAND_MAX < dens[di] || dens[di] == 1.0) kept[b][nk[b]++] = j; else xz[b].qs[j] = 0; }
        gemv_stock(K, ys, w4, xz, N); gemv_col(N, K, yc, wc, x, kept, nk);
        const int same = memcmp(ys, yc, 4 * N) == 0;
        double t0 = now(); for (int r = 0; r < reps; r++) gemv_stock(K, ys, w4, xz, N); double ts = (now() - t0) / reps;
        t0 = now(); for (int r = 0; r < reps; r++) gemv_col(N, K, yc, wc, x, kept, nk); double tc = (now() - t0) / reps;
        if (di == 0) t_stock = ts;
        int tot = 0; for (int b = 0; b < nb; b++) tot += nk[b];
        const double d = (double) tot / K;
        printf("density %.2f (realised %.3f): bit-identical to stock(zeroed x) %s; stock %.3f ms, column %.3f ms; column/stock(dense) %.3f; cost per touched weight %.2f x stock\n",
               dens[di], d, same ? "YES" : "NO", ts * 1e3, tc * 1e3, tc / t_stock, tc / (d * t_stock));
    }
    // H6b: block-granular skip (whole 32-input blocks), kept fraction 1.0 / 0.7 / 0.6 / 0.5, random blocks
    const double kf[] = { 1.0, 0.9, 0.7, 0.6, 0.5 }; int * kb = malloc(sizeof(int) * nb);
    block_q4_0x4 * wbm = malloc(sizeof(block_q4_0x4) * (size_t) N * nb / 4);   // block-major copy of the same groups
    for (int g = 0; g < N / 4; g++) for (int b = 0; b < nb; b++) wbm[(size_t) b * (N / 4) + g] = w4[(size_t) g * nb + b];
    for (int di = 0; di < 5; di++) {
        memcpy(xz, x, sizeof(block_q8_0) * nb); int nkb = 0;
        for (int b = 0; b < nb; b++) {
            if (kf[di] == 1.0 || (double) rand() / RAND_MAX < kf[di]) kb[nkb++] = b;
            else { xz[b].d = (__fp16) 0.0f; memset(xz[b].qs, 0, 32); }   // a dropped block quantises to d = 0, qs = 0
        }
        gemv_stock(K, ys, w4, xz, N); gemv_stock_skip(K, yc, w4, x, N, kb, nkb);
        const int same = memcmp(ys, yc, 4 * N) == 0;
        double t0 = now(); for (int r = 0; r < reps; r++) gemv_stock_skip(K, yc, w4, x, N, kb, nkb); double tk = (now() - t0) / reps;
        const double d = (double) nkb / nb;
        printf("block-skip kept %.2f (realised %.3f): bit-identical to stock(zeroed blocks) %s; skip-list %.3f ms; /stock(dense) %.3f; cost per touched weight %.2f x stock; lines touched %.3f of dense\n",
               kf[di], d, same ? "YES" : "NO", tk * 1e3, tk / t_stock, tk / (d * t_stock), lines_touched(nb, N / 4, kb, nkb, 0));
        gemv_blockmajor(K, yc, wbm, x, N, kb, nkb); const int same2 = memcmp(ys, yc, 4 * N) == 0;
        t0 = now(); for (int r = 0; r < reps; r++) gemv_blockmajor(K, yc, wbm, x, N, kb, nkb); double tb = (now() - t0) / reps;
        printf("block-major kept %.2f (realised %.3f): bit-identical to stock(zeroed blocks) %s; %.3f ms; /stock(dense) %.3f; cost per touched weight %.2f x stock; lines touched %.3f of dense\n",
               kf[di], d, same2 ? "YES" : "NO", tb * 1e3, tb / t_stock, tb / (d * t_stock), lines_touched(nb, N / 4, kb, nkb, 1));
    }
    return 0;
}
