// host microbench (campaign 3 memo-01 H6): exact int4 column kernel against ggml's Q4_0 4x4 repacked gemv.
// The stock kernel is ggml_gemv_q4_0_4x4_q8_0 (arch/arm/repack.cpp at 620d2225c, NEON + dotprod path) copied verbatim
// with its repack (make_block_q4_0x4, interleave 4). The column kernel keeps the file's Q4_0 blocks (32 inputs, one fp16
// scale per row and block) but stores them input-major: per tile of TR rows and per block, the TR scales, then 32
// columns of TR nibbles. A dropped input skips its column (TR/2 contiguous bytes). Per block the kept columns are summed
// exactly in int16 (|w| <= 8, |x| <= 127, 32 terms < 2^15), then acc = fma(acc, (float)isum, dx*dw) in block order,
// which is the stock kernel's float step, so the result is bit-identical to the stock kernel with the dropped inputs'
// q8 values set to 0. Checked below at density 1.0 and at each density.
// build: clang -O3 -march=armv8.2-a+dotprod+fp16 -o colkern tools/c3_colkern_bench.c
// usage: ./colkern [N=14336] [K=4096] [reps=50] [k1|k1lines]   (k1: memo-05 T1/T2/T4, see c3_k1_bench.h)
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
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
// ---- memo-05 (c3 joint round 3): T1 K1, T2 Y1b, T4 gemm vs gemv, T7 lines touched. Mode "k1" / "k1lines". ----
// K1 (strip-register axpy, bias-form nibbles). Layout, tr rows per tile (128 | tr): [tile][block] { fp16 d[tr];
// uint8 col[32][tr/2] }. Column j of strip st (128 rows) is the 64 B line col + j*tr/2 + st*64; byte k of it holds row
// st*128 + (k/16)*32 + k%16 (low nibble) and that row + 16 (high nibble), as the file's unsigned u (no XOR). Per strip
// the 128 int16 sums live in 16 registers; sum (u-8)x = sum ux - 8 sum x, with sum x over the block's kept inputs. The
// int16 sums wrap (15*127*32 > 2^15) but the true sum fits (8*127*32 < 2^15), so modular arithmetic makes it exact.
// Float step as stock: acc = fma(acc, (float) isum, dx*dw), blocks in increasing order.
static size_t k1_block_bytes(int tr) { return (size_t) tr * 18; }
static void to_k1(const block_q4_0 * w, int N, int K, int tr, uint8_t * out) {
    const int nb = K / QK;
    for (int t = 0; t < N / tr; t++) for (int b = 0; b < nb; b++) {
        uint8_t * p = out + ((size_t) t * nb + b) * k1_block_bytes(tr); __fp16 * d = (__fp16 *) p; uint8_t * col = p + tr * 2;
        for (int r = 0; r < tr; r++) d[r] = w[(size_t)(t * tr + r) * nb + b].d;
        for (int j = 0; j < 32; j++) for (int st = 0; st < tr / 128; st++) for (int k = 0; k < 64; k++) {
            const int r0 = t * tr + st * 128 + (k / 16) * 32 + k % 16;
            const block_q4_0 * lo = &w[(size_t) r0 * nb + b], * hi = &w[(size_t)(r0 + 16) * nb + b];
            const uint8_t ql = j < 16 ? lo->qs[j] & 15 : lo->qs[j - 16] >> 4, qh = j < 16 ? hi->qs[j] & 15 : hi->qs[j - 16] >> 4;
            col[j * (tr / 2) + st * 64 + k] = ql | (qh << 4);
        }
    }
}
static void gemv_k1(int N, int K, int tr, int pf, float * s, const uint8_t * wc, const block_q8_0 * x, const uint8_t (*kept)[32], const int * nk) {
    const int nb = K / QK; float accf[1024] __attribute__((aligned(64))); const uint8x16_t m4 = vdupq_n_u8(15);
    for (int t = 0; t < N / tr; t++) {
        memset(accf, 0, sizeof(float) * tr);
        for (int b = 0; b < nb; b++) {
            if (nk[b] == 0) continue;   // stock: isum = 0 adds nothing
            const uint8_t * p = wc + ((size_t) t * nb + b) * k1_block_bytes(tr); const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + tr * 2;
            if (pf && b + 1 < nb) {   // pf: prefetch the next block's scales and kept columns (variant, not in Bonnie's spec)
                const uint8_t * pn = p + k1_block_bytes(tr);
                for (int l = 0; l < tr * 2; l += 64) __builtin_prefetch(pn + l);
                for (int q = 0; q < nk[b + 1]; q++) for (int l = 0; l < tr / 2; l += 64) __builtin_prefetch(pn + tr * 2 + kept[b + 1][q] * (tr / 2) + l);
            }
            int sx = 0; for (int q = 0; q < nk[b]; q++) sx += x[b].qs[kept[b][q]];
            const int16x8_t corr = vdupq_n_s16((int16_t)(8 * sx)); const float32x4_t vdx = vdupq_n_f32((float) x[b].d);
            for (int st = 0; st < tr / 128; st++) {
                int16x8_t a[16];
#pragma clang loop unroll(full)
                for (int i = 0; i < 16; i++) a[i] = vdupq_n_s16(0);
                for (int q = 0; q < nk[b]; q++) {
                    const int j = kept[b][q]; const int8x16_t xv = vdupq_n_s8(x[b].qs[j]); const int8x8_t xh = vget_low_s8(xv);
                    const uint8_t * c = col + j * (tr / 2) + st * 64;
#pragma clang loop unroll(full)
                    for (int L = 0; L < 4; L++) {
                        const uint8x16_t v = vld1q_u8(c + 16 * L);
                        const int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(v, m4)), hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
                        a[4 * L + 0] = vmlal_s8(a[4 * L + 0], vget_low_s8(lo), xh); a[4 * L + 1] = vmlal_high_s8(a[4 * L + 1], lo, xv);
                        a[4 * L + 2] = vmlal_s8(a[4 * L + 2], vget_low_s8(hi), xh); a[4 * L + 3] = vmlal_high_s8(a[4 * L + 3], hi, xv);
                    }
                }
                float * af = accf + st * 128; const __fp16 * ds = d + st * 128;   // register i = rows st*128 + 8i .. +7
#pragma clang loop unroll(full)
                for (int i = 0; i < 16; i++) {
                    const int16x8_t v = vsubq_s16(a[i], corr); const float16x8_t dw = vld1q_f16(ds + 8 * i);
                    vst1q_f32(af + 8 * i, vfmaq_f32(vld1q_f32(af + 8 * i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))), vmulq_f32(vdx, vcvt_f32_f16(vget_low_f16(dw)))));
                    vst1q_f32(af + 8 * i + 4, vfmaq_f32(vld1q_f32(af + 8 * i + 4), vcvtq_f32_s32(vmovl_high_s16(v)), vmulq_f32(vdx, vcvt_high_f32_f16(dw))));
                }
            }
        }
        memcpy(s + (size_t) t * tr, accf, sizeof(float) * tr);
    }
}
// Y1b: in-place 4-row gemv on the file's own Q4_0 rows (18 B blocks, row-major). Per row: XOR 0x88 (bias to sign form),
// shl 4 / and 0xF0 as stock, two sdot (non-lane) against x[0..15] / x[16..31], then 3 addp give the 4 rows' 16*isum.
// Integer sums are exact, the float step is stock's, so the result equals the 4x4 gemv.
static inline float32x4_t y1b_group(int nb, const block_q4_0 * w0, const block_q4_0 * w1, const block_q4_0 * w2, const block_q4_0 * w3, const block_q8_0 * x) {
    const int8x16_t m88 = vdupq_n_s8((int8_t) 0x88), mf0 = vdupq_n_s8((int8_t) 0xF0); const int32x4_t z = vdupq_n_s32(0); float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++) {
        const int8x16_t a0 = vld1q_s8(x[b].qs), a1 = vld1q_s8(x[b].qs + 16);
        const int8x16_t v0 = veorq_s8(vld1q_s8((const int8_t *) w0[b].qs), m88), v1 = veorq_s8(vld1q_s8((const int8_t *) w1[b].qs), m88);
        const int8x16_t v2 = veorq_s8(vld1q_s8((const int8_t *) w2[b].qs), m88), v3 = veorq_s8(vld1q_s8((const int8_t *) w3[b].qs), m88);
        const int32x4_t d0 = vdotq_s32(vdotq_s32(z, vshlq_n_s8(v0, 4), a0), vandq_s8(v0, mf0), a1);
        const int32x4_t d1 = vdotq_s32(vdotq_s32(z, vshlq_n_s8(v1, 4), a0), vandq_s8(v1, mf0), a1);
        const int32x4_t d2 = vdotq_s32(vdotq_s32(z, vshlq_n_s8(v2, 4), a0), vandq_s8(v2, mf0), a1);
        const int32x4_t d3 = vdotq_s32(vdotq_s32(z, vshlq_n_s8(v3, 4), a0), vandq_s8(v3, mf0), a1);
        const int32x4_t ret = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        float16x4_t bd = vld1_dup_f16(&w0[b].d); bd = vld1_lane_f16(&w1[b].d, bd, 1); bd = vld1_lane_f16(&w2[b].d, bd, 2); bd = vld1_lane_f16(&w3[b].d, bd, 3);
        acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4), vmulq_f32(vcvt_f32_f16(vld1_dup_f16((const __fp16 *) &x[b].d)), vcvt_f32_f16(bd)));
    }
    return acc;
}
static void gemv_y1b(int K, float * s, const block_q4_0 * w, const block_q8_0 * x, int N) {
    const int nb = K / QK;
    for (int r = 0; r < N; r += 4) { const block_q4_0 * w0 = w + (size_t) r * nb; vst1q_f32(s + r, y1b_group(nb, w0, w0 + nb, w0 + 2 * nb, w0 + 3 * nb, x)); }
}
// E6 (Ebony): the same kernel on four arbitrary rows (a kept-row list, n % 4 == 0); s[q] = row rows[q]
static void gemv_y1b_rows(int K, float * s, const block_q4_0 * w, const block_q8_0 * x, const int * rows, int n) {
    const int nb = K / QK;
    for (int q = 0; q < n; q += 4)
        vst1q_f32(s + q, y1b_group(nb, w + (size_t) rows[q] * nb, w + (size_t) rows[q + 1] * nb, w + (size_t) rows[q + 2] * nb, w + (size_t) rows[q + 3] * nb, x));
}
#ifdef GGML_LINK   // the lab tree's compiled ggml-cpu (armv8.2-a+dotprod+fp16), linked for the identity checks and T4
void ggml_cpu_init(void);
void ggml_gemv_q4_0_4x4_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_quantize_mat_q8_0_4x4(const float * x, void * vy, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_q4_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
#endif
enum { KS_STOCK, KS_COL, KS_K1_128, KS_K1_512, KS_Y1B, KS_LIBGEMV, KS_VECDOT, KS_Y1BROWS, KS_K1_128PF, KS_K1_512PF, KS_N };
static const char * ks_name[KS_N] = { "stock4x4", "h6col", "K1-tr128", "K1-tr512", "Y1b", "lib-gemv", "Y1c-vecdot", "Y1b-rows", "K1-tr128-pf", "K1-tr512-pf" };
typedef struct { int K; const block_q8_0 * x, * xz; const uint8_t (*kept)[32]; const int * nk; const int * rows; int nrows; } kargs;
static void kcall(int ks, const void * buf, int n, float * y, const kargs * a) {
    switch (ks) {
    case KS_STOCK: gemv_stock(a->K, y, buf, a->xz, n); break;
    case KS_COL: gemv_col(n, a->K, y, buf, a->x, a->kept, a->nk); break;
    case KS_K1_128: gemv_k1(n, a->K, 128, 0, y, buf, a->x, a->kept, a->nk); break;
    case KS_K1_512: gemv_k1(n, a->K, 512, 0, y, buf, a->x, a->kept, a->nk); break;
    case KS_K1_128PF: gemv_k1(n, a->K, 128, 1, y, buf, a->x, a->kept, a->nk); break;
    case KS_K1_512PF: gemv_k1(n, a->K, 512, 1, y, buf, a->x, a->kept, a->nk); break;
    case KS_Y1B: gemv_y1b(a->K, y, buf, a->x, n); break;
    case KS_Y1BROWS: gemv_y1b_rows(a->K, y, buf, a->x, a->rows, a->nrows); break;
#ifdef GGML_LINK
    case KS_LIBGEMV: ggml_gemv_q4_0_4x4_q8_0(a->K, y, 0, buf, a->x, 1, n); break;
    case KS_VECDOT: { const size_t rb = (size_t)(a->K / QK) * sizeof(block_q4_0);
        for (int r = 0; r < n; r++) ggml_vec_dot_q4_0_q8_0(a->K, y + r, 0, (const char *) buf + r * rb, 0, a->x, 0, 1); } break;
#endif
    }
}
// seconds per call, best of 3 trials of `reps` calls. cold: round-robin over `ncopy` copies (stride bytes apart).
static double ktime(int ks, const uint8_t * buf, size_t stride, int ncopy, int n, int reps, float * y, const kargs * a) {
    double best = 1e9;
    for (int tr = 0; tr < 3; tr++) {
        const double t0 = now(); for (int r = 0; r < reps; r++) kcall(ks, buf + (size_t)(r % ncopy) * stride, n, y, a);
        const double t = (now() - t0) / reps; if (t < best) best = t;
    }
    return best;
}
static int k1_main(int N, int K, int reps) {
    const int nb = K / QK; srand(1);
    if (N % 512) { fprintf(stderr, "N must be a multiple of 512\n"); return 1; }
#ifdef GGML_LINK
    ggml_cpu_init();
#endif
    block_q4_0 * w = malloc(sizeof(block_q4_0) * (size_t) N * nb);
    for (size_t i = 0; i < (size_t) N * nb; i++) { w[i].d = (__fp16)(0.001f + 0.01f * rand() / RAND_MAX); for (int j = 0; j < 16; j++) w[i].qs[j] = rand(); }
    block_q4_0x4 * w4 = malloc(sizeof(block_q4_0x4) * (size_t) N * nb / 4);
    for (int r = 0; r < N; r += 4) for (int b = 0; b < nb; b++) { block_q4_0 in[4]; for (int k = 0; k < 4; k++) in[k] = w[(size_t)(r + k) * nb + b]; w4[(size_t)(r / 4) * nb + b] = make_block_q4_0x4(in); }
    const size_t bytes = sizeof(block_q4_0) * (size_t) N * nb;   // every layout is 0.5625 B per weight
    uint8_t * wc = malloc(col_block_bytes() * (size_t)(N / TR) * nb); to_columns(w, N, K, wc);
    uint8_t * k128 = malloc(bytes), * k512 = malloc(bytes); to_k1(w, N, K, 128, k128); to_k1(w, N, K, 512, k512);
    const void * lay[KS_N] = { w4, wc, k128, k512, w, w4, w, w, k128, k512 };
    block_q8_0 * x = malloc(sizeof(block_q8_0) * nb), * xz = malloc(sizeof(block_q8_0) * nb);
    for (int b = 0; b < nb; b++) { x[b].d = (__fp16)(0.01f + 0.02f * rand() / RAND_MAX); for (int j = 0; j < 32; j++) x[b].qs[j] = (int8_t)(rand() % 255 - 127); }
    float * ys = malloc(4 * (size_t) N), * yk = malloc(4 * (size_t) N); uint8_t (*kept)[32] = malloc(32 * (size_t) nb); int * nk = malloc(sizeof(int) * nb);
    const int nhot = 1024, ncold = (int)((1ull << 30) / bytes) + 1;   // hot: first 512 rows, L2-resident; cold: >= 1 GiB arena
    printf("k1 N %d K %d; %.1f MB per layout; hot = first %d rows (%.2f MB) repeated; warm = whole matrix repeated (memo-02 method); cold = %d copies (%.2f GB) round-robin; single thread, host, relative only; h6col TR %d\n",
           N, K, bytes / 1e6, nhot, bytes * (double) nhot / N / 1e6, ncold, ncold * (double) bytes / 1e9, TR);
#ifdef GGML_LINK
    { // copy vs lib, generic, vec_dot (Y1c) at density 1
        kargs a = { K, x, x, kept, nk, NULL, 0 }; float * yl = malloc(4 * (size_t) N), * yg = malloc(4 * (size_t) N), * yv = malloc(4 * (size_t) N);
        gemv_stock(K, ys, w4, x, N); kcall(KS_LIBGEMV, w4, N, yl, &a); ggml_gemv_q4_0_4x4_q8_0_generic(K, yg, 0, w4, x, 1, N); kcall(KS_VECDOT, w, N, yv, &a);
        double mg = 0, mv = 0; int ng = 0, nv = 0;
        for (int i = 0; i < N; i++) { double dg = fabs(yg[i] - ys[i]) / fabs(ys[i]), dv = fabs(yv[i] - ys[i]) / fabs(ys[i]); if (dg > mg) mg = dg; if (dv > mv) mv = dv; ng += yg[i] != ys[i]; nv += yv[i] != ys[i]; }
        printf("ref: bench copy == lib ggml_gemv_q4_0_4x4_q8_0 %s; lib generic gemv differs in %d of %d (max rel %.2e); lib vec_dot (Y1c, no repack) differs in %d of %d (max rel %.2e)\n",
               memcmp(ys, yl, 4 * (size_t) N) ? "NO" : "YES", ng, N, mg, nv, N, mv);
        free(yl); free(yg); free(yv);
    }
#endif
    const double dens[] = { 1.0, 0.65, 0.60 }; double ts[3] = { 0 };
    for (int di = 0; di < 3; di++) {
        memcpy(xz, x, sizeof(block_q8_0) * nb);
        for (int b = 0; b < nb; b++) { nk[b] = 0; for (int j = 0; j < 32; j++) if (dens[di] == 1.0 || (double) rand() / RAND_MAX < dens[di]) kept[b][nk[b]++] = j; else xz[b].qs[j] = 0; }
        int tot = 0; for (int b = 0; b < nb; b++) tot += nk[b]; const double d = (double) tot / K;
        kargs a = { K, x, xz, kept, nk, NULL, 0 };
        gemv_stock(K, ys, w4, xz, N);
        const int list[] = { KS_COL, KS_K1_128, KS_K1_512, KS_K1_128PF, KS_K1_512PF, KS_Y1B, KS_LIBGEMV, KS_VECDOT };
        for (int li = 0; li < 8; li++) {
            const int ks = list[li];
            if ((ks == KS_Y1B || ks == KS_LIBGEMV || ks == KS_VECDOT) && di) continue;   // dense-only kernels
#ifndef GGML_LINK
            if (ks == KS_LIBGEMV || ks == KS_VECDOT) continue;
#endif
            kcall(ks, lay[ks], N, yk, &a);
            const int same = memcmp(ys, yk, 4 * (size_t) N) == 0;
            if (ks != KS_VECDOT) printf("identity d %.2f (realised %.3f) %s == stock 4x4 gemv(zeroed x): %s\n", dens[di], d, ks_name[ks], same ? "YES" : "NO");
        }
        for (int reg = 0; reg < 3; reg++) {   // 0 hot, 1 warm, 2 cold
            const char * rn[] = { "hot", "warm", "cold" }; const int n = reg ? N : nhot, rp = reg ? reps : reps * (N / nhot);
            if (di == 0) { // stock reference per regime
                uint8_t * arena = (uint8_t *) w4;
                if (reg == 2) { arena = malloc(bytes * ncold); for (int c = 0; c < ncold; c++) memcpy(arena + c * bytes, w4, bytes); }
                ts[reg] = ktime(KS_STOCK, arena, bytes, reg == 2 ? ncold : 1, n, rp, ys, &a);
                if (reg == 2) free(arena);
                printf("time %s d 1.00 stock4x4 %.4f ms (%.3f ns/w)\n", rn[reg], ts[reg] * 1e3, ts[reg] * 1e9 / ((double) n * K));
            }
            const int list2[] = { KS_COL, KS_K1_128, KS_K1_512, KS_K1_128PF, KS_K1_512PF, KS_Y1B, KS_LIBGEMV, KS_VECDOT };
            for (int li = 0; li < 8; li++) {
                if (list2[li] == KS_COL && n < TR) continue;
                const int ks = list2[li];
                if ((ks == KS_Y1B || ks == KS_LIBGEMV || ks == KS_VECDOT) && di) continue;
#ifndef GGML_LINK
                if (ks == KS_LIBGEMV || ks == KS_VECDOT) continue;
#endif
                uint8_t * arena = (uint8_t *) lay[ks];
                if (reg == 2) { arena = malloc(bytes * ncold); for (int c = 0; c < ncold; c++) memcpy(arena + c * bytes, lay[ks], bytes); }
                const double t = ktime(ks, arena, bytes, reg == 2 ? ncold : 1, n, rp, yk, &a);
                if (reg == 2) free(arena);
                printf("time %s d %.2f (realised %.3f) %s %.4f ms; /stock dense %.3f; k (per touched weight) %.2f\n", rn[reg], dens[di], d, ks_name[ks], t * 1e3, t / ts[reg],
                       t / ((ks == KS_Y1B || ks == KS_LIBGEMV || ks == KS_VECDOT ? 1.0 : d) * ts[reg]));
            }
        }
    }
    { // E6: Y1b on a kept-row list (four arbitrary row pointers), kept fractions 1.0 / 0.5 / 0.3, random rows, sorted
        const double rf[] = { 1.0, 0.5, 0.3 }; int * rows = malloc(sizeof(int) * N); kargs a = { K, x, x, kept, nk, rows, 0 };
        gemv_stock(K, ys, w4, x, N);
        for (int fi = 0; fi < 3; fi++) for (int reg = 0; reg < 3; reg++) {
            const char * rn[] = { "hot", "warm", "cold" }; const int n = reg ? N : nhot, rp = reg ? reps : reps * (N / nhot); int m = 0;
            for (int r = 0; r < n; r++) if (rf[fi] == 1.0 || (double) rand() / RAND_MAX < rf[fi]) rows[m++] = r;
            m -= m % 4; a.nrows = m;
            if (reg == 1) { kcall(KS_Y1BROWS, w, 0, yk, &a); int same = 1; for (int q = 0; q < m; q++) if (memcmp(&yk[q], &ys[rows[q]], 4)) same = 0;
                printf("identity E6 rows %.2f (realised %.3f) Y1b-rows == stock 4x4 gemv on the same rows: %s\n", rf[fi], (double) m / n, same ? "YES" : "NO"); }
            uint8_t * arena = (uint8_t *) w;
            if (reg == 2) { arena = malloc(bytes * ncold); for (int c = 0; c < ncold; c++) memcpy(arena + c * bytes, w, bytes); }
            const double t = ktime(KS_Y1BROWS, arena, bytes, reg == 2 ? ncold : 1, 0, rp, yk, &a);
            if (reg == 2) free(arena);
            printf("time %s E6 rows %.2f (realised %.3f) Y1b-rows %.4f ms; /stock dense %.3f; k (per touched weight) %.2f\n", rn[reg], rf[fi], (double) m / n, t * 1e3, t / ts[reg], t / ((double) m / n * ts[reg]));
        }
        free(rows);
    }
#ifdef GGML_LINK
    { // T4: 4x4 gemm (8 activation rows) against gemv per row, and quantize_mat_q8_0_4x4 against quantize_row_q8_0
        const int M = 8; float * xf = malloc(sizeof(float) * (size_t) M * K);
        for (size_t i = 0; i < (size_t) M * K; i++) { float u = 0; for (int k = 0; k < 4; k++) u += (float) rand() / RAND_MAX - 0.5f; xf[i] = u * (i % 97 == 0 ? 20.f : 1.f); }
        block_q8_0 * qr = malloc(sizeof(block_q8_0) * (size_t) M * nb);
        for (int m = 0; m < M; m++) quantize_row_q8_0(xf + (size_t) m * K, qr + (size_t) m * nb, K);
        typedef struct { __fp16 d[4]; int8_t qs[128]; } block_q8_0x4;
        block_q8_0x4 * q4 = malloc(sizeof(block_q8_0x4) * (size_t)(M / 4) * nb);
        for (int g = 0; g < M / 4; g++) ggml_quantize_mat_q8_0_4x4(xf + (size_t) g * 4 * K, q4 + (size_t) g * nb, K);
        int qsame = 1;
        for (int m = 0; m < M; m++) for (int b = 0; b < nb; b++) {
            const block_q8_0x4 * p = &q4[(size_t)(m / 4) * nb + b]; const block_q8_0 * r = &qr[(size_t) m * nb + b];
            if (memcmp(&p->d[m % 4], &r->d, 2)) qsame = 0;
            for (int e = 0; e < 32; e++) if (p->qs[(e / 4) * 16 + (m % 4) * 4 + e % 4] != r->qs[e]) qsame = 0;
        }
        float * yg = malloc(sizeof(float) * (size_t) M * N), * yv = malloc(sizeof(float) * (size_t) N), * yy = malloc(sizeof(float) * (size_t) N);
        ggml_gemm_q4_0_4x4_q8_0(K, yg, N, w4, q4, M, N);
        int gsame = 1, ksame = 1, bsame = 1; size_t ndiff = 0; double mrel = 0;
        for (int m = 0; m < M; m++) {
            ggml_gemv_q4_0_4x4_q8_0(K, yv, 0, w4, qr + (size_t) m * nb, 1, N);
            if (memcmp(yg + (size_t) m * N, yv, 4 * (size_t) N)) gsame = 0;
            for (int i = 0; i < N; i++) if (yg[(size_t) m * N + i] != yv[i]) { ndiff++; double r = fabs(yg[(size_t) m * N + i] - yv[i]) / fabs(yv[i]); if (r > mrel) mrel = r; }
            for (int b = 0; b < nb; b++) { nk[b] = 32; for (int j = 0; j < 32; j++) kept[b][j] = j; }
            gemv_k1(N, K, 512, 0, yy, k512, qr + (size_t) m * nb, kept, nk); if (memcmp(yy, yv, 4 * (size_t) N)) ksame = 0;
            gemv_y1b(K, yy, w, qr + (size_t) m * nb, N); if (memcmp(yy, yv, 4 * (size_t) N)) bsame = 0;
        }
        printf("T4: quantize_mat_q8_0_4x4 == quantize_row_q8_0 per row: %s; lib gemm (8 rows) == lib gemv per row: %s (%zu of %zu differ, max rel %.2e); K1-tr512 per row == gemv: %s; Y1b per row == gemv: %s\n",
               qsame ? "YES" : "NO", gsame ? "YES" : "NO", ndiff, (size_t) M * N, mrel, ksame ? "YES" : "NO", bsame ? "YES" : "NO");
    }
#endif
    return 0;
}
// T7: exact distinct 64 B lines of a K1 layout touched by a random mask, per tile size, and the holes (runs of untouched
// lines) by length. One (tile, block) region is 18*tr bytes = scales (tr/32 lines) + 32 columns of tr/128 lines.
static int k1_lines_main(int N, int K) {
    const int nb = K / QK; const int trs[] = { 128, 512, 1024 }; const double dens[] = { 0.65, 0.60 };
    for (int di = 0; di < 2; di++) for (int ti = 0; ti < 3; ti++) {
        const int tr = trs[ti]; if (N % tr) { printf("lines d %.2f tr %d: N %% tr != 0, skipped\n", dens[di], tr); continue; }
        srand(7 + di);
        const size_t per = (size_t) tr * 18 / 64, tot = per * (size_t)(N / tr) * nb; uint8_t * seen = calloc(tot, 1);
        uint8_t (*keep)[32] = malloc(32 * (size_t) nb); size_t nkept = 0;
        for (int b = 0; b < nb; b++) for (int j = 0; j < 32; j++) { keep[b][j] = (double) rand() / RAND_MAX < dens[di]; nkept += keep[b][j]; }
        for (int t = 0; t < N / tr; t++) for (int b = 0; b < nb; b++) {   // mask is per input, shared by all tiles
            uint8_t * L = seen + ((size_t) t * nb + b) * per; int any = 0;
            for (int j = 0; j < 32; j++) if (keep[b][j]) { any = 1; for (int l = 0; l < tr / 128; l++) L[tr / 32 + j * (tr / 128) + l] = 1; }
            if (any) for (int l = 0; l < tr / 32; l++) L[l] = 1;
        }
        size_t touched = 0, hist[6] = { 0 }; const int edges[] = { 1, 2, 4, 8, 16 };   // hole lengths 1, 2-3, 4-7, 8-15, 16+, skipped lines in each
        size_t run = 0;
        for (size_t i = 0; i <= tot; i++) {
            if (i < tot && !seen[i]) { run++; continue; }
            if (i < tot) touched++;
            if (run) { int h = 0; while (h < 5 && run >= (size_t) edges[h]) h++; hist[h - 1] += run; run = 0; }
        }
        const size_t skipped = tot - touched;
        printf("lines d %.2f (realised %.4f) tr %d: touched %.4f of %zu lines (b = (4 + 32 d)/36 = %.4f); skipped lines in holes of 1 / 2-3 / 4-7 / 8-15 / 16+ lines: %.3f / %.3f / %.3f / %.3f / %.3f\n",
               dens[di], (double) nkept / K, tr, (double) touched / tot, tot, (4 + 32.0 * nkept / K) / 36,
               (double) hist[0] / skipped, (double) hist[1] / skipped, (double) hist[2] / skipped, (double) hist[3] / skipped, (double) hist[4] / skipped);
        free(seen); free(keep);
    }
    return 0;
}
int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 14336, K = argc > 2 ? atoi(argv[2]) : 4096, reps = argc > 3 ? atoi(argv[3]) : 50, nb = K / QK;
    if (argc > 4 && !strcmp(argv[4], "k1")) return k1_main(N, K, reps);
    if (argc > 4 && !strcmp(argv[4], "k1lines")) return k1_lines_main(N, K);
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
