// host microbench (campaign 3 memo-08, Tom): variants of K1 (the column kernel of patches/llama.cpp-c3/0002-d-rows.patch,
// TR 128) against ggml's NEON Q4_0 4x4 gemv, at the subjects' shapes (Yi-1.5-6B, Llama-2-7B, DeepSeek-7B share them:
// 11008x4096 gate/up, 4096x11008 down, 4096x4096 q/o). Bench only: nothing here is in a runtime, no knob.
// Every variant is checked bit for bit (memcmp) against the stock gemv fed the same q8 input with the dropped inputs
// set to 0, before it is timed.
//   stock  ggml_gemv_q4_0_4x4_q8_0 (NEON + dotprod path, copied from tools/c3_colkern_bench.c)
//   k1     0002's kernel verbatim: per kept column 4 x (AND, USHR, 4 SMLAL) into 16 int16x8 registers, float step per block
//   k1x2   k1 with two kept columns per iteration (register blocking: two independent load/extract chains)
//   k1t    TBL + SDOT: the kept columns are taken four at a time; one 4-register TBL per 4 rows transposes 4 columns x
//          4 rows into row-quads, then SDOT (16 MACs per instruction, int32, as stock) against the 4 kept x packed in one
//          32-bit lane. A ragged last quad repeats a kept column with x = 0 (adds exactly 0). Same K1 layout in memory.
//   k1m, k1m2, k1m3  K1 with an exact integer float step (see the K1m family below)
//   k1z    k1t with the transposition done by ZIP (see k1z)
//   k1tp   k1t with software prefetch of block b+PF's kept columns (PF = -DPFD, default 2)
// List building (bitmask of |x| over threshold -> kept index list, packed x quads, sum of kept x) runs once per call and
// is timed inside every column variant; its share is printed separately ("lists").
// k = ns per touched weight / stock ns per weight, same regime and thread count.
// build: clang -O3 -march=armv8.2-a+dotprod+fp16 -o lab/c3/bench/k1v tools/c3_k1v_bench.c -lpthread
// usage: k1v <rows> <inputs> <keep> <threads> [reps=20]
#include <arm_neon.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif
#define QK 32
#define KT 128
#define KB (KT * 18)
#ifndef PFD
#define PFD 2
#endif
typedef struct { __fp16 d; uint8_t qs[16]; } bq4;
typedef struct { __fp16 d; int8_t qs[32]; } bq8;
typedef struct { __fp16 d[4]; uint8_t qs[64]; } bq4x4;

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) rs; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ---- stock (ggml_gemv_q4_0_4x4_q8_0, NEON), rows [r0, r1)
static __attribute__((noinline)) void stock(const uint8_t * w, int nb, const bq8 * x, float * y, int r0, int r1) {
    for (int c = r0; c < r1; c += 4) {
        const bq4x4 * b = (const bq4x4 *) w + (size_t)(c / 4) * nb; float32x4_t acc = vdupq_n_f32(0);
        for (int k = 0; k < nb; k++, b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *) b->qs), b1 = vld1q_s8((const int8_t *) b->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b->qs + 32), b3 = vld1q_s8((const int8_t *) b->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b->d);
            int8x16_t a0 = vld1q_s8(x[k].qs), a1 = vld1q_s8(x[k].qs + 16);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &x[k].d);
            int32x4_t r = vdupq_n_s32(0);
            r = vdotq_laneq_s32(r, b0 << 4, a0, 0); r = vdotq_laneq_s32(r, b1 << 4, a0, 1);
            r = vdotq_laneq_s32(r, b2 << 4, a0, 2); r = vdotq_laneq_s32(r, b3 << 4, a0, 3);
            r = vdotq_laneq_s32(r, b0 & 0xf0U, a1, 0); r = vdotq_laneq_s32(r, b1 & 0xf0U, a1, 1);
            r = vdotq_laneq_s32(r, b2 & 0xf0U, a1, 2); r = vdotq_laneq_s32(r, b3 & 0xf0U, a1, 3);
            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(r, 4), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
        }
        vst1q_f32(y + c, acc);
    }
}

// ---- per-call kept lists (built from a bitmask, timed with the kernels)
typedef struct { int nb; uint32_t * mask; uint8_t * kc; int * nk; int * sx; int32_t * xq; int * jq; int * nq; } Lists;
static void build_lists(Lists * L, const bq8 * x) {
    for (int b = 0; b < L->nb; b++) {
        uint32_t m = L->mask[b]; int n = 0, s = 0; uint8_t * kc = L->kc + b * 32; int8_t * xq = (int8_t *)(L->xq + b * 8); int * jq = L->jq + b * 32;
        while (m) { int j = __builtin_ctz(m); m &= m - 1; kc[n] = (uint8_t) j; xq[n] = x[b].qs[j]; jq[n] = j * 64; s += x[b].qs[j]; n++; }
        L->nk[b] = n; L->sx[b] = s; int q = (n + 3) / 4; L->nq[b] = q;
        for (int i = n; i < q * 4; i++) { xq[i] = 0; jq[i] = n ? jq[0] : 0; }   // pad: any kept column, x = 0
    }
}

// float step shared by the int16 variants: 16 registers = rows 8i..8i+7 of the tile
#define K1_FLOAT16(a, af, d, corr, vdx) \
    for (int i = 0; i < 16; i++) { \
        const int16x8_t v = vsubq_s16(a[i], corr); const float16x8_t dw = vld1q_f16(d + 8 * i); \
        vst1q_f32(af + 8 * i, vfmaq_f32(vld1q_f32(af + 8 * i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))), vmulq_f32(vdx, vcvt_f32_f16(vget_low_f16(dw))))); \
        vst1q_f32(af + 8 * i + 4, vfmaq_f32(vld1q_f32(af + 8 * i + 4), vcvtq_f32_s32(vmovl_high_s16(v)), vmulq_f32(vdx, vcvt_high_f32_f16(dw)))); }

#define K1_COL(c, xv, xh, a) \
    _Pragma("clang loop unroll(full)") for (int L_ = 0; L_ < 4; L_++) { \
        const uint8x16_t v_ = vld1q_u8((c) + 16 * L_); \
        const int8x16_t lo_ = vreinterpretq_s8_u8(vandq_u8(v_, m4)), hi_ = vreinterpretq_s8_u8(vshrq_n_u8(v_, 4)); \
        a[4 * L_ + 0] = vmlal_s8(a[4 * L_ + 0], vget_low_s8(lo_), xh); a[4 * L_ + 1] = vmlal_high_s8(a[4 * L_ + 1], lo_, xv); \
        a[4 * L_ + 2] = vmlal_s8(a[4 * L_ + 2], vget_low_s8(hi_), xh); a[4 * L_ + 3] = vmlal_high_s8(a[4 * L_ + 3], hi_, xv); }

static void k1(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    for (int t = t0; t < t1; t++) {
        float * af = y + (size_t) t * KT;
        for (int i = 0; i < KT; i += 4) vst1q_f32(af + i, vdupq_n_f32(0));
        for (int b = 0; b < nb; b++) {
            const int nk = Ls->nk[b]; if (!nk) continue;
            const uint8_t * p = w + ((size_t) t * nb + b) * KB; const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + KT * 2;
            const uint8_t * kc = Ls->kc + b * 32;
            const int16x8_t corr = vdupq_n_s16((int16_t)(8 * Ls->sx[b])); const float32x4_t vdx = vdupq_n_f32((float) x[b].d);
            int16x8_t a[16];
            for (int i = 0; i < 16; i++) a[i] = vdupq_n_s16(0);
            for (int q = 0; q < nk; q++) {
                const int j = kc[q]; const int8x16_t xv = vdupq_n_s8(x[b].qs[j]); const int8x8_t xh = vget_low_s8(xv);
                K1_COL(col + j * 64, xv, xh, a)
            }
            K1_FLOAT16(a, af, d, corr, vdx)
        }
    }
}

static void k1x2(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    for (int t = t0; t < t1; t++) {
        float * af = y + (size_t) t * KT;
        for (int i = 0; i < KT; i += 4) vst1q_f32(af + i, vdupq_n_f32(0));
        for (int b = 0; b < nb; b++) {
            const int nk = Ls->nk[b]; if (!nk) continue;
            const uint8_t * p = w + ((size_t) t * nb + b) * KB; const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + KT * 2;
            const uint8_t * kc = Ls->kc + b * 32;
            const int16x8_t corr = vdupq_n_s16((int16_t)(8 * Ls->sx[b])); const float32x4_t vdx = vdupq_n_f32((float) x[b].d);
            int16x8_t a[16], e[16];   // two accumulator sets (sums wrap mod 2^16 and are added at the end: exact, as k1)
            for (int i = 0; i < 16; i++) a[i] = e[i] = vdupq_n_s16(0);
            int q = 0;
            for (; q + 1 < nk; q += 2) {
                const int j0 = kc[q], j1 = kc[q + 1];
                const int8x16_t xv0 = vdupq_n_s8(x[b].qs[j0]), xv1 = vdupq_n_s8(x[b].qs[j1]);
                const int8x8_t xh0 = vget_low_s8(xv0), xh1 = vget_low_s8(xv1);
                K1_COL(col + j0 * 64, xv0, xh0, a)
                K1_COL(col + j1 * 64, xv1, xh1, e)
            }
            if (q < nk) { const int j = kc[q]; const int8x16_t xv = vdupq_n_s8(x[b].qs[j]); const int8x8_t xh = vget_low_s8(xv); K1_COL(col + j * 64, xv, xh, a) }
            for (int i = 0; i < 16; i++) a[i] = vaddq_s16(a[i], e[i]);
            K1_FLOAT16(a, af, d, corr, vdx)
        }
    }
}

// K1m family: the float step without SCVTF (and in k1m3 without FCVTL), exact by construction.
//   magic convert: for |i| < 2^22, bits(i + 0x4B400000) as float = 12582912 + i exactly, and FSUB 12582912 gives i exactly,
//   i.e. what SCVTF gives (the block sums here are |i| <= 8 * 127 * 32 = 32512).
//   k1m : SSHLL (widen) + ADD magic + FSUB            (Bonnie, memo-08 bonnie-path.md section 2)
//   k1m2: SADDW (widen and add magic in one) + FSUB
//   k1m3: k1m2, and the fp16 scale without FCVTL: y = bits((h & 0x7fff) << 13) = |dw| * 2^-112 exactly (normal and subnormal
//         halves alike), multiplied by dx * 2^112 (exact: a power of two, dx < 2^16): round(dx2 * y) = round(dx * |dw|). The
//         sign of dw goes into the integer sum (i -> -i where dw < 0): fma(acc, -i, p) = fma(acc, i, -p) exactly. A -0 scale
//         differs only when acc is -0, and acc starts at +0 and cannot become -0 under round-to-nearest.
//   k1m4: k1m2, and the signed scale in two ops: SSHLL #13 sign-extends h, so bit 31 is the sign and bits 30..28 copy it;
//         BIC 0x70000000 clears them: bits = sign | (h & 0x7fff) << 13 = dw * 2^-112 with its sign (-0 stays -0). No sign
//         trick on the sum: p = round(dx2 * y) = round(dx * dw) exactly, as stock.
#define K1M_FLOAT16(a, af, d, corr, vdx, MODE) \
    for (int i = 0; i < 16; i++) { \
        int16x8_t v = vsubq_s16(a[i], corr); float32x4_t pl, ph; \
        if (MODE == 3) { const uint16x8_t h = vld1q_u16((const uint16_t *)(d + 8 * i)); const int16x8_t sg = vshrq_n_s16(vreinterpretq_s16_u16(h), 15); \
            v = vsubq_s16(veorq_s16(v, sg), sg); const uint16x8_t mg = vandq_u16(h, vdupq_n_u16(0x7fff)); \
            pl = vmulq_f32(vdx, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(mg), 13))); ph = vmulq_f32(vdx, vreinterpretq_f32_u32(vshll_high_n_u16(mg, 13))); } \
        else if (MODE == 4) { const int16x8_t h = vld1q_s16((const int16_t *)(d + 8 * i)); \
            pl = vmulq_f32(vdx, vreinterpretq_f32_s32(vbicq_s32(vshll_n_s16(vget_low_s16(h), 13), vdupq_n_s32(0x70000000)))); \
            ph = vmulq_f32(vdx, vreinterpretq_f32_s32(vbicq_s32(vshll_high_n_s16(h, 13), vdupq_n_s32(0x70000000)))); } \
        else { const float16x8_t dw = vld1q_f16(d + 8 * i); pl = vmulq_f32(vdx, vcvt_f32_f16(vget_low_f16(dw))); ph = vmulq_f32(vdx, vcvt_high_f32_f16(dw)); } \
        int32x4_t il, ih; \
        if (MODE == 1) { il = vaddq_s32(vmovl_s16(vget_low_s16(v)), vmagic); ih = vaddq_s32(vmovl_high_s16(v), vmagic); } \
        else { il = vaddw_s16(vmagic, vget_low_s16(v)); ih = vaddw_high_s16(vmagic, v); } \
        const float32x4_t fl = vsubq_f32(vreinterpretq_f32_s32(il), vmagf), fh = vsubq_f32(vreinterpretq_f32_s32(ih), vmagf); \
        vst1q_f32(af + 8 * i, vfmaq_f32(vld1q_f32(af + 8 * i), fl, pl)); vst1q_f32(af + 8 * i + 4, vfmaq_f32(vld1q_f32(af + 8 * i + 4), fh, ph)); }

static inline __attribute__((always_inline)) void k1m_body(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1, const int MODE) {
    const uint8x16_t m4 = vdupq_n_u8(15); const int32x4_t vmagic0 = vdupq_n_s32(0x4B400000); const float32x4_t vmagf0 = vdupq_n_f32(12582912.0f);
    for (int t = t0; t < t1; t++) {
        float * af = y + (size_t) t * KT;
        for (int i = 0; i < KT; i += 4) vst1q_f32(af + i, vdupq_n_f32(0));
        for (int b = 0; b < nb; b++) {
            const int nk = Ls->nk[b]; if (!nk) continue;
            const uint8_t * p = w + ((size_t) t * nb + b) * KB; const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + KT * 2;
            const uint8_t * kc = Ls->kc + b * 32;
            const int16x8_t corr = vdupq_n_s16((int16_t)(8 * Ls->sx[b]));
            const float32x4_t vdx = vdupq_n_f32(MODE >= 3 ? (float) x[b].d * 0x1p112f : (float) x[b].d);
            int16x8_t a[16];
            for (int i = 0; i < 16; i++) a[i] = vdupq_n_s16(0);
            for (int q = 0; q < nk; q++) {
                const int j = kc[q]; const int8x16_t xv = vdupq_n_s8(x[b].qs[j]); const int8x8_t xh = vget_low_s8(xv);
                K1_COL(col + j * 64, xv, xh, a)
            }
            int32x4_t vmagic = vmagic0; float32x4_t vmagf = vmagf0; __asm__ volatile("" : "+w"(vmagic), "+w"(vmagf));   // keep the constants in registers (no per-row DUP)
            K1M_FLOAT16(a, af, d, corr, vdx, MODE)
        }
    }
}
static void k1m(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) { k1m_body(w, nb, x, Ls, y, t0, t1, 1); }
static void k1m2(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) { k1m_body(w, nb, x, Ls, y, t0, t1, 2); }
static void k1m3(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) { k1m_body(w, nb, x, Ls, y, t0, t1, 3); }
static void k1m4(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) { k1m_body(w, nb, x, Ls, y, t0, t1, 4); }

// TBL index: output byte 4r + c (row r of 4, column c of 4) = table byte 16c + 4v + r, for row group v
static uint8x16_t TIDX[4];
static void init_tidx(void) {
    for (int v = 0; v < 4; v++) { uint8_t t[16]; for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) t[4 * r + c] = (uint8_t)(16 * c + 4 * v + r); TIDX[v] = vld1q_u8(t); }
}
static inline __attribute__((always_inline)) void k1t_body(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1, int pf) {
    const uint8x16_t m4 = vdupq_n_u8(15); const uint8x16_t i0 = TIDX[0], i1 = TIDX[1], i2 = TIDX[2], i3 = TIDX[3];
    for (int t = t0; t < t1; t++) {
        float * af = y + (size_t) t * KT;
        for (int i = 0; i < KT; i += 4) vst1q_f32(af + i, vdupq_n_f32(0));
        for (int b = 0; b < nb; b++) {
            const int nq = Ls->nq[b]; if (!nq) continue;
            const uint8_t * p = w + ((size_t) t * nb + b) * KB; const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + KT * 2;
            if (pf && b + PFD < nb) { const uint8_t * pp = w + ((size_t) t * nb + b + PFD) * KB; const int * jp = Ls->jq + (b + PFD) * 32;
                __builtin_prefetch(pp); __builtin_prefetch(pp + 64); __builtin_prefetch(pp + 128); __builtin_prefetch(pp + 192);
                for (int q = 0; q < Ls->nk[b + PFD]; q++) __builtin_prefetch(pp + KT * 2 + jp[q]); }
            const int * jq = Ls->jq + b * 32; const int32_t * xq = Ls->xq + b * 8;
            const int32x4_t corr = vdupq_n_s32(8 * Ls->sx[b]); const float32x4_t vdx = vdupq_n_f32((float) x[b].d);
            for (int L = 0; L < 4; L++) {   // 32 rows: low nibbles rows 32L + 0..15, high nibbles rows 32L + 16..31
                int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0, h0 = a0, h1 = a0, h2 = a0, h3 = a0;
                const uint8_t * cl = col + 16 * L;
                for (int q = 0; q < nq; q++) {
                    uint8x16x4_t tb; tb.val[0] = vld1q_u8(cl + jq[4 * q]); tb.val[1] = vld1q_u8(cl + jq[4 * q + 1]);
                    tb.val[2] = vld1q_u8(cl + jq[4 * q + 2]); tb.val[3] = vld1q_u8(cl + jq[4 * q + 3]);
                    const int8x16_t xv = vreinterpretq_s8_s32(vdupq_n_s32(xq[q]));
                    uint8x16_t u;
                    u = vqtbl4q_u8(tb, i0); a0 = vdotq_s32(a0, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h0 = vdotq_s32(h0, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                    u = vqtbl4q_u8(tb, i1); a1 = vdotq_s32(a1, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h1 = vdotq_s32(h1, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                    u = vqtbl4q_u8(tb, i2); a2 = vdotq_s32(a2, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h2 = vdotq_s32(h2, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                    u = vqtbl4q_u8(tb, i3); a3 = vdotq_s32(a3, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h3 = vdotq_s32(h3, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                }
                const int32x4_t s[8] = { a0, a1, a2, a3, h0, h1, h2, h3 };
                for (int g = 0; g < 8; g++) {
                    const int r = 32 * L + (g < 4 ? 4 * g : 16 + 4 * (g - 4));
                    const float32x4_t dw = vcvt_f32_f16(vld1_f16(d + r));
                    vst1q_f32(af + r, vfmaq_f32(vld1q_f32(af + r), vcvtq_f32_s32(vsubq_s32(s[g], corr)), vmulq_f32(vdx, dw)));
                }
            }
        }
    }
}
// k1z: as k1t, but the 4x4 transposition is two ZIP levels (8 ZIP per 4 columns x 16 rows) instead of 4 TBL4: the same
// vector-op count as k1 (24 per kept quad and 32 rows) with SDOT on any vector pipe instead of SMLAL
static void k1z(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    for (int t = t0; t < t1; t++) {
        float * af = y + (size_t) t * KT;
        for (int i = 0; i < KT; i += 4) vst1q_f32(af + i, vdupq_n_f32(0));
        for (int b = 0; b < nb; b++) {
            const int nq = Ls->nq[b]; if (!nq) continue;
            const uint8_t * p = w + ((size_t) t * nb + b) * KB; const __fp16 * d = (const __fp16 *) p; const uint8_t * col = p + KT * 2;
            const int * jq = Ls->jq + b * 32; const int32_t * xq = Ls->xq + b * 8;
            const int32x4_t corr = vdupq_n_s32(8 * Ls->sx[b]); const float32x4_t vdx = vdupq_n_f32((float) x[b].d);
            for (int L = 0; L < 4; L++) {
                int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0, h0 = a0, h1 = a0, h2 = a0, h3 = a0;
                const uint8_t * cl = col + 16 * L;
                for (int q = 0; q < nq; q++) {
                    const uint8x16_t c0 = vld1q_u8(cl + jq[4 * q]), c1 = vld1q_u8(cl + jq[4 * q + 1]), c2 = vld1q_u8(cl + jq[4 * q + 2]), c3 = vld1q_u8(cl + jq[4 * q + 3]);
                    const int8x16_t xv = vreinterpretq_s8_s32(vdupq_n_s32(xq[q]));
                    const uint16x8_t p0 = vreinterpretq_u16_u8(vzip1q_u8(c0, c1)), p1 = vreinterpretq_u16_u8(vzip2q_u8(c0, c1));
                    const uint16x8_t r0 = vreinterpretq_u16_u8(vzip1q_u8(c2, c3)), r1 = vreinterpretq_u16_u8(vzip2q_u8(c2, c3));
                    uint8x16_t u;
                    u = vreinterpretq_u8_u16(vzip1q_u16(p0, r0)); a0 = vdotq_s32(a0, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h0 = vdotq_s32(h0, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                    u = vreinterpretq_u8_u16(vzip2q_u16(p0, r0)); a1 = vdotq_s32(a1, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h1 = vdotq_s32(h1, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                    u = vreinterpretq_u8_u16(vzip1q_u16(p1, r1)); a2 = vdotq_s32(a2, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h2 = vdotq_s32(h2, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                    u = vreinterpretq_u8_u16(vzip2q_u16(p1, r1)); a3 = vdotq_s32(a3, vreinterpretq_s8_u8(vandq_u8(u, m4)), xv); h3 = vdotq_s32(h3, vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), xv);
                }
                const int32x4_t s[8] = { a0, a1, a2, a3, h0, h1, h2, h3 };
                for (int g = 0; g < 8; g++) {
                    const int r = 32 * L + (g < 4 ? 4 * g : 16 + 4 * (g - 4));
                    const float32x4_t dw = vcvt_f32_f16(vld1_f16(d + r));
                    vst1q_f32(af + r, vfmaq_f32(vld1q_f32(af + r), vcvtq_f32_s32(vsubq_s32(s[g], corr)), vmulq_f32(vdx, dw)));
                }
            }
        }
    }
}
static void k1t(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) { k1t_body(w, nb, x, Ls, y, t0, t1, 0); }
static void k1tp(const uint8_t * w, int nb, const bq8 * x, const Lists * Ls, float * y, int t0, int t1) { k1t_body(w, nb, x, Ls, y, t0, t1, 1); }

typedef void (*colfn)(const uint8_t *, int, const bq8 *, const Lists *, float *, int, int);

// ---- layouts from the file's row blocks
static void to_4x4(const bq4 * w, int N, int nb, uint8_t * out) {
    for (int r = 0; r < N; r += 4) for (int b = 0; b < nb; b++) {
        bq4x4 * o = (bq4x4 *) out + (size_t)(r / 4) * nb + b;
        for (int k = 0; k < 4; k++) o->d[k] = w[(size_t)(r + k) * nb + b].d;
        for (int i = 0; i < 16; ++i) { uint32_t e; memcpy(&e, &w[(size_t)(r + i % 4) * nb + b].qs[(i / 4) * 4], 4); e ^= 0x88888888u; memcpy(&o->qs[i * 4], &e, 4); }
    }
}
static void to_k1(const bq4 * w, int N, int nb, uint8_t * out) {
    for (int t = 0; t < N / KT; t++) for (int b = 0; b < nb; b++) {
        uint8_t * p = out + ((size_t) t * nb + b) * KB; __fp16 * d = (__fp16 *) p; uint8_t * col = p + KT * 2;
        for (int r = 0; r < KT; r++) d[r] = w[(size_t)(t * KT + r) * nb + b].d;
        for (int j = 0; j < 32; j++) for (int k = 0; k < 64; k++) {
            const int r0 = t * KT + (k / 16) * 32 + k % 16; const bq4 * lo = &w[(size_t) r0 * nb + b], * hi = &w[(size_t)(r0 + 16) * nb + b];
            const uint8_t ql = j < 16 ? lo->qs[j] & 15 : lo->qs[j - 16] >> 4, qh = j < 16 ? hi->qs[j] & 15 : hi->qs[j - 16] >> 4;
            col[j * 64 + k] = ql | (qh << 4);
        }
    }
}

// ---- threads: persistent workers, spin on a generation counter
typedef struct { int kind; colfn f; const uint8_t * w; int nb; const bq8 * x; const Lists * L; float * y; int N; } Job;
static Job job; static atomic_int gen, done; static int NT = 1;
static void run_part(int id) {
    if (job.kind == 0) { int g = job.N / 4, a = (int)((long) g * id / NT) * 4, e = (int)((long) g * (id + 1) / NT) * 4; stock(job.w, job.nb, job.x, job.y, a, e); }
    else { int g = job.N / KT, a = (int)((long) g * id / NT), e = (int)((long) g * (id + 1) / NT); job.f(job.w, job.nb, job.x, job.L, job.y, a, e); }
}
static void * worker(void * arg) {
    int id = (int)(long) arg, seen = 0;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    for (;;) { int g; while ((g = atomic_load_explicit(&gen, memory_order_acquire)) == seen) {} seen = g; if (g < 0) return 0; run_part(id); atomic_fetch_add_explicit(&done, 1, memory_order_release); }
}
static void run_job(void) {
    atomic_store(&done, 0); atomic_fetch_add_explicit(&gen, 1, memory_order_release); run_part(0);
    while (atomic_load_explicit(&done, memory_order_acquire) < NT - 1) ;
}

// exhaustive checks of the two K1m tricks (run at start): every int in [-2^22, 2^22) through the magic convert against
// SCVTF; every fp16 bit pattern h (finite) through k1m3's scale path against FCVTL, product with 4096 dx values
static int selfcheck(void) {
    int bad = 0; const int32x4_t mg = vdupq_n_s32(0x4B400000); const float32x4_t mf = vdupq_n_f32(12582912.0f);
    for (int i = -(1 << 22); i < (1 << 22); i += 4) {
        const int32x4_t v = { i, i + 1, i + 2, i + 3 };
        const float32x4_t a = vcvtq_f32_s32(v), b = vsubq_f32(vreinterpretq_f32_s32(vaddq_s32(v, mg)), mf);
        bad += memcmp(&a, &b, 16) != 0;
    }
    for (int k = 0; k < 4096; k++) {
        __fp16 dxh; uint16_t hb = (uint16_t)(rnd() & 0x7bff); memcpy(&dxh, &hb, 2); if (k == 0) dxh = (__fp16) 65504.f;
        const float dx = (float) dxh, dx2 = dx * 0x1p112f;
        for (uint32_t h = 0; h < 65536; h++) {
            if ((h & 0x7c00) == 0x7c00) continue;   // inf / nan scales do not occur in a valid file
            __fp16 hh; uint16_t hu = (uint16_t) h; memcpy(&hh, &hu, 2);
            const float ref = dx * (float) hh; uint32_t yb = (h & 0x7fff) << 13; float y; memcpy(&y, &yb, 4);
            float got = dx2 * y; if (h & 0x8000) got = -got;
            { int32_t sb = ((int32_t)(int16_t) hu) * 8192; uint32_t ub = (uint32_t) sb & ~0x70000000u; float y4; memcpy(&y4, &ub, 4); const float g4 = dx2 * y4; bad += memcmp(&ref, &g4, 4) != 0; }   // the sign goes to the integer sum in the kernel: same value
            bad += memcmp(&ref, &got, 4) != 0 && !(ref == 0 && got == 0);   // +-0 products: see the -0 argument above
        }
    }
    return bad;
}
int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: k1v <rows> <inputs> <keep> <threads> [reps]\n"); return 1; }
    const int N = atoi(argv[1]), K = atoi(argv[2]), reps = argc > 5 ? atoi(argv[5]) : 20; const double keep = atof(argv[3]); NT = atoi(argv[4]);
    const int nb = K / QK; const size_t mb = (size_t) N * nb * 18;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    init_tidx();
    if (getenv("K1V_SELFCHECK")) { int b = selfcheck(); printf("selfcheck magic+scale: %d mismatches\n", b); if (b) return 2; }
    pthread_t th[16]; for (int i = 1; i < NT; i++) pthread_create(&th[i], 0, worker, (void *)(long) i);
    // weights, input, mask
    bq4 * w = malloc(mb); for (size_t i = 0; i < (size_t) N * nb; i++) { { const uint32_t u = rnd() & 1023; float dv = 0.001f + 0.01f * (rnd() & 1023) / 1023.f; if (rnd() & 1) dv = -dv;
        w[i].d = (__fp16) dv; if (u < 10) w[i].d = (__fp16) -0.0f; else if (u < 15) w[i].d = (__fp16) 0.0f; else if (u < 25) { uint16_t h = (uint16_t)((rnd() & 0x3ff) | (rnd() & 0x8000)); memcpy(&w[i].d, &h, 2); } else if (u < 30) w[i].d = (__fp16)(dv * 4000.f); } for (int k = 0; k < 16; k++) w[i].qs[k] = (uint8_t) rnd(); }
    bq8 * x = malloc(sizeof(bq8) * nb), * xz = malloc(sizeof(bq8) * nb);
    for (int b = 0; b < nb; b++) { x[b].d = (__fp16)(0.01f + 0.02f * (rnd() & 1023) / 1023.f); if (b % 97 == 5) x[b].d = (__fp16) 0.0f; if (b % 89 == 7) x[b].d = (__fp16) 60000.f; for (int k = 0; k < 32; k++) x[b].qs[k] = (int8_t)((int)(rnd() % 255) - 127); }
    Lists L = { nb, calloc(nb, 4), calloc(nb, 32), calloc(nb, sizeof(int)), calloc(nb, sizeof(int)), calloc(nb, 32), calloc(nb, 32 * sizeof(int)), calloc(nb, sizeof(int)) };
    long kept = 0; memcpy(xz, x, sizeof(bq8) * nb);
    for (int b = 0; b < nb; b++) for (int j = 0; j < 32; j++) { if ((rnd() & 0xffff) < keep * 65536) { L.mask[b] |= 1u << j; kept++; } else xz[b].qs[j] = 0; }
    const double dr = kept / (double) K;
    // identity: each column variant against stock on xz
    uint8_t * w4 = malloc(mb), * wk = malloc(mb); to_4x4(w, N, nb, w4); to_k1(w, N, nb, wk);
    float * ys = malloc(4 * N), * yk = malloc(4 * N);
    const char * names[] = { "k1", "k1m", "k1m2", "k1m3", "k1m4", "k1x2", "k1t", "k1tp", "k1z" }; colfn fns[] = { k1, k1m, k1m2, k1m3, k1m4, k1x2, k1t, k1tp, k1z }; const int NV = 9;
    job = (Job){ 0, 0, w4, nb, xz, 0, ys, N }; run_job();
    int idok[16];
    for (int v = 0; v < NV; v++) { memset(yk, 0, 4 * N); build_lists(&L, x); job = (Job){ 1, fns[v], wk, nb, x, &L, yk, N }; run_job(); idok[v] = !memcmp(ys, yk, 4 * N); }
    // timing arenas: warm = the whole matrix repeated; cold = copies through ~1 GB
    const size_t AR = (size_t) 1 << 30; const int ncopy = (int)(AR / mb) < 2 ? 2 : (int)(AR / mb);
    uint8_t * a4 = malloc(mb * ncopy), * ak = malloc(mb * ncopy);
    for (int c = 0; c < ncopy; c++) { memcpy(a4 + c * mb, w4, mb); memcpy(ak + c * mb, wk, mb); }
    printf("shape %dx%d keep %.3f (realised %.3f) threads %d copies %d MB %.1f identity", N, K, keep, dr, NT, ncopy, mb / 1048576.0);
    for (int v = 0; v < NV; v++) printf(" %s=%d", names[v], idok[v]); printf("\n");
    for (int cold = 0; cold < 2; cold++) {
        double best[16]; for (int v = 0; v < 16; v++) best[v] = 1e30;
        for (int trial = 0; trial < 5; trial++) {
            for (int v = -1; v <= NV; v++) {   // -1 stock, 0..NV-1 column variants, NV = lists only; stock again at the end of trial
                double t0 = now();
                for (int r = 0; r < reps; r++) {
                    const int c = cold ? (trial * reps + r) % ncopy : 0;
                    if (v == NV) { build_lists(&L, x); continue; }
                    if (v < 0) job = (Job){ 0, 0, a4 + c * mb, nb, x, 0, ys, N };
                    else { build_lists(&L, x); job = (Job){ 1, fns[v], ak + c * mb, nb, x, &L, yk, N }; }
                    run_job();
                }
                const double t = (now() - t0) / reps; if (t < best[v + 1]) best[v + 1] = t;
            }
        }
        const double st = best[0] / ((double) N * K);
        printf("%s stock ns/w %.4f (%.1f GB/s)", cold ? "cold" : "warm", st * 1e9, mb / best[0] / 1e9);
        for (int v = 0; v < NV; v++) { const double tw = best[v + 1] / ((double) N * kept); printf(" | %s k %.3f t/stock %.3f", names[v], tw / st, best[v + 1] / best[0]); }
        printf(" | lists %.1f us (%.2f %% of stock call)\n", best[NV + 1] * 1e6, 100 * best[NV + 1] / best[0]);
    }
    atomic_store(&gen, -1); for (int i = 1; i < NT; i++) pthread_join(th[i], 0);
    return 0;
}
