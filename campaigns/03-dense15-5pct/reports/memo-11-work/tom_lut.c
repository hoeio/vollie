// c3 memo-11 U2 (Tom): scratch T-MAC-style LUT gemv for uniform 2-bit and 3-bit weights against the repacked
// Q4_0 4x4 gemv (ggml_gemv_q4_0_4x4_q8_0, NEON + dotprod path of base 620d2225c, copied from tools/c3_k1v_bench.c).
// Scratch only: no runtime code, nothing in patches/.
//
// LUT format (per 16-row tile, per block of G inputs): 16 fp16 row scales, then for plane p = B-1 .. 0 and pair
// j < G/8 one 16-byte vector: byte l = row l's 4-bit index of group 2j (low nibble) and group 2j+1 (high nibble).
// Weight w = s * (q - (2^B-1)/2) = s/2 * sum_p 2^p sigma_p, sigma = +-1 (symmetric grid, no zero point).
// Table per group of 4 inputs: T[i] = sum_j sigma_j(i) x_j, 16 entries, quantised to int8 with one scale per G-block
// (ts = max |T| / 127). Kernel: TBL per (16 rows, group, plane), widening add to int16 per plane, Horner over planes
// (int16 holds 7 * 127 * G/4 for G <= 128), then float: y += float(acc) * s * ts.
//
// build (host):  clang -O3 -march=armv8.2-a+dotprod+fp16 [-DG=128] [-DACC=2|4] -o lut tom_lut.c -lpthread
// run:           ./lut <rows> <inputs> <threads> <copies> [reps]     copies 1 = hot (same matrix), >1 = cold ring
//                LUT_OUTLIERS=0.005 puts 0.5 % of the input channels at 20x (NMSE line only matters then)
// A78 asm:       <NDK r28>/aarch64-linux-android28-clang -O3 -mcpu=cortex-a78 -DASM_ONLY [-DG=..] [-DACC=..] -S -o lut-a78.s tom_lut.c
//                then one function per file (awk '$1=="lut2:"{p=1} p&&/^\.Lfunc_end/{exit} p') for tools/c3_a78_pipes.py
// The plane loop carries an unroll(full) pragma so that the innermost loop is one block (needed for the pipe count);
// without it (tom-u2.md "run 2") G=128 ACC=1 was faster on the host.
#include <arm_neon.h>
#include <math.h>
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
#ifndef G
#define G 32
#endif
#ifndef ACC
#define ACC 1   // lookups added in int8 before widening: 1 = T-MAC int8 tables (+-127); 2, 4 = coarser tables (+-63, +-31), lossy
#endif
#define QK 32
typedef struct { __fp16 d; int8_t qs[32]; } bq8;
typedef struct { __fp16 d[4]; uint8_t qs[64]; } bq4x4;

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t) rs; }
static double gauss(void) { double u = (rnd() + 1.0) / 4294967297.0, v = rnd() / 4294967296.0; return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ---- stock: ggml_gemv_q4_0_4x4_q8_0 (NEON), rows [r0, r1)
__attribute__((noinline)) void stock(const uint8_t * w, int nb, const bq8 * x, float * y, int r0, int r1) {
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
// ggml quantize_row_q8_0 (round to nearest)
__attribute__((noinline)) void quant_q8(const float * x, bq8 * q, int K) {
    for (int b = 0; b < K / QK; b++) {
        float m = 0; for (int j = 0; j < QK; j++) m = fmaxf(m, fabsf(x[b * QK + j]));
        const float d = m / 127, id = d ? 1 / d : 0; q[b].d = (__fp16) d;
        for (int j = 0; j < QK; j++) q[b].qs[j] = (int8_t) roundf(x[b * QK + j] * id);
    }
}

// ---- LUT tables: tab[g*16 + i] (int8), ts[blk] (includes the 1/2 of the weight grid)
__attribute__((noinline)) void build_tab(const float * x, int K, int8_t * tab, float * ts) {
    float T[G / 4][16];
    for (int blk = 0; blk < K / G; blk++) {
        float m = 0;
        for (int g = 0; g < G / 4; g++) {
            const float * xg = x + blk * G + g * 4;
            for (int i = 0; i < 16; i++) T[g][i] = (i & 1 ? xg[0] : -xg[0]) + (i & 2 ? xg[1] : -xg[1]) + (i & 4 ? xg[2] : -xg[2]) + (i & 8 ? xg[3] : -xg[3]);
            m = fmaxf(m, T[g][15] > 0 ? T[g][15] : -T[g][15]) ; m = fmaxf(m, fabsf(xg[0]) + fabsf(xg[1]) + fabsf(xg[2]) + fabsf(xg[3]));   // max |T| = sum |x_j|
        }
        const float d = m / (127 / ACC), id = d ? 1 / d : 0; ts[blk] = d * 0.5f;
        for (int g = 0; g < G / 4; g++) for (int i = 0; i < 16; i++) tab[(blk * G / 4 + g) * 16 + i] = (int8_t) lrintf(T[g][i] * id);
    }
}
#define TB(B) (32 + (B) * (G / 8) * 16)   // bytes per tile-block
static inline __attribute__((always_inline)) void lut_body(const int B, const uint8_t * w, int nbk, const int8_t * tab, const float * ts, float * y, int t0, int t1) {
    const uint8x16_t m4 = vdupq_n_u8(15);
    for (int t = t0; t < t1; t++) {
        const uint8_t * p = w + (size_t) t * nbk * TB(B);
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int blk = 0; blk < nbk; blk++) {
            const float16x8_t s01 = vld1q_f16((const __fp16 *) p), s23 = vld1q_f16((const __fp16 *) p + 8); p += 32;
            const int8_t * tb = tab + (size_t) blk * (G / 4) * 16;
            int16x8_t hl = vdupq_n_s16(0), hh = hl;
            _Pragma("clang loop unroll(full)") for (int pl = 0; pl < B; pl++) {
                int16x8_t sl = vdupq_n_s16(0), sh = sl;
                int8x16_t part = vdupq_n_s8(0);
                _Pragma("clang loop unroll(full)") for (int j = 0; j < G / 8; j++) {
                    const uint8x16_t v = vld1q_u8(p); p += 16;
                    const int8x16_t l0 = vqtbl1q_s8(vld1q_s8(tb + 32 * j), vandq_u8(v, m4)), l1 = vqtbl1q_s8(vld1q_s8(tb + 32 * j + 16), vshrq_n_u8(v, 4));
                    if (ACC == 1) { sl = vaddq_s16(sl, vaddl_s8(vget_low_s8(l0), vget_low_s8(l1))); sh = vaddq_s16(sh, vaddl_high_s8(l0, l1)); continue; }
                    part = (2 * j) % ACC == 0 ? vaddq_s8(l0, l1) : vaddq_s8(part, vaddq_s8(l0, l1));   // ACC lookups summed in int8 (tables at +-127/ACC)
                    if ((2 * j + 2) % ACC == 0) { sl = vaddw_s8(sl, vget_low_s8(part)); sh = vaddw_high_s8(sh, part); }
                }
                hl = vaddq_s16(vaddq_s16(hl, hl), sl); hh = vaddq_s16(vaddq_s16(hh, hh), sh);   // Horner: planes stored B-1 first
            }
            const float tsb = ts[blk];
            a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hl))), vmulq_n_f32(vcvt_f32_f16(vget_low_f16(s01)), tsb));
            a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_high_s16(hl)), vmulq_n_f32(vcvt_high_f32_f16(s01), tsb));
            a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hh))), vmulq_n_f32(vcvt_f32_f16(vget_low_f16(s23)), tsb));
            a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_high_s16(hh)), vmulq_n_f32(vcvt_high_f32_f16(s23), tsb));
        }
        vst1q_f32(y + 16 * t, a0); vst1q_f32(y + 16 * t + 4, a1); vst1q_f32(y + 16 * t + 8, a2); vst1q_f32(y + 16 * t + 12, a3);
    }
}
__attribute__((noinline)) void lut2(const uint8_t * w, int nbk, const int8_t * tab, const float * ts, float * y, int t0, int t1) { lut_body(2, w, nbk, tab, ts, y, t0, t1); }
__attribute__((noinline)) void lut3(const uint8_t * w, int nbk, const int8_t * tab, const float * ts, float * y, int t0, int t1) { lut_body(3, w, nbk, tab, ts, y, t0, t1); }

#ifndef ASM_ONLY
static void pack_lut(int B, const uint8_t * q, const __fp16 * s, int N, int K, uint8_t * out) {   // q[r*K+k] in [0, 2^B), s[r*nbk+blk]
    const int nbk = K / G;
    for (int t = 0; t < N / 16; t++) for (int blk = 0; blk < nbk; blk++) {
        uint8_t * p = out + ((size_t) t * nbk + blk) * TB(B); __fp16 * d = (__fp16 *) p; p += 32;
        for (int l = 0; l < 16; l++) d[l] = s[(size_t)(16 * t + l) * nbk + blk];
        for (int pl = B - 1; pl >= 0; pl--) for (int j = 0; j < G / 8; j++) for (int l = 0; l < 16; l++, p++) {
            const uint8_t * qr = q + (size_t)(16 * t + l) * K + blk * G + j * 8; uint8_t lo = 0, hi = 0;
            for (int i = 0; i < 4; i++) { lo |= ((qr[i] >> pl) & 1) << i; hi |= ((qr[4 + i] >> pl) & 1) << i; }
            *p = lo | (hi << 4);
        }
    }
}
// ---- threads (persistent spinning workers, as tools/c3_k1v_bench.c)
typedef struct { int kind; const uint8_t * w; int nb; const bq8 * x; const int8_t * tab; const float * ts; float * y; int N; } Job;
static Job job; static atomic_int gen, done; static int NT = 1;
static void run_part(int id) {
    const int q = job.kind == 0 ? 4 : 16, g = job.N / q, a = (int)((long) g * id / NT), e = (int)((long) g * (id + 1) / NT);
    if (job.kind == 0) stock(job.w, job.nb, job.x, job.y, a * 4, e * 4);
    else (job.kind == 2 ? lut2 : lut3)(job.w, job.nb, job.tab, job.ts, job.y, a, e);
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
#define NTR 9
static int dcmp(const void * a, const void * b) { const double x = *(const double *) a, y = *(const double *) b; return (x > y) - (x < y); }
static double nmse(const float * y, const double * r, int N) { double e = 0, s = 0; for (int i = 0; i < N; i++) { e += (y[i] - r[i]) * (y[i] - r[i]); s += r[i] * r[i]; } return e / s; }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: lut <rows> <inputs> <threads> <copies> [reps]\n"); return 1; }
    const int N = atoi(argv[1]), K = atoi(argv[2]), ncopy = atoi(argv[4]), reps = argc > 5 ? atoi(argv[5]) : 20; NT = atoi(argv[3]);
    const int nb = K / QK, nbk = K / G; const double outl = getenv("LUT_OUTLIERS") ? atof(getenv("LUT_OUTLIERS")) : 0;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    pthread_t th[16]; for (int i = 1; i < NT; i++) pthread_create(&th[i], 0, worker, (void *)(long) i);
    // activations: gaussian; LUT_OUTLIERS=f puts f of the channels at 20x
    float * x = malloc(4 * K); for (int k = 0; k < K; k++) x[k] = (float)(gauss() * ((rnd() & 0xffffff) < outl * 16777216 ? 20 : 1));
    // Q4_0: q in [0,16), w = d (q - 8), d of either sign
    const size_t m4 = (size_t) N * nb * 18; uint8_t * w4 = malloc(m4); double * r4 = calloc(N, 8);
    for (int r = 0; r < N; r += 4) for (int b = 0; b < nb; b++) {
        bq4x4 * o = (bq4x4 *) w4 + (size_t)(r / 4) * nb + b; uint8_t q[4][32];
        for (int k = 0; k < 4; k++) { float d = 0.001f + 0.01f * (rnd() & 1023) / 1023.f; if (rnd() & 1) d = -d; o->d[k] = (__fp16) d;
            for (int j = 0; j < 32; j++) { q[k][j] = rnd() & 15; r4[r + k] += (double) o->d[k] * (q[k][j] - 8) * x[b * 32 + j]; } }
        for (int i = 0; i < 16; ++i) for (int e = 0; e < 4; e++) {   // ggml repack: byte = (q[j] | q[j+16] << 4) ^ 0x88, rows interleaved by 4 bytes
            const int k = i % 4, j = (i / 4) * 4 + e; o->qs[i * 4 + e] = (uint8_t)((q[k][j] | (q[k][j + 16] << 4)) ^ 0x88); }
    }
    // LUT weights, B = 2 and 3
    uint8_t * wl[4] = { 0 }; double * rl[4] = { 0 }; size_t ml[4] = { 0 };
    uint8_t * qq = malloc((size_t) N * K); __fp16 * ss = malloc(sizeof(__fp16) * (size_t) N * nbk);
    for (int B = 2; B <= 3; B++) {
        ml[B] = (size_t)(N / 16) * nbk * TB(B); wl[B] = malloc(ml[B]); rl[B] = calloc(N, 8);
        for (size_t i = 0; i < (size_t) N * K; i++) qq[i] = rnd() & ((1 << B) - 1);
        for (size_t i = 0; i < (size_t) N * nbk; i++) { float d = 0.001f + 0.01f * (rnd() & 1023) / 1023.f; if (rnd() & 1) d = -d; ss[i] = (__fp16) d; }
        for (int r = 0; r < N; r++) for (int k = 0; k < K; k++) rl[B][r] += (double) ss[(size_t) r * nbk + k / G] * (qq[(size_t) r * K + k] - ((1 << B) - 1) / 2.0) * x[k];
        pack_lut(B, qq, ss, N, K, wl[B]);
    }
    // correctness: LUT kernel against a double evaluation of the same int8 tables (only float rounding may differ)
    int8_t * tab = malloc((size_t) K * 4); float * ts = malloc(4 * nbk); build_tab(x, K, tab, ts);
    bq8 * xq = malloc(sizeof(bq8) * nb); quant_q8(x, xq, K);
    float * y = malloc(4 * N); double e4, el[4], ck[4] = { 0 };
    job = (Job){ 0, w4, nb, xq, 0, 0, y, N }; run_job(); e4 = nmse(y, r4, N);
    for (int B = 2; B <= 3; B++) {
        job = (Job){ B, wl[B], nbk, 0, tab, ts, y, N }; run_job(); el[B] = nmse(y, rl[B], N);
        // re-derive: unpack the packed file, evaluate table sums in double
        for (int r = 0; r < N; r += 97) {
            double acc = 0; const int t = r / 16, l = r % 16;
            for (int blk = 0; blk < nbk; blk++) {
                const uint8_t * p = wl[B] + ((size_t) t * nbk + blk) * TB(B); const double s = ((const __fp16 *) p)[l]; p += 32; long h = 0;
                for (int pl = B - 1; pl >= 0; pl--) { long sp = 0; for (int j = 0; j < G / 8; j++, p += 16) { const int g = blk * G / 4 + 2 * j; sp += tab[g * 16 + (p[l] & 15)] + tab[(g + 1) * 16 + (p[l] >> 4)]; } h = 2 * h + sp; }
                acc += s * ts[blk] * h;
            }
            const double d = fabs(acc - y[r]) / (fabs(acc) + 1e-30); if (d > ck[B]) ck[B] = d;
        }
    }
    printf("shape %dx%d G %d ACC %d threads %d copies %d | MB q4_0 %.2f lut2 %.2f lut3 %.2f | NMSE vs float64: q4_0x4(q8 acts) %.3e lut2 %.3e lut3 %.3e | lut vs own-table double max rel %.1e %.1e | outliers %.3f\n",
           N, K, G, ACC, NT, ncopy, m4 / 1048576.0, ml[2] / 1048576.0, ml[3] / 1048576.0, e4, el[2], el[3], ck[2], ck[3], outl);
    // timing: rings of copies (copies 1 = hot)
    uint8_t * a4 = malloc(m4 * ncopy), * a2 = malloc(ml[2] * ncopy), * a3 = malloc(ml[3] * ncopy);
    for (int c = 0; c < ncopy; c++) { memcpy(a4 + c * m4, w4, m4); memcpy(a2 + c * ml[2], wl[2], ml[2]); memcpy(a3 + c * ml[3], wl[3], ml[3]); }
    double best[5] = { 1e30, 1e30, 1e30, 1e30, 1e30 }, kt[2][NTR];   // kt: per-trial ratio lut/stock (interleaved, robust to contention)
    for (int trial = 0; trial < NTR; trial++) for (int v = 0; v < 5; v++) {   // 0 stock, 2 lut2, 3 lut3, 1 quant_q8, 4 build_tab
        const double t0 = now();
        for (int r = 0; r < reps; r++) {
            const int c = (trial * reps + r) % ncopy;
            if (v == 1) { quant_q8(x, xq, K); continue; }
            if (v == 4) { build_tab(x, K, tab, ts); continue; }
            job = v == 0 ? (Job){ 0, a4 + c * m4, nb, xq, 0, 0, y, N } : (Job){ v, (v == 2 ? a2 : a3) + c * ml[v], nbk, 0, tab, ts, y, N };
            run_job();
        }
        const double t = (now() - t0) / reps; if (t < best[v]) best[v] = t;
        static double ts0; if (v == 0) ts0 = t; if (v == 2 || v == 3) kt[v - 2][trial] = t / ts0;
    }
    for (int b = 0; b < 2; b++) qsort(kt[b], NTR, sizeof(double), dcmp);
    const double W = (double) N * K, st = best[0] / W;
    printf("  stock ns/w %.4f (%.1f GB/s) | lut2 ns/w %.4f k %.3f (%.1f GB/s) | lut3 ns/w %.4f k %.3f (%.1f GB/s) | prep us: q8 %.2f tab %.2f -> k with prep: lut2 %.3f lut3 %.3f | median per-trial k: lut2 %.3f lut3 %.3f\n",
           st * 1e9, m4 / best[0] / 1e9, best[2] / W * 1e9, best[2] / best[0], ml[2] / best[2] / 1e9, best[3] / W * 1e9, best[3] / best[0], ml[3] / best[3] / 1e9,
           best[1] * 1e6, best[4] * 1e6, (best[2] + best[4]) / (best[0] + best[1]), (best[3] + best[4]) / (best[0] + best[1]), kt[0][NTR / 2], kt[1][NTR / 2]);
    atomic_store(&gen, -1); for (int i = 1; i < NT; i++) pthread_join(th[i], 0);
    return 0;
}
#endif
