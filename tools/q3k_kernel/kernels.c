// Vollie R33 / round 10 audit: Q3_K expert gemv prototypes (host + NDK asm). Not in the runtime.
// Every kernel: 4 Q3_K rows (stride bx, stock layout unless named) x one Q8_K activation -> s[0..3].
// The float step per block and row is the stock one (sum = fma(y.d * fp16(x.d), (float) isum, sum)),
// so a kernel whose integer block sums are exact is bit-identical to ggml_vec_dot_q3_K_q8_K.
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; __fp16 d; } bq3;   // block_q3_K, 110 B
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } bq8;                        // block_q8_K, 292 B

// 16 6-bit scales -> int8 (minus 32); the same bytes as the stock utmp code (= ggml_q3k_scales_vec).
static inline int8x16_t scales_vec(const uint8_t * sc) {
    uint32_t aux[3];
    memcpy(aux, sc, 12);
    const uint32x2_t a01 = vld1_u32(aux);
    const uint32x4_t lo  = vcombine_u32(a01, a01);
    const uint32x4_t hi  = vdupq_n_u32(aux[2]);
    const int32x4_t  shl = { 0, 0, -4, -4 };
    const int32x4_t  shh = { 0, -2, -4, -6 };
    const uint32x4_t u = vorrq_u32(vandq_u32(vshlq_u32(lo, shl), vdupq_n_u32(0x0f0f0f0f)),
                                   vshlq_n_u32(vandq_u32(vshlq_u32(hi, shh), vdupq_n_u32(0x03030303)), 4));
    return vsubq_s8(vreinterpretq_s8_u32(u), vdupq_n_s8(32));
}

// 4x4 transpose of 32-bit words: out[w] = { r0 word w, r1 word w, r2 word w, r3 word w }.
static inline void tr4(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d, uint8x16_t out[4]) {
    const uint32x4_t t0 = vtrn1q_u32(vreinterpretq_u32_u8(a), vreinterpretq_u32_u8(b));
    const uint32x4_t t1 = vtrn2q_u32(vreinterpretq_u32_u8(a), vreinterpretq_u32_u8(b));
    const uint32x4_t t2 = vtrn1q_u32(vreinterpretq_u32_u8(c), vreinterpretq_u32_u8(d));
    const uint32x4_t t3 = vtrn2q_u32(vreinterpretq_u32_u8(c), vreinterpretq_u32_u8(d));
    out[0] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    out[1] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    out[2] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    out[3] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
}

// 4 rows' int8x16 scales -> S[m] = groups 4m..4m+3 x rows 0..3 (byte g*4 + r).
static inline void scales4(const int8x16_t s8[4], int8x16_t S[4]) {
    const int8x16x2_t z01 = vzipq_s8(s8[0], s8[1]);
    const int8x16x2_t z23 = vzipq_s8(s8[2], s8[3]);
    const int16x8x2_t lo = vzipq_s16(vreinterpretq_s16_s8(z01.val[0]), vreinterpretq_s16_s8(z23.val[0]));
    const int16x8x2_t hi = vzipq_s16(vreinterpretq_s16_s8(z01.val[1]), vreinterpretq_s16_s8(z23.val[1]));
    S[0] = vreinterpretq_s8_s16(lo.val[0]); S[1] = vreinterpretq_s8_s16(lo.val[1]);
    S[2] = vreinterpretq_s8_s16(hi.val[0]); S[3] = vreinterpretq_s8_s16(hi.val[1]);
}

static inline void finish4(float32x4_t * sumv, const bq3 * const x[4], int i, const bq8 * y, int32x4_t isum) {
    const float32x4_t xd = { (float) x[0][i].d, (float) x[1][i].d, (float) x[2][i].d, (float) x[3][i].d };
    const float32x4_t d  = vmulq_n_f32(xd, y[i].d);            // y.d * fp16(x.d) per row, as stock
    *sumv = vfmaq_f32(*sumv, d, vcvtq_f32_s32(isum));           // stock: fmadd per block and row
}

// One 16-weight group g for 4 rows in lane = row form: 4 by-element sdots into an accumulator
// that starts at -4 * bsum[g] (the "+4" value form q2 + 4h, see below), then times the 4 rows' scale.
#define GROUP(V0, V1, V2, V3, g) do {                                                   \
        const int8x16_t yv = vld1q_s8(y[i].qs + 16*(g));                                \
        int32x4_t a = vdupq_n_s32(nb4[g]);                                              \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V0), yv, 0);                         \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V1), yv, 1);                         \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V2), yv, 2);                         \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V3), yv, 3);                         \
        isum = vmlaq_s32(isum, a, sc32[g]);                                             \
    } while (0)

// q2 + 4h (0..7) for plane k from a qs vector Q and an hmask vector H (bit k = this plane's high bit).
static inline uint8x16_t unpack_k(uint8x16_t Q, uint8x16_t H, int k) {
    const uint8x16_t m3 = vdupq_n_u8(3), m4 = vdupq_n_u8(4);
    switch (k) {
    case 0:  return vbslq_u8(m3, Q, vandq_u8(vshlq_n_u8(H, 2), m4));
    case 1:  return vbslq_u8(m3, vshrq_n_u8(Q, 2), vandq_u8(vshlq_n_u8(H, 1), m4));
    case 2:  return vbslq_u8(m3, vshrq_n_u8(Q, 4), vandq_u8(H, m4));
    default: return vorrq_u8(vshrq_n_u8(Q, 6), vandq_u8(vshrq_n_u8(H, 1), m4));
    }
}

static inline void prep_scales(const bq3 * const x[4], int i, int32x4_t sc32[16]) {
    int8x16_t s8[4], S[4];
    _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) s8[r] = scales_vec(x[r][i].scales);
    scales4(s8, S);
    _Pragma("clang loop unroll(full)") for (int m = 0; m < 4; ++m) {
        const int16x8_t l = vmovl_s8(vget_low_s8(S[m])), h = vmovl_s8(vget_high_s8(S[m]));
        sc32[4*m + 0] = vmovl_s16(vget_low_s16(l)); sc32[4*m + 1] = vmovl_s16(vget_high_s16(l));
        sc32[4*m + 2] = vmovl_s16(vget_low_s16(h)); sc32[4*m + 3] = vmovl_s16(vget_high_s16(h));
    }
}

// (a) stock layout, 4 row pointers, interleave in registers (no repack, no copy).
void q3k_t4(int n, float * s, const void * vx, size_t bx, const void * vy) {
    const int nb = n / 256;
    const bq3 * const x[4] = { vx, (const char *) vx + bx, (const char *) vx + 2*bx, (const char *) vx + 3*bx };
    const bq8 * y = vy;
    float32x4_t sumv = vdupq_n_f32(0);
    for (int i = 0; i < nb; ++i) {
        int32x4_t sc32[16];
        prep_scales(x, i, sc32);
        int32_t nb4[16];
        _Pragma("clang loop unroll(full)") for (int g = 0; g < 16; ++g) nb4[g] = -4 * y[i].bsums[g];
        int32x4_t isum = vdupq_n_s32(0);
        uint8x16_t H[2][4];
        _Pragma("clang loop unroll(full)") for (int e = 0; e < 2; ++e)
            tr4(vld1q_u8(x[0][i].hmask + 16*e), vld1q_u8(x[1][i].hmask + 16*e), vld1q_u8(x[2][i].hmask + 16*e), vld1q_u8(x[3][i].hmask + 16*e), H[e]);
        _Pragma("clang loop unroll(full)") for (int hf = 0; hf < 2; ++hf) {
            _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; ++c) {                       // c = which 16 of the half's 32 qs bytes
                uint8x16_t Q[4];
                tr4(vld1q_u8(x[0][i].qs + 32*hf + 16*c), vld1q_u8(x[1][i].qs + 32*hf + 16*c),
                    vld1q_u8(x[2][i].qs + 32*hf + 16*c), vld1q_u8(x[3][i].qs + 32*hf + 16*c), Q);
                const uint8x16_t * h = H[c];
                uint8x16_t hh[4];
                if (hf) { for (int w = 0; w < 4; ++w) hh[w] = vshrq_n_u8(h[w], 4); h = hh; }
                _Pragma("clang loop unroll(full)") for (int k = 0; k < 4; ++k) {
                    const int g = 8*hf + 2*k + c;
                    GROUP(unpack_k(Q[0], h[0], k), unpack_k(Q[1], h[1], k), unpack_k(Q[2], h[2], k), unpack_k(Q[3], h[3], k), g);
                }
            }
        }
        finish4(&sumv, x, i, y, isum);
    }
    vst1q_f32(s, sumv);
}

// (a-rp) the same kernel reading a block_q3_Kx4 copy: per block of 4 rows, hmask[2][4] and qs[2][2][4]
// vectors already transposed, then the 4 rows' scales[12] and d (4*110 = 440 B, same bytes as 4 rows).
typedef struct { uint8_t hmask[128]; uint8_t qs[256]; uint8_t scales[4][12]; __fp16 d[4]; } bq3x4;
void q3k_repack4(const void * vx, size_t bx, int nb, bq3x4 * out) {
    const bq3 * const x[4] = { vx, (const char *) vx + bx, (const char *) vx + 2*bx, (const char *) vx + 3*bx };
    for (int i = 0; i < nb; ++i) {
        _Pragma("clang loop unroll(full)") for (int e = 0; e < 2; ++e) {
            uint8x16_t t[4];
            tr4(vld1q_u8(x[0][i].hmask + 16*e), vld1q_u8(x[1][i].hmask + 16*e), vld1q_u8(x[2][i].hmask + 16*e), vld1q_u8(x[3][i].hmask + 16*e), t);
            _Pragma("clang loop unroll(full)") for (int w = 0; w < 4; ++w) vst1q_u8(out[i].hmask + 64*e + 16*w, t[w]);
        }
        _Pragma("clang loop unroll(full)") for (int c = 0; c < 4; ++c) {
            uint8x16_t t[4];
            tr4(vld1q_u8(x[0][i].qs + 16*c), vld1q_u8(x[1][i].qs + 16*c), vld1q_u8(x[2][i].qs + 16*c), vld1q_u8(x[3][i].qs + 16*c), t);
            _Pragma("clang loop unroll(full)") for (int w = 0; w < 4; ++w) vst1q_u8(out[i].qs + 64*c + 16*w, t[w]);
        }
        _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) { memcpy(out[i].scales[r], x[r][i].scales, 12); out[i].d[r] = x[r][i].d; }
    }
}
void q3k_t4_rp(int n, float * s, const bq3x4 * xb, const void * vy) {
    const int nb = n / 256;
    const bq8 * y = vy;
    float32x4_t sumv = vdupq_n_f32(0);
    for (int i = 0; i < nb; ++i) {
        const bq3x4 * b = xb + i;
        int8x16_t s8[4], S[4];
        _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) s8[r] = scales_vec(b->scales[r]);
        scales4(s8, S);
        int32x4_t sc32[16];
        _Pragma("clang loop unroll(full)") for (int m = 0; m < 4; ++m) {
            const int16x8_t l = vmovl_s8(vget_low_s8(S[m])), h = vmovl_s8(vget_high_s8(S[m]));
            sc32[4*m + 0] = vmovl_s16(vget_low_s16(l)); sc32[4*m + 1] = vmovl_s16(vget_high_s16(l));
            sc32[4*m + 2] = vmovl_s16(vget_low_s16(h)); sc32[4*m + 3] = vmovl_s16(vget_high_s16(h));
        }
        int32_t nb4[16];
        _Pragma("clang loop unroll(full)") for (int g = 0; g < 16; ++g) nb4[g] = -4 * y[i].bsums[g];
        int32x4_t isum = vdupq_n_s32(0);
        _Pragma("clang loop unroll(full)") for (int hf = 0; hf < 2; ++hf) {
            _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; ++c) {
                uint8x16_t Q[4], h[4];
                _Pragma("clang loop unroll(full)") for (int w = 0; w < 4; ++w) {
                    Q[w] = vld1q_u8(b->qs + 64*(2*hf + c) + 16*w);
                    h[w] = vld1q_u8(b->hmask + 64*c + 16*w);
                    if (hf) h[w] = vshrq_n_u8(h[w], 4);
                }
                _Pragma("clang loop unroll(full)") for (int k = 0; k < 4; ++k) {
                    const int g = 8*hf + 2*k + c;
                    GROUP(unpack_k(Q[0], h[0], k), unpack_k(Q[1], h[1], k), unpack_k(Q[2], h[2], k), unpack_k(Q[3], h[3], k), g);
                }
            }
        }
        const float32x4_t xd = { (float) b->d[0], (float) b->d[1], (float) b->d[2], (float) b->d[3] };
        sumv = vfmaq_f32(sumv, vmulq_n_f32(xd, y[i].d), vcvtq_f32_s32(isum));
    }
    vst1q_f32(s, sumv);
}

// (b) compute-floor reference "Q3_K8": the 4 rows' values pre-expanded to int8 (q2 + 4h, 0..7) in the
// transposed order, the 6-bit scales and d kept as Q3_K. 4*256 + 4*12 + 4*2 B per 4 rows (2.53x the bytes).
typedef struct { uint8_t v[1024]; uint8_t scales[4][12]; __fp16 d[4]; } bq3x4_8;
void q3k8_expand4(const bq3x4 * in, int nb, bq3x4_8 * out) {
    for (int i = 0; i < nb; ++i) {
        _Pragma("clang loop unroll(full)") for (int hf = 0; hf < 2; ++hf) for (int c = 0; c < 2; ++c) for (int w = 0; w < 4; ++w) {
            const uint8x16_t Q = vld1q_u8(in[i].qs + 64*(2*hf + c) + 16*w);
            uint8x16_t h = vld1q_u8(in[i].hmask + 64*c + 16*w);
            if (hf) h = vshrq_n_u8(h, 4);
            _Pragma("clang loop unroll(full)") for (int k = 0; k < 4; ++k) vst1q_u8(out[i].v + 64*(8*hf + 2*k + c) + 16*w, unpack_k(Q, h, k));
        }
        memcpy(out[i].scales, in[i].scales, sizeof(in[i].scales));
        memcpy(out[i].d, in[i].d, sizeof(in[i].d));
    }
}
void q3k8_t4(int n, float * s, const bq3x4_8 * xb, const void * vy) {
    const int nb = n / 256;
    const bq8 * y = vy;
    float32x4_t sumv = vdupq_n_f32(0);
    for (int i = 0; i < nb; ++i) {
        const bq3x4_8 * b = xb + i;
        int8x16_t s8[4], S[4];
        _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) s8[r] = scales_vec(b->scales[r]);
        scales4(s8, S);
        int32x4_t sc32[16];
        _Pragma("clang loop unroll(full)") for (int m = 0; m < 4; ++m) {
            const int16x8_t l = vmovl_s8(vget_low_s8(S[m])), h = vmovl_s8(vget_high_s8(S[m]));
            sc32[4*m + 0] = vmovl_s16(vget_low_s16(l)); sc32[4*m + 1] = vmovl_s16(vget_high_s16(l));
            sc32[4*m + 2] = vmovl_s16(vget_low_s16(h)); sc32[4*m + 3] = vmovl_s16(vget_high_s16(h));
        }
        int32_t nb4[16];
        _Pragma("clang loop unroll(full)") for (int g = 0; g < 16; ++g) nb4[g] = -4 * y[i].bsums[g];
        int32x4_t isum = vdupq_n_s32(0);
        _Pragma("clang loop unroll(full)") for (int g = 0; g < 16; ++g) {
            const uint8_t * v = b->v + 64*g;
            GROUP(vld1q_u8(v), vld1q_u8(v + 16), vld1q_u8(v + 32), vld1q_u8(v + 48), g);
        }
        const float32x4_t xd = { (float) b->d[0], (float) b->d[1], (float) b->d[2], (float) b->d[3] };
        sumv = vfmaq_f32(sumv, vmulq_n_f32(xd, y[i].d), vcvtq_f32_s32(isum));
    }
    vst1q_f32(s, sumv);
}

// (c) minimal change: the shipped x4 ILP kernel's structure (per-row sdot + vpaddq tree, activation
// shared by 4 rows) with the "+4" unpack (q2 + 4h via bsl, bias -4*bsum folded into the group sums).
void q3k_x4c(int n, float * s, const void * vx, size_t bx, const void * vy) {
    const int nb = n / 256;
    const bq3 * const x[4] = { vx, (const char *) vx + bx, (const char *) vx + 2*bx, (const char *) vx + 3*bx };
    const bq8 * y = vy;
    float sum[4] = { 0, 0, 0, 0 };
    const int32x4_t z = vdupq_n_s32(0);
    for (int i = 0; i < nb; ++i) {
        const int8_t * q8 = y[i].qs;
        int8x16_t yv[16];
        _Pragma("clang loop unroll(full)") for (int g = 0; g < 16; ++g) yv[g] = vld1q_s8(q8 + 16*g);
        // -4 * bsums, 4 groups per vector (group order of the vpaddq tree below: g = 8hf + 2k + c)
        const int16x8_t b0 = vld1q_s16(y[i].bsums), b1 = vld1q_s16(y[i].bsums + 8);
        const int32x4_t nbv[4] = { vshlq_n_s32(vnegq_s32(vmovl_s16(vget_low_s16(b0))), 2), vshlq_n_s32(vnegq_s32(vmovl_s16(vget_high_s16(b0))), 2),
                                   vshlq_n_s32(vnegq_s32(vmovl_s16(vget_low_s16(b1))), 2), vshlq_n_s32(vnegq_s32(vmovl_s16(vget_high_s16(b1))), 2) };
        _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) {
            const bq3 * xr = &x[r][i];
            const int8x16_t s8 = scales_vec(xr->scales);
            const int16x8_t sl = vmovl_s8(vget_low_s8(s8)), sh = vmovl_s8(vget_high_s8(s8));
            const int32x4_t sc[4] = { vmovl_s16(vget_low_s16(sl)), vmovl_s16(vget_high_s16(sl)), vmovl_s16(vget_low_s16(sh)), vmovl_s16(vget_high_s16(sh)) };
            uint8x16_t H0 = vld1q_u8(xr->hmask), H1 = vld1q_u8(xr->hmask + 16);
            int32x4_t acc = z;
            _Pragma("clang loop unroll(full)") for (int hf = 0; hf < 2; ++hf) {
                const uint8x16_t Q0 = vld1q_u8(xr->qs + 32*hf), Q1 = vld1q_u8(xr->qs + 32*hf + 16);
                _Pragma("clang loop unroll(full)") for (int kk = 0; kk < 2; ++kk) {          // groups 8hf + 4kk .. +3 = planes 2kk, 2kk+1, c = 0, 1
                    const int k0 = 2*kk, k1 = 2*kk + 1, gb = 8*hf + 4*kk;
                    const int32x4_t p0 = vdotq_s32(z, vreinterpretq_s8_u8(unpack_k(Q0, H0, k0)), yv[gb + 0]);
                    const int32x4_t p1 = vdotq_s32(z, vreinterpretq_s8_u8(unpack_k(Q1, H1, k0)), yv[gb + 1]);
                    const int32x4_t p2 = vdotq_s32(z, vreinterpretq_s8_u8(unpack_k(Q0, H0, k1)), yv[gb + 2]);
                    const int32x4_t p3 = vdotq_s32(z, vreinterpretq_s8_u8(unpack_k(Q1, H1, k1)), yv[gb + 3]);
                    const int32x4_t p = vaddq_s32(vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3)), nbv[2*hf + kk]);
                    acc = vmlaq_s32(acc, p, sc[2*hf + kk]);
                }
                H0 = vshrq_n_u8(H0, 4); H1 = vshrq_n_u8(H1, 4);
            }
            sum[r] = __builtin_fmaf(y[i].d * (float) xr->d, (float) vaddvq_s32(acc), sum[r]);
        }
    }
    _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) s[r] = sum[r];
}

// (a2) as (a), but the high bit is not merged into the 2-bit value: each plane is masked in place
// (q2 * 4^k, k = 1, 2; ushr for k = 3) and the hmask bit j in place (h * 2^j; j = 7 reads as -128), each with
// its own by-element sdot accumulator. Integer scaling to 128 x the block sum (exact, |128 x sum| < 2^29.1):
// 128 * sum_g sc_g * (q2dot + 4 hdot - 4 bsum) = sum_g sc_g * (2^a_k accQ + 2^b_j accH), accQ starting at
// -4 * 4^k_scale * bsum; then >> 7 (exact) before the stock float step.
void q3k_t4h(int n, float * s, const void * vx, size_t bx, const void * vy) {
    const int nb = n / 256;
    const bq3 * const x[4] = { vx, (const char *) vx + bx, (const char *) vx + 2*bx, (const char *) vx + 3*bx };
    const bq8 * y = vy;
    float32x4_t sumv = vdupq_n_f32(0);
    for (int i = 0; i < nb; ++i) {
        int32x4_t sc32[16];
        prep_scales(x, i, sc32);
        int32x4_t isum = vdupq_n_s32(0);
        uint8x16_t H[2][4];
        _Pragma("clang loop unroll(full)") for (int e = 0; e < 2; ++e)
            tr4(vld1q_u8(x[0][i].hmask + 16*e), vld1q_u8(x[1][i].hmask + 16*e), vld1q_u8(x[2][i].hmask + 16*e), vld1q_u8(x[3][i].hmask + 16*e), H[e]);
        _Pragma("clang loop unroll(full)") for (int hf = 0; hf < 2; ++hf) {
            _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; ++c) {
                uint8x16_t Q[4];
                tr4(vld1q_u8(x[0][i].qs + 32*hf + 16*c), vld1q_u8(x[1][i].qs + 32*hf + 16*c),
                    vld1q_u8(x[2][i].qs + 32*hf + 16*c), vld1q_u8(x[3][i].qs + 32*hf + 16*c), Q);
                _Pragma("clang loop unroll(full)") for (int k = 0; k < 4; ++k) {
                    const int g = 8*hf + 2*k + c, j = 4*hf + k;
                    const int lq = k == 1 ? 2 : k == 2 ? 4 : 0;           // log2 of the in-place q2 factor
                    const int8x16_t yv = vld1q_s8(y[i].qs + 16*g);
                    int32x4_t aq = vdupq_n_s32(-4 * (1 << lq) * y[i].bsums[g]);
                    int32x4_t ah = vdupq_n_s32(0);
                    const uint8x16_t mq = vdupq_n_u8(3 << (2*k)), mh = vdupq_n_u8(1 << j);
#define QH(w) do {                                                                                          \
        const uint8x16_t q = k == 3 ? vshrq_n_u8(Q[w], 6) : vandq_u8(Q[w], mq);                              \
        aq = vdotq_laneq_s32(aq, vreinterpretq_s8_u8(q), yv, w);                                             \
        ah = vdotq_laneq_s32(ah, vreinterpretq_s8_u8(vandq_u8(H[c][w], mh)), yv, w);                         \
    } while (0)
                    QH(0); QH(1); QH(2); QH(3);
#undef QH
                    // 128 / 2^lq and 512 / 2^j (j = 7: the byte is -128, so -4)
                    const int32x4_t t = vaddq_s32(vshlq_s32(aq, vdupq_n_s32(7 - lq)),
                                                  j == 7 ? vmulq_n_s32(ah, -4) : vshlq_s32(ah, vdupq_n_s32(9 - j)));
                    isum = vmlaq_s32(isum, t, sc32[g]);
                }
            }
        }
        finish4(&sumv, x, i, y, vshrq_n_s32(isum, 7));
    }
    vst1q_f32(s, sumv);
}

// (a3) = (a) tuned: the 4 rows' 6-bit scales unpacked together (lanes = rows, one tbl to group-major),
// planes 1 and 2 masked in place (4 q2 / 16 q2, the high bit placed above them; the group sum is shifted
// back exactly before the scale), the hmask shifted straight to its target bit (no per-half >> 4 copy).
static inline void prep_scales4(const bq3 * const x[4], int i, int32x4_t sc32[16]) {
    uint32_t aux[12];
    _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) memcpy(aux + 3*r, x[r][i].scales, 12);
    const uint32x4x3_t a = vld3q_u32(aux);                       // a.val[w] = aux[w] of rows 0..3
    const uint32x4_t k2 = vdupq_n_u32(0x0f0f0f0f), k1 = vdupq_n_u32(0x03030303);
    const uint32x4_t u[4] = {
        vorrq_u32(vandq_u32(a.val[0], k2),                    vshlq_n_u32(vandq_u32(a.val[2], k1), 4)),
        vorrq_u32(vandq_u32(a.val[1], k2),                    vshlq_n_u32(vandq_u32(vshrq_n_u32(a.val[2], 2), k1), 4)),
        vorrq_u32(vandq_u32(vshrq_n_u32(a.val[0], 4), k2),    vshlq_n_u32(vandq_u32(vshrq_n_u32(a.val[2], 4), k1), 4)),
        vorrq_u32(vandq_u32(vshrq_n_u32(a.val[1], 4), k2),    vshlq_n_u32(vandq_u32(vshrq_n_u32(a.val[2], 6), k1), 4)),
    };
    const uint8x16_t tix = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };
    _Pragma("clang loop unroll(full)") for (int m = 0; m < 4; ++m) {
        const int8x16_t S = vsubq_s8(vreinterpretq_s8_u8(vqtbl1q_u8(vreinterpretq_u8_u32(u[m]), tix)), vdupq_n_s8(32));
        const int16x8_t l = vmovl_s8(vget_low_s8(S)), h = vmovl_s8(vget_high_s8(S));
        sc32[4*m + 0] = vmovl_s16(vget_low_s16(l)); sc32[4*m + 1] = vmovl_s16(vget_high_s16(l));
        sc32[4*m + 2] = vmovl_s16(vget_low_s16(h)); sc32[4*m + 3] = vmovl_s16(vget_high_s16(h));
    }
}

// shift so that hmask bit `src` lands on bit `dst`
#define HSH(H, src, dst) vshlq_u8(H, vdupq_n_s8((dst) - (src)))   // folds to one shl/ushr (or nothing) after unrolling

void q3k_t4b(int n, float * s, const void * vx, size_t bx, const void * vy) {
    const int nb = n / 256;
    const bq3 * const x[4] = { vx, (const char *) vx + bx, (const char *) vx + 2*bx, (const char *) vx + 3*bx };
    const bq8 * y = vy;
    float32x4_t sumv = vdupq_n_f32(0);
    for (int i = 0; i < nb; ++i) {
        int32x4_t sc32[16];
        prep_scales4(x, i, sc32);
        int32x4_t isum = vdupq_n_s32(0);
        uint8x16_t H[2][4];
        _Pragma("clang loop unroll(full)") for (int e = 0; e < 2; ++e)
            tr4(vld1q_u8(x[0][i].hmask + 16*e), vld1q_u8(x[1][i].hmask + 16*e), vld1q_u8(x[2][i].hmask + 16*e), vld1q_u8(x[3][i].hmask + 16*e), H[e]);
        _Pragma("clang loop unroll(full)") for (int hf = 0; hf < 2; ++hf) {
            _Pragma("clang loop unroll(full)") for (int c = 0; c < 2; ++c) {
                uint8x16_t Q[4];
                tr4(vld1q_u8(x[0][i].qs + 32*hf + 16*c), vld1q_u8(x[1][i].qs + 32*hf + 16*c),
                    vld1q_u8(x[2][i].qs + 32*hf + 16*c), vld1q_u8(x[3][i].qs + 32*hf + 16*c), Q);
#define V3(w, k) (k == 0 ? vorrq_u8(vandq_u8(Q[w], vdupq_n_u8(0x03)), vandq_u8(HSH(H[c][w], 4*hf + 0, 2), vdupq_n_u8(0x04))) : \
                  k == 1 ? vorrq_u8(vandq_u8(Q[w], vdupq_n_u8(0x0c)), vandq_u8(HSH(H[c][w], 4*hf + 1, 4), vdupq_n_u8(0x10))) : \
                  k == 2 ? vorrq_u8(vandq_u8(Q[w], vdupq_n_u8(0x30)), vandq_u8(HSH(H[c][w], 4*hf + 2, 6), vdupq_n_u8(0x40))) : \
                           vorrq_u8(vshrq_n_u8(Q[w], 6),              vandq_u8(HSH(H[c][w], 4*hf + 3, 2), vdupq_n_u8(0x04))))
#define G3(k) do {                                                                                   \
        const int g = 8*hf + 2*(k) + c, lf = (k) == 1 ? 2 : (k) == 2 ? 4 : 0;                         \
        const int8x16_t yv = vld1q_s8(y[i].qs + 16*g);                                               \
        int32x4_t a = vdupq_n_s32(-(4 << lf) * y[i].bsums[g]);                                       \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V3(0, k)), yv, 0);                                \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V3(1, k)), yv, 1);                                \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V3(2, k)), yv, 2);                                \
        a = vdotq_laneq_s32(a, vreinterpretq_s8_u8(V3(3, k)), yv, 3);                                \
        if (lf) a = vshlq_s32(a, vdupq_n_s32(-lf));      /* exact: a = 4^k * the group sum */          \
        isum = vmlaq_s32(isum, a, sc32[g]);                                                          \
    } while (0)
                G3(0); G3(1); G3(2); G3(3);
#undef G3
#undef V3
            }
        }
        finish4(&sumv, x, i, y, isum);
    }
    vst1q_f32(s, sumv);
}
