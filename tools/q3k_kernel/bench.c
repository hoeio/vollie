// Vollie R33 / round 10 audit: host microbench + numerics check for the Q3_K expert gemv prototypes
// (kernels.c) against the build-888 ggml kernels. Usage: bench <slot.bin> (gate, up, down of one expert).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <arm_neon.h>

typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; __fp16 d; } bq3;
typedef struct { uint8_t hmask[128]; uint8_t qs[256]; uint8_t scales[4][12]; __fp16 d[4]; } bq3x4;
typedef struct { uint8_t v[1024]; uint8_t scales[4][12]; __fp16 d[4]; } bq3x4_8;

// build-888 (lib)
void ggml_cpu_init(void);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void quantize_row_q4_0_ref(const float * x, void * y, int64_t k);
void ggml_vec_dot_q3_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K_x4(int n, float * s, const void * vx, size_t bx, const void * vy);
void ggml_vec_dot_q4_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_gemv_q4_0_4x4_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_cpu_head2_q4_0_pack4(void * dst, const void * src, int64_t nb, int il);
extern int ggml_cpu_q3k_ilp, ggml_cpu_q3k_vec;
// kernels.c
void q3k_t4(int n, float * s, const void * vx, size_t bx, const void * vy);
void q3k_repack4(const void * vx, size_t bx, int nb, bq3x4 * out);
void q3k_t4_rp(int n, float * s, const bq3x4 * xb, const void * vy);
void q3k8_expand4(const bq3x4 * in, int nb, bq3x4_8 * out);
void q3k8_t4(int n, float * s, const bq3x4_8 * xb, const void * vy);
void q3k_x4c(int n, float * s, const void * vx, size_t bx, const void * vy);
void q3k_t4b(int n, float * s, const void * vx, size_t bx, const void * vy);
void q3k_t4h(int n, float * s, const void * vx, size_t bx, const void * vy);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec*1e9 + t.tv_nsec; }
static volatile float sink;

// ns per row: best of 21 trials, each `reps` passes over all rows (cache-hot: one tensor is 338 KB)
#define TIME(label, rows, BODY) do {                                              \
        double best = 1e30; const int reps = 40;                                  \
        for (int t = 0; t < 21; ++t) {                                            \
            double t0 = now();                                                    \
            for (int p = 0; p < reps; ++p) { BODY; }                              \
            double dt = (now() - t0) / reps / (rows);                             \
            if (dt < best) best = dt;                                             \
        }                                                                         \
        sink += out[0];                                                           \
        printf("%-34s %8.2f ns/row\n", label, best);                              \
        res = best;                                                               \
    } while (0)

static int cmp(const char * label, const float * a, const float * ref, int n) {
    double maxa = 0, maxr = 0; int same = 0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double) a[i] - ref[i]);
        if (d > maxa) maxa = d;
        if (ref[i] != 0 && d / fabs(ref[i]) > maxr) maxr = d / fabs(ref[i]);
        same += memcmp(&a[i], &ref[i], 4) == 0;
    }
    printf("  check %-26s max abs %.3g  max rel %.3g  bit-identical %d/%d\n", label, maxa, maxr, same, n);
    return same == n;
}

int main(int argc, char ** argv) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    ggml_cpu_init();
    ggml_cpu_q3k_vec = 1; ggml_cpu_q3k_ilp = 1; // the device defaults
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    static uint8_t slot[3*337920];
    if (fread(slot, 1, sizeof slot, f) != sizeof slot) { fprintf(stderr, "short slot file\n"); return 1; }
    fclose(f);
    // tensor t: rows x K (gate, up: 512 x 1536; down: 1536 x 512)
    const int rows[3] = { 512, 512, 1536 }, K[3] = { 1536, 1536, 512 };
    const char * name[3] = { "gate", "up", "down" };
    int ok = 1;
    double slot_ns[10] = { 0 };
    srand(1);
    for (int t = 0; t < 3; ++t) {
        const int R = rows[t], n = K[t], nb = n/256;
        const size_t bx = (size_t) nb * 110;
        const uint8_t * W = slot + t*337920;
        float * act = malloc(n * sizeof(float));
        for (int i = 0; i < n; ++i) { float u = (rand() + 1.0f) / RAND_MAX, v = (float) rand() / RAND_MAX; act[i] = sqrtf(-2*logf(u)) * cosf(6.2831853f*v); }
        uint8_t * yk = malloc(nb * 292), * y0 = malloc(n/32 * 34);
        quantize_row_q8_K(act, yk, n);
        quantize_row_q8_0(act, y0, n);
        float * ref = malloc(R*4), * out = malloc(R*4);
        double res;
        printf("== %s: %d rows x %d (Q3_K %zu B/row)\n", name[t], R, n, bx);
        // reference: the stock per-row dot
        for (int r = 0; r < R; ++r) ggml_vec_dot_q3_K_q8_K(n, &ref[r], 0, W + r*bx, 0, yk, 0, 1);
        TIME("q3_K vec_dot (stock, 1 row)", R, for (int r = 0; r < R; ++r) ggml_vec_dot_q3_K_q8_K(n, &out[r], 0, W + r*bx, 0, yk, 0, 1));
        slot_ns[0] += res*R;
        TIME("q3_K x4 ILP (device path)", R, for (int r = 0; r < R; r += 4) ggml_vec_dot_q3_K_q8_K_x4(n, &out[r], W + r*bx, bx, yk));
        ok &= cmp("x4 ILP", out, ref, R); slot_ns[1] += res*R;
        TIME("(c) x4 + bsl unpack", R, for (int r = 0; r < R; r += 4) q3k_x4c(n, &out[r], W + r*bx, bx, yk));
        ok &= cmp("(c)", out, ref, R); slot_ns[2] += res*R;
        TIME("(a) t4 register interleave", R, for (int r = 0; r < R; r += 4) q3k_t4(n, &out[r], W + r*bx, bx, yk));
        ok &= cmp("(a)", out, ref, R); slot_ns[3] += res*R;
        TIME("(a2) t4, h as its own sdot", R, for (int r = 0; r < R; r += 4) q3k_t4h(n, &out[r], W + r*bx, bx, yk));
        ok &= cmp("(a2)", out, ref, R); slot_ns[8] += res*R;
        TIME("(a3) t4 tuned", R, for (int r = 0; r < R; r += 4) q3k_t4b(n, &out[r], W + r*bx, bx, yk));
        ok &= cmp("(a3)", out, ref, R); slot_ns[9] += res*R;
        bq3x4 * rp = malloc((size_t) R/4 * nb * sizeof(bq3x4));
        TIME("  repack Q3_Kx4 (cost)", R, for (int r = 0; r < R; r += 4) q3k_repack4(W + r*bx, bx, nb, rp + (size_t) r/4*nb));
        slot_ns[4] += res*R;
        TIME("(a-rp) t4 on repacked copy", R, for (int r = 0; r < R; r += 4) q3k_t4_rp(n, &out[r], rp + (size_t) r/4*nb, yk));
        ok &= cmp("(a-rp)", out, ref, R); slot_ns[5] += res*R;
        bq3x4_8 * x8 = malloc((size_t) R/4 * nb * sizeof(bq3x4_8));
        q3k8_expand4(rp, R/4*nb, x8);
        TIME("(b) Q3_K8 int8 pre-expanded", R, for (int r = 0; r < R; r += 4) q3k8_t4(n, &out[r], x8 + (size_t) r/4*nb, yk));
        ok &= cmp("(b)", out, ref, R); slot_ns[6] += res*R;
        // Q4_0 references on the same shape (random weights, timing only)
        float * wf = malloc((size_t) R*n*sizeof(float));
        for (size_t i = 0; i < (size_t) R*n; ++i) wf[i] = (float) rand() / RAND_MAX - 0.5f;
        const size_t b40 = n/32*18;
        uint8_t * q40 = malloc(R*b40), * q4x4 = malloc(R*b40);
        for (int r = 0; r < R; ++r) quantize_row_q4_0_ref(wf + (size_t) r*n, q40 + r*b40, n);
        for (int r = 0; r < R; r += 4) ggml_cpu_head2_q4_0_pack4(q4x4 + r*b40, q40 + r*b40, n/32, 4);
        TIME("q4_0 vec_dot (1 row)", R, for (int r = 0; r < R; ++r) ggml_vec_dot_q4_0_q8_0(n, &out[r], 0, q40 + r*b40, 0, y0, 0, 1));
        TIME("q4_0x4 gemv (repacked, sdot)", R, ggml_gemv_q4_0_4x4_q8_0(n, out, 0, q4x4, y0, 1, R));
        slot_ns[7] += res*R;
        uint8_t * q6 = malloc((size_t) R*nb*210);
        for (size_t i = 0; i < (size_t) R*nb*210; ++i) q6[i] = rand();
        for (size_t b = 0; b < (size_t) R/8*nb; ++b) for (int j = 0; j < 8; ++j) ((__fp16 *) (q6 + b*1680))[j] = 0.01f;
        TIME("q6_Kx8 gemv (repacked, 8x4)", R, ggml_gemv_q6_K_8x4_q8_K(n, out, 0, q6, yk, 1, R));
        free(act); free(yk); free(y0); free(ref); free(out); free(rp); free(x8); free(wf); free(q40); free(q4x4); free(q6);
    }
    const char * sn[10] = { "stock vec_dot", "x4 ILP (device)", "(c)", "(a) t4", "repack cost", "(a-rp)", "(b) Q3_K8", "q4_0x4 gemv", "(a2) t4h", "(a3) t4 tuned" };
    printf("== one slot (gate+up+down, 2,560 rows, 1 thread, cache-hot)\n");
    for (int v = 0; v < 10; ++v) printf("%-18s %8.1f us  (%.2fx device path)\n", sn[v], slot_ns[v]/1e3, slot_ns[v]/slot_ns[1]);
    printf(ok ? "ALL CHECKS BIT-IDENTICAL\n" : "CHECK FAILED: a kernel differs from ggml_vec_dot_q3_K_q8_K\n");
    return ok ? 0 : 1;
}
