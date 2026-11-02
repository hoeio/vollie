// Vollie R23 check: ggml_vec_dot_q3_K_q8_K_x4 (default, GGML_Q3K_ILP, GGML_Q3K_PF) is bit-identical to
// four ggml_vec_dot_q3_K_q8_K calls on random Q3_K rows; also times the variants (1 thread, cache-hot).
// build+run: sh lab/checks/q3k_x4_check.sh
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
void ggml_vec_dot_q3_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K_x4(int n, float * s, const void * vx, size_t bx, const void * vy);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
extern int ggml_cpu_q3k_ilp, ggml_cpu_q3k_pf, ggml_cpu_q3k_vec;
#define QK 256
#define BQ3 110            // sizeof(block_q3_K): hmask 32, qs 64, scales 12, d fp16
#define BQ8 (4 + QK + 32)  // sizeof(block_q8_K): d f32, qs 256, bsums 16 x int16
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
int main(void) {
    const int n = getenv("N") ? atoi(getenv("N")) : 1536, nb = n/QK, rows = 512;
    const size_t bx = (size_t) nb*BQ3;
    uint8_t * w = malloc(bx*rows + 4*bx); // + 4 rows: the prefetch of rows past the end stays in bounds
    srand(1);
    for (size_t i = 0; i < bx*(rows + 4); i++) w[i] = rand();
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) { // sane fp16 d: 0x2000-0x27ff (~0.008-0.03)
        uint8_t * d = w + r*bx + b*BQ3 + 108; d[0] = rand(); d[1] = 0x20 | (rand() & 7);
    }
    float * xf = malloc(n*sizeof(float)); for (int i = 0; i < n; i++) xf[i] = (rand()/(float) RAND_MAX - 0.5f)*4;
    uint8_t * q8 = malloc(nb*BQ8); quantize_row_q8_K(xf, q8, n);
    float ref[512], out[512]; if (n % QK || n > 4096) return 2;
    for (int r = 0; r < rows; r++) ggml_vec_dot_q3_K_q8_K(n, &ref[r], 0, w + r*bx, 0, q8, 0, 1);
    int bad = 0;
    const char * name[4] = { "x4", "x4+pf", "x4+ilp", "x4+ilp+pf" };
    double us[4] = { 1e9, 1e9, 1e9, 1e9 };
    for (int round = 0; round < 7; round++) { // interleaved rounds, best of 7 per variant
        for (int v = 0; v < 4; v++) {
            ggml_cpu_q3k_pf = v & 1; ggml_cpu_q3k_ilp = v >> 1;
            for (int r = 0; r < rows; r += 4) ggml_vec_dot_q3_K_q8_K_x4(n, &out[r], w + r*bx, bx, q8);
            for (int r = 0; r < rows; r++) bad += memcmp(&out[r], &ref[r], 4) != 0;
            double t0 = now();
            for (int it = 0; it < 500; it++) for (int r = 0; r < rows; r += 4) ggml_vec_dot_q3_K_q8_K_x4(n, &out[r], w + r*bx, bx, q8);
            const double t = (now() - t0)/500*1e6;
            if (t < us[v]) us[v] = t;
        }
    }
    for (int v = 0; v < 4; v++) {
        printf("%-10s %.2f us per 512 x %d matrix (%.3fx of x4), 1 thread, cache-hot\n", name[v], us[v], n, us[v]/us[0]);
    }
    printf("rows x variants x rounds differing from vec_dot: %d\n", bad);
    printf(bad ? "FAIL\n" : "PASS: bit-identical\n");
    return bad != 0;
}
