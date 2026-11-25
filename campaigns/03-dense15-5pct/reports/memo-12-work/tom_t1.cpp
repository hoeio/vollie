// T1 (Ebony r18 §4, Bonnie r34 §9): pool stall on the CPU-alone 22016 x 4096 Q4_0 gemv. *host*, relative only.
// arm "none": graph_compute back to back; "pre": a 200 us busy stall on the calling thread before each graph_compute
// (not counted in the graph time); "in": the stall inside the KleidiAI Q4_0 forward on thread 0 after its barrier
// (scratch-tree hack, env T1_STALL_US; this binary only reports). Prints per-graph time and voluntary switches per thread.
// usage: t1 <arm none|pre|in> <stall_us> <n_threads> <copies> <seconds>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <dirent.h>
#include <unistd.h>
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
static std::vector<std::pair<int, long>> vcsw() {   // (tid, voluntary switches) of every thread of this process
    std::vector<std::pair<int, long>> r; DIR * d = opendir("/proc/self/task"); if (!d) return r;
    while (dirent * e = readdir(d)) { if (e->d_name[0] == '.') continue; std::string p = std::string("/proc/self/task/") + e->d_name + "/status";
        FILE * f = fopen(p.c_str(), "r"); if (!f) continue; char l[256]; long v = -1;
        while (fgets(l, sizeof l, f)) if (!strncmp(l, "voluntary_ctxt_switches:", 24)) v = atol(l + 24);
        fclose(f); r.push_back({ atoi(e->d_name), v }); }
    closedir(d); std::sort(r.begin(), r.end()); return r;
}
int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: t1 arm stall_us nth copies secs\n"); return 2; }
    const std::string arm = argv[1]; const int stall = atoi(argv[2]), nth = atoi(argv[3]), NR = atoi(argv[4]); const double secs = atof(argv[5]);
    const int M = getenv("T1_M") ? atoi(getenv("T1_M")) : 22016, K = 4096;
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU); ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_cpu_set_n_threads(be, nth);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto ex = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t bt = ggml_backend_dev_buffer_type(dev);
    for (ggml_backend_buffer_type_t * p = ex ? ex(dev) : nullptr; p && *p; p++) if (strstr(ggml_backend_buft_name(*p), "KLEIDIAI")) bt = *p;
    ggml_init_params ip{ ggml_tensor_overhead() * (NR * 3 + 4) + ggml_graph_overhead() * NR, nullptr, true };
    ggml_context * cw = ggml_init(ip), * cx = ggml_init(ip), * cg = ggml_init(ip);
    std::vector<ggml_tensor *> w(NR); for (auto & t : w) t = ggml_new_tensor_2d(cw, GGML_TYPE_Q4_0, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(cx, GGML_TYPE_F32, K, 1);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, bt), bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_dev_buffer_type(dev));
    if (!bw || !bx) { fprintf(stderr, "alloc failed\n"); return 1; }
    std::mt19937 rng(11); std::vector<uint8_t> raw(ggml_nbytes(w[0]));
    for (size_t i = 0; i < raw.size(); i++) raw[i] = (uint8_t) rng();
    for (size_t b = 0; b < raw.size(); b += 18) { raw[b] = 0x66; raw[b + 1] = 0x2E; }   // block scale 0.1 (fp16), finite
    for (auto * t : w) ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
    std::vector<float> xf(K); std::normal_distribution<float> nd; for (auto & v : xf) v = nd(rng); ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * 4);
    std::vector<ggml_cgraph *> gf(NR); std::vector<ggml_tensor *> y(NR);
    ggml_gallocr_t ga = nullptr; std::vector<ggml_gallocr_t> gas;
    for (int i = 0; i < NR; i++) { gf[i] = ggml_new_graph(cg); y[i] = ggml_mul_mat(cg, w[i], x); ggml_build_forward_expand(gf[i], y[i]);
        ga = ggml_gallocr_new(ggml_backend_dev_buffer_type(dev)); ggml_gallocr_alloc_graph(ga, gf[i]); gas.push_back(ga); }
    auto spin = [](int u) { const auto t0 = clk::now(); while (us(t0, clk::now()) < u) {} };
    auto one = [&](int i) { if (arm == "pre") spin(stall); const auto a = clk::now(); ggml_backend_graph_compute(be, gf[i % NR]); return us(a, clk::now()); };
    for (auto t0 = clk::now(); us(t0, clk::now()) < 1e6;) for (int i = 0; i < NR; i++) one(i);   // warm 1 s
    std::vector<float> y0(M); ggml_backend_tensor_get(y[0], y0.data(), 0, M * 4);
    uint64_t h = 1469598103934665603ull; for (float v : y0) { uint32_t b; memcpy(&b, &v, 4); h = (h ^ b) * 1099511628211ull; }
    const auto s0 = vcsw(); std::vector<double> t; const auto t0 = clk::now(); long n = 0;
    while (us(t0, clk::now()) < secs * 1e6) t.push_back(one(n++));
    const auto s1 = vcsw();
    std::sort(t.begin(), t.end()); double mean = 0; for (double v : t) mean += v; mean /= t.size();
    const char * cm = getenv("GGML_KLEIDIAI_CHUNK_MULTIPLIER"); const char * ins = getenv("T1_STALL_US");
    printf("t1 M=%d arm=%s stall_us=%d in_op_stall_us=%s mult=%s nth=%d copies=%d buft=%s graphs=%zu graph_us_p10=%.1f p50=%.1f p90=%.1f mean=%.1f y0_hash=%016llx\n", M,
           arm.c_str(), stall, ins ? ins : "0", cm ? cm : "4(default)", nth, NR, ggml_backend_buft_name(bt), t.size(), t[t.size() / 10], t[t.size() / 2], t[t.size() * 9 / 10], mean, (unsigned long long) h);
    std::string vs; for (auto & p : s1) { long v0 = -1; for (auto & q : s0) if (q.first == p.first) v0 = q.second; if (v0 >= 0 && p.second - v0 > 0) { char b[64]; snprintf(b, sizeof b, " tid%d:%.3f", p.first, (p.second - v0) / (double) t.size()); vs += b; } }
    printf("t1 main_tid=%d vcsw_per_graph%s\n", (int) getpid(), vs.empty() ? " none" : vs.c_str());
    return 0;
}
