// mc3t: host (PoCL) micro-timing of upstream's kernel_gemv_noshuffle_q4_0_f32 (1 column) against
// kernel_gemv_noshuffle_q4_0_f32_mc3 (n = 2..4), with the dispatch of ggml_cl_mul_mat_q4_0_f32_adreno at 620d2225c.
// Also: two variant kernels (file k_var.cl) to split the mc3 cost. Host numbers are relative only.
// Usage: mc3t <kernel.cl> <opts> [M K reps]
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>

#define CK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%s:%d err %d\n", __FILE__, __LINE__, e_); exit(1); } } while (0)

static float h2f(uint16_t h) { // IEEE half -> float
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = std::ldexp((float) m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = std::ldexp((float)(m | 1024), (int) e - 25);
    return s ? -v : v;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mc3t k.cl opts [M K reps]\n"); return 2; }
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf(); std::string src = ss.str();
    const std::string opts = argv[2];
    const int M = argc > 3 ? atoi(argv[3]) : 11008, K = argc > 4 ? atoi(argv[4]) : 4096, reps = argc > 5 ? atoi(argv[5]) : 15;
    cl_platform_id pl; cl_device_id dev; CK(clGetPlatformIDs(1, &pl, nullptr)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr));
    cl_int e; cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CK(e);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &e); CK(e);
    const char * s = src.c_str(); cl_program prg = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e); CK(e);
    if (clBuildProgram(prg, 1, &dev, opts.c_str(), nullptr, nullptr)) {
        size_t n; clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n); std::string lg(n, 0);
        clGetProgramBuildInfo(prg, dev, CL_PROGRAM_BUILD_LOG, n, &lg[0], nullptr); fprintf(stderr, "build failed:\n%s\n", lg.c_str()); return 1;
    }
    // weights in the noshuffle layout the kernels read: q as uint per (row pair, block, j), d as half2 per (row pair, block)
    const int NB = K / 32, LS = M / 2; std::mt19937 rng(7);
    std::vector<uint32_t> qv((size_t) M * K / 8); for (auto & x : qv) x = rng();
    std::vector<uint16_t> dv((size_t) M * NB); std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (auto & x : dv) { float v = ud(rng); uint32_t b; memcpy(&b, &v, 4); x = (uint16_t)(((b >> 16) & 0x8000) | ((((b >> 23) & 255) - 112) << 10) | ((b >> 13) & 1023)); }
    std::vector<float> xs((size_t) K * 4); std::normal_distribution<float> nd; for (auto & x : xs) x = nd(rng);
    cl_mem qb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qv.size() * 4, qv.data(), &e); CK(e);
    cl_mem db = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dv.size() * 2, dv.data(), &e); CK(e);
    cl_mem xb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xs.size() * 4, xs.data(), &e); CK(e);
    cl_mem yb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) M * 4 * 4, nullptr, &e); CK(e);
    cl_image_format fq = { CL_R, CL_UNSIGNED_INT32 }; cl_image_desc ds{}; ds.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; ds.image_width = (size_t) M * K / 8; ds.buffer = qb;
    cl_mem qimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fq, &ds, nullptr, &e); CK(e);
    cl_mem bimg[5] = {};
    for (int n = 1; n <= 4; n++) { cl_image_format fb = { CL_RGBA, CL_FLOAT }; cl_image_desc d2{}; d2.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; d2.image_width = (size_t) K * n / 4; d2.buffer = xb;
        bimg[n] = clCreateImage(ctx, CL_MEM_READ_ONLY, &fb, &d2, nullptr, &e); CK(e); }
    cl_mem img1[4] = {};
    for (int c = 0; c < 4; c++) { cl_mem b1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t) K * 4, xs.data() + (size_t) c * K, &e); CK(e);
        cl_image_format fb = { CL_RGBA, CL_FLOAT }; cl_image_desc d2{}; d2.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; d2.image_width = (size_t) K / 4; d2.buffer = b1;
        img1[c] = clCreateImage(ctx, CL_MEM_READ_ONLY, &fb, &d2, nullptr, &e); CK(e); }
    // float64 reference of column c
    auto ref = [&](int c) { std::vector<double> r(M, 0.0);
        for (int p = 0; p < LS; p++) for (int k = 0; k < NB; k++) {
            const double d0 = h2f(dv[(size_t) p * 2 + (size_t) k * LS * 2]), d1 = h2f(dv[(size_t) p * 2 + 1 + (size_t) k * LS * 2]);
            for (int j = 0; j < 8; j++) { const uint32_t u = qv[(size_t) p + (size_t) k * 4 * M + (size_t) LS * j];
                for (int i = 0; i < 4; i++) { const double xv = xs[(size_t) c * K + k * 32 + j * 4 + i];
                    r[2 * p] += (double)((int)((u >> (4 * i)) & 15) - 8) * d0 * xv; r[2 * p + 1] += (double)((int)((u >> (16 + 4 * i)) & 15) - 8) * d1 * xv; } } }
        return r; };
    struct KR { const char * name; bool mc; int nsg; };
    std::vector<KR> ks = { { "kernel_gemv_noshuffle_q4_0_f32", false, 4 }, { "kernel_gemv_noshuffle_q4_0_f32_mc3", true, 4 } };
    for (const char * vn : { "var_mc_fixed2" }) {   // present only in k_var.cl
        cl_kernel t = clCreateKernel(prg, vn, &e); if (t) { clReleaseKernel(t); ks.push_back({ vn, true, 4 }); }
    }
    std::vector<std::vector<double>> R; for (int c = 0; c < 4; c++) R.push_back(ref(c));
    double rms0 = 0; for (double v : R[0]) rms0 += v * v; rms0 = std::sqrt(rms0 / M);
    std::vector<float> g1[4];   // general kernel output per column, for the bitwise check
    printf("mc3t M=%d K=%d reps=%d opts=%s\n", M, K, reps, opts.c_str());
    for (auto & kr : ks) {
        cl_kernel k = clCreateKernel(prg, kr.name, &e); CK(e);
        size_t pm = 0, lm = 0; clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof pm, &pm, nullptr); clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof lm, &lm, nullptr);
        for (int n = kr.mc ? 2 : 1; n <= 4; n++) {
            if (!strcmp(kr.name, "var_mc_fixed2") && n != 2) continue;
            auto launch = [&](int col, cl_event * ev) {
                const uint64_t z = 0; int a = 0;
                if (kr.mc) {
                    CK(clSetKernelArg(k, a++, sizeof(cl_mem), &qimg)); CK(clSetKernelArg(k, a++, sizeof(cl_mem), &db)); CK(clSetKernelArg(k, a++, sizeof(cl_mem), &bimg[n]));
                    CK(clSetKernelArg(k, a++, sizeof(cl_mem), &yb)); CK(clSetKernelArg(k, a++, 8, &z)); CK(clSetKernelArg(k, a++, 4, &K)); CK(clSetKernelArg(k, a++, 4, &M)); CK(clSetKernelArg(k, a++, 4, &n));
                } else {   // one column: column col of the 4-column buffer through a sub-buffer image, as ggml's ne1 == 1 path
                    const int one = 1;
                    CK(clSetKernelArg(k, a++, sizeof(cl_mem), &qimg)); CK(clSetKernelArg(k, a++, sizeof(cl_mem), &db));
                    cl_mem im = img1[col];   // PoCL refuses an image on a sub-buffer (-38): one buffer per column
                    const uint64_t off = (uint64_t) col * M * 4;
                    CK(clSetKernelArg(k, a++, sizeof(cl_mem), &im)); CK(clSetKernelArg(k, a++, 8, &z)); CK(clSetKernelArg(k, a++, sizeof(cl_mem), &yb)); CK(clSetKernelArg(k, a++, 8, &off));
                    CK(clSetKernelArg(k, a++, 4, &K)); CK(clSetKernelArg(k, a++, 4, &M)); CK(clSetKernelArg(k, a++, 4, &one)); CK(clSetKernelArg(k, a++, 4, &K));
                    CK(clSetKernelArg(k, a++, 4, &one)); CK(clSetKernelArg(k, a++, 4, &M)); CK(clSetKernelArg(k, a++, 4, &one)); CK(clSetKernelArg(k, a++, 4, &one)); CK(clSetKernelArg(k, a++, 4, &one));
                }
                const size_t g[3] = { (size_t)((M / 2 + 63) / 64) * 64, (size_t) kr.nsg, 1 }, l[3] = { 64, (size_t) kr.nsg, 1 };
                CK(clEnqueueNDRangeKernel(q, k, 3, nullptr, g, l, 0, nullptr, ev));
            };
            // correctness: every column against float64; mc3 columns also bitwise against the general kernel
            const int ncol = kr.mc ? n : 4;
            for (int c = 0; c < (kr.mc ? 1 : 4); c++) launch(c, nullptr);
            CK(clFinish(q)); std::vector<float> y((size_t) M * 4); CK(clEnqueueReadBuffer(q, yb, CL_TRUE, 0, y.size() * 4, y.data(), 0, nullptr, nullptr));
            double mre = 0; long ndiff = 0;
            for (int c = 0; c < ncol; c++) for (int i = 0; i < M; i++) {
                const float v = y[(size_t) c * M + i]; mre = std::max(mre, std::fabs(v - R[c][i]) / rms0);
                if (!kr.mc) g1[c].assign(y.begin() + (size_t) c * M, y.begin() + (size_t)(c + 1) * M);
                else if (memcmp(&v, &g1[c][i], 4)) ndiff++;
            }
            if (getenv("DBG")) for (int i = 0; i < 6; i++) printf("dbg %s row %d y=%.5f ref=%.5f\n", kr.name, i, y[i], R[0][i]);
            // timing: profiling time of reps launches, median
            std::vector<double> t;
            for (int r = 0; r < reps + 2; r++) { cl_event ev; launch(0, &ev); CK(clWaitForEvents(1, &ev)); cl_ulong a, b;
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &a, nullptr); clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &b, nullptr); clReleaseEvent(ev);
                if (r >= 2) t.push_back((b - a) * 1e-3); }
            std::sort(t.begin(), t.end());
            printf("mc3t kern=%s n_cols=%d nsg=%d private_b=%zu local_b=%zu max_rel_err=%.3e bits_vs_general_diff=%ld us_med=%.1f us_min=%.1f\n",
                   kr.name, n, kr.nsg, pm, lm, mre, kr.mc ? ndiff : -1L, t[t.size() / 2], t[0]);
            if (!kr.mc) break;
        }
        clReleaseKernel(k);
    }
    return 0;
}
