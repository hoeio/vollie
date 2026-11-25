// T2 (Ebony r18 §4): a trailing one-line kernel that stores a sequence number into a fine-grain SVM word behind
// upstream's general q4_0 gemv; the host polls the word. Checks (*host*, PoCL; order and visibility only):
//  identity: the gemv output with the trailing kernel equals the output without it, bit for bit, every cycle;
//  order/visibility: when the host sees the flag, every row of dst (fine-grain SVM) already holds its final value.
// usage: t2 <gemv.cl> <gemv opts> <M> <K> <cycles>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <atomic>
#include <chrono>
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
using clk = std::chrono::steady_clock;
static const char * DONE_SRC = R"(
__kernel void done11(__global atomic_int * f, int seq) {
    if (get_global_id(0) == 0) atomic_store_explicit(f, seq, memory_order_release, memory_scope_all_svm_devices);
}
)";
static cl_program build(cl_context ctx, cl_device_id dev, const std::string & src, const char * opt) {
    cl_int e; const char * s = src.c_str(); cl_program p = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e); CK(e);
    if (clBuildProgram(p, 1, &dev, opt, nullptr, nullptr)) { size_t n; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string l(n, 0); clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, &l[0], nullptr); fprintf(stderr, "build (%s) failed:\n%s\n", opt, l.c_str()); exit(1); }
    return p;
}
int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: t2 gemv.cl opts M K cycles\n"); return 2; }
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
    const int M = atoi(argv[3]), K = atoi(argv[4]), NC = atoi(argv[5]);
    cl_platform_id pl; cl_device_id dev; CK(clGetPlatformIDs(1, &pl, nullptr)); CK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 1, &dev, nullptr));
    cl_device_svm_capabilities svm = 0; clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof svm, &svm, nullptr);
    if (!(svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) { printf("t2 ok=0 reason=no_fine_grain_svm caps=%llx\n", (unsigned long long) svm); return 0; }
    const bool atom = svm & CL_DEVICE_SVM_ATOMICS;
    cl_int e; cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e); CK(e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &e); CK(e);   // in-order, as upstream
    cl_program pg = build(ctx, dev, ss.str(), argv[2]), pd = build(ctx, dev, DONE_SRC, "-cl-std=CL3.0");
    cl_kernel kg = clCreateKernel(pg, "kernel_gemv_noshuffle_q4_0_f32", &e); CK(e); cl_kernel kd = clCreateKernel(pd, "done11", &e); CK(e);
    const int NB = K / 32; std::mt19937 rng(5);
    std::vector<uint32_t> qv((size_t) M * K / 8); for (auto & x : qv) x = rng();
    std::vector<uint16_t> dv((size_t) M * NB, 0x2E66);
    std::vector<float> xs(K); std::normal_distribution<float> nd; for (auto & x : xs) x = nd(rng);
    cl_mem qb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qv.size() * 4, qv.data(), &e); CK(e);
    cl_mem db = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dv.size() * 2, dv.data(), &e); CK(e);
    cl_mem xb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xs.size() * 4, xs.data(), &e); CK(e);
    cl_mem yb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) M * 4, nullptr, &e); CK(e);
    cl_image_format fq = { CL_R, CL_UNSIGNED_INT32 }; cl_image_desc ds{}; ds.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; ds.image_width = (size_t) M * K / 8; ds.buffer = qb;
    cl_mem qimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fq, &ds, nullptr, &e); CK(e);
    cl_image_format fb = { CL_RGBA, CL_FLOAT }; cl_image_desc d2{}; d2.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER; d2.image_width = (size_t) K / 4; d2.buffer = xb;
    cl_mem bimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fb, &d2, nullptr, &e); CK(e);
    const cl_svm_mem_flags sf = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | (atom ? CL_MEM_SVM_ATOMICS : 0);
    float * ys = (float *) clSVMAlloc(ctx, sf, (size_t) M * 4, 0); int * flag = (int *) clSVMAlloc(ctx, sf, 64, 0);
    if (!ys || !flag) { printf("t2 ok=0 reason=svm_alloc\n"); return 0; }
    auto gemv = [&](bool svm_dst) {
        const uint64_t z = 0; const int one = 1; int a = 0;
        CK(clSetKernelArg(kg, a++, sizeof(cl_mem), &qimg)); CK(clSetKernelArg(kg, a++, sizeof(cl_mem), &db)); CK(clSetKernelArg(kg, a++, sizeof(cl_mem), &bimg)); CK(clSetKernelArg(kg, a++, 8, &z));
        if (svm_dst) CK(clSetKernelArgSVMPointer(kg, a++, ys)); else CK(clSetKernelArg(kg, a++, sizeof(cl_mem), &yb));
        CK(clSetKernelArg(kg, a++, 8, &z)); CK(clSetKernelArg(kg, a++, 4, &K)); CK(clSetKernelArg(kg, a++, 4, &M)); CK(clSetKernelArg(kg, a++, 4, &one));
        CK(clSetKernelArg(kg, a++, 4, &K)); CK(clSetKernelArg(kg, a++, 4, &one)); CK(clSetKernelArg(kg, a++, 4, &M)); CK(clSetKernelArg(kg, a++, 4, &one));
        CK(clSetKernelArg(kg, a++, 4, &one)); CK(clSetKernelArg(kg, a++, 4, &one));
        const size_t g[3] = { (size_t)((M / 2 + 63) / 64) * 64, 4, 1 }, l[3] = { 64, 4, 1 };
        CK(clEnqueueNDRangeKernel(q, kg, 3, nullptr, g, l, 0, nullptr, nullptr));
    };
    // reference: the gemv alone into an ordinary buffer
    gemv(false); CK(clFinish(q)); std::vector<float> ref(M); CK(clEnqueueReadBuffer(q, yb, CL_TRUE, 0, (size_t) M * 4, ref.data(), 0, nullptr, nullptr));
    auto * fa = reinterpret_cast<std::atomic<int> *>(flag); fa->store(0);
    long early_bad = 0, final_bad = 0, timeouts = 0; std::vector<double> seen_us, fin_us;
    for (int c = 1; c <= NC; c++) {
        for (int i = 0; i < M; i++) ys[i] = NAN;
        static const bool first = getenv("T2_FLAG_FIRST") != nullptr;   // negative control: the flag ahead of the gemv
        const size_t g1 = 1; CK(clSetKernelArgSVMPointer(kd, 0, flag)); CK(clSetKernelArg(kd, 1, 4, &c));
        if (first) CK(clEnqueueNDRangeKernel(q, kd, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
        gemv(true);
        if (!first) CK(clEnqueueNDRangeKernel(q, kd, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
        const auto t0 = clk::now(); CK(clFlush(q));
        bool seen = false; while (!(seen = fa->load(std::memory_order_acquire) >= c)) if (std::chrono::duration<double>(clk::now() - t0).count() > 2.0) break;
        const auto t1 = clk::now();
        if (!seen) timeouts++;
        else for (int i = 0; i < M; i++) if (memcmp(&ys[i], &ref[i], 4)) early_bad++;   // rows not final when the flag is seen
        CK(clFinish(q)); const auto t2 = clk::now();
        for (int i = 0; i < M; i++) if (memcmp(&ys[i], &ref[i], 4)) final_bad++;   // identity after the queue drains
        seen_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()); fin_us.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
    }
    std::sort(seen_us.begin(), seen_us.end()); std::sort(fin_us.begin(), fin_us.end());
    printf("t2 M=%d K=%d cycles=%d svm_atomics=%d timeouts=%ld rows_not_final_at_flag=%ld rows_differ_after_finish=%ld flush_to_seen_us_p50=%.1f p90=%.1f flush_to_finish_return_us_p50=%.1f ok=%d\n",
           M, K, NC, (int) atom, timeouts, early_bad, final_bad, seen_us[NC / 2], seen_us[NC * 9 / 10], fin_us[NC / 2], !timeouts && !early_bad && !final_bad);
    return 0;
}
