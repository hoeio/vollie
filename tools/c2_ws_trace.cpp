// Page-touch tracer: which pages of the mmapped model file does the real runtime read, per phase?
// After load (+ the app's one-token "Hi" warm-up), the whole file mapping is set PROT_NONE; each first touch in a
// phase faults, is logged (phase, page) and re-opened. At every phase boundary the mapping is closed again, so each
// phase's set is exact at the host page size (16 KB on Apple silicon). Output: "phase page_index" lines + "# " notes.
// Campaign 2, memo-18. Run with the device's env (GGML_*), PT_NOREPACK=1 for repack off. Input of tools/c2_ws_trace.py.
// Build against a llama.cpp tree T (shared libs):
//   c++ -O2 -std=c++17 c2_ws_trace.cpp -I$T/include -I$T/ggml/include -L$T/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$T/build/bin -o c2_ws_trace
// usage: c2_ws_trace model.gguf rendered_prompt.txt forced_reply.txt n_gen out.log [threads=3] [runs=2]
#include "llama.h"
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static char * g_base = nullptr; static size_t g_len = 0, g_pg = 0;
static std::atomic<int> g_phase{-2};
static std::vector<uint64_t> g_log; static std::atomic<size_t> g_n{0};

static void on_fault(int sig, siginfo_t * si, void *) {
    char * a = (char *) si->si_addr;
    if (!g_base || a < g_base || a >= g_base + g_len) { signal(sig, SIG_DFL); return; }
    size_t pi = (size_t)(a - g_base) / g_pg;
    size_t i = g_n.fetch_add(1);
    if (i < g_log.size()) g_log[i] = ((uint64_t)(g_phase.load() + 2) << 40) | pi;
    mprotect(g_base + pi * g_pg, g_pg, PROT_READ);
}

#include "gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <sys/stat.h>
static gguf_context * g_gg = nullptr; static const char * g_path = nullptr;
static bool find_cb(struct ggml_tensor * t, bool ask, void *) { // finds the mapping from an expert tensor's address
    if (g_base || !ask) return false;
    for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) {
        const ggml_tensor * w = t->src[s];
        if (!w->data || !strstr(w->name, "_exps")) continue;
        int64_t id = gguf_find_tensor(g_gg, w->name); if (id < 0) continue;
        size_t off = gguf_get_data_offset(g_gg) + gguf_get_tensor_offset(g_gg, id);
        struct stat st; stat(g_path, &st);
        g_base = (char *) w->data - off; g_len = ((size_t) st.st_size + g_pg - 1) / g_pg * g_pg;
        fprintf(stderr, "base from %s: %p len %zu\n", w->name, (void *) g_base, g_len);
        break;
    }
    return false;
}
static size_t g_unmapped = 0; static bool g_first = true;
static void phase(int p) { // ranges llama.cpp unmapped after load (unmap_fragment) are skipped
    g_phase = p;
    if (!mprotect(g_base, g_len, PROT_NONE)) return;
    const size_t CH = 1 << 20;
    for (size_t o = 0; o < g_len; o += CH) {
        size_t l = std::min(CH, g_len - o);
        if (!mprotect(g_base + o, l, PROT_NONE)) continue;
        for (size_t q = o; q < o + l; q += g_pg) if (mprotect(g_base + q, g_pg, PROT_NONE) && g_first) g_unmapped++;
    }
    if (g_first) { fprintf(stderr, "unmapped pages %zu (%.1f MB)\n", g_unmapped, g_unmapped * g_pg / 1e6); g_first = false; }
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: ptrace model rendered.txt forced.txt n_gen out [threads] [runs]\n"); return 1; }
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string prompt = ss.str();
    std::ifstream ff(argv[3]); std::stringstream s2; s2 << ff.rdbuf(); std::string forced_txt = s2.str();
    int n_gen = atoi(argv[4]); FILE * out = fopen(argv[5], "w");
    int nth = argc > 6 ? atoi(argv[6]) : 3, runs = argc > 7 ? atoi(argv[7]) : 2;
    g_pg = getpagesize(); g_log.resize(1u << 26);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; if (getenv("PT_NOREPACK")) mp.use_extra_bufts = false;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = nth; cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.cb_eval = find_cb;
    g_path = argv[1]; { gguf_init_params gp = {true, nullptr}; g_gg = gguf_init_from_file(argv[1], gp); }
    llama_context * ctx = llama_init_from_model(m, cp);
    const llama_vocab * v = llama_model_get_vocab(m);
    struct sigaction sa = {}; sa.sa_sigaction = on_fault; sa.sa_flags = SA_SIGINFO; sigemptyset(&sa.sa_mask);
    sigaction(SIGBUS, &sa, nullptr); sigaction(SIGSEGV, &sa, nullptr);
    std::vector<llama_token> toks(4096), forced(4096);
    int n = llama_tokenize(v, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true); toks.resize(n);
    int nf = llama_tokenize(v, forced_txt.c_str(), forced_txt.size(), forced.data(), forced.size(), false, false); forced.resize(nf);
    fprintf(out, "# prompt_tokens %d forced %d\n", n, nf);
    { // the app's warm-up after load: "Hi", one token (phase -1)
        std::vector<llama_token> hi(16); int nh = llama_tokenize(v, "Hi", 2, hi.data(), hi.size(), true, true); hi.resize(nh);
        llama_decode(ctx, llama_batch_get_one(hi.data(), nh));
        llama_memory_clear(llama_get_memory(ctx), true);
    }
    fprintf(out, "# base_len %zu page %zu\n", g_len, g_pg);
    if (!g_base) { fprintf(stderr, "mapping not found\n"); return 2; }
    std::string text;
    for (int run = 0; run < runs; run++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        phase(run * 1000);                       // prefill of run r: phase r*1000; decode token i: r*1000 + i
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), n))) return 3;
        for (int i = 0; i < n_gen; i++) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            int nv = llama_vocab_n_tokens(v), best = 0;
            for (int j = 1; j < nv; j++) if (lg[j] > lg[best]) best = j;
            if (i < nf) best = forced[i];
            if (run == 0) { char pc[256]; int pl = llama_token_to_piece(v, best, pc, sizeof pc, 0, true); if (pl > 0) text.append(pc, pl); }
            llama_token t = best;
            phase(run * 1000 + i + 1);
            if (llama_decode(ctx, llama_batch_get_one(&t, 1))) return 4;
        }
    }
    g_phase = -2; mprotect(g_base, g_len, PROT_READ);
    size_t N = std::min(g_n.load(), g_log.size());
    for (size_t i = 0; i < N; i++) fprintf(out, "%lld %llu\n", (long long)(g_log[i] >> 40) - 2, (unsigned long long)(g_log[i] & ((1ull << 40) - 1)));
    fprintf(stderr, "events %zu text: %s\n", N, text.c_str());
    fclose(out); llama_free(ctx); llama_model_free(m);
    return 0;
}
