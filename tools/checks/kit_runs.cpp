// Portable entry point for outside readers: tools/reproduce.sh (this check assumes the private lab/ layout).
// Vollie: host stand-in for the research kit's run sequence in ONE process (lab/checks/kit_runs.sh).
// Usage: kit_runs <model> <threads> "<n_tokens>|ENV=V,ENV=V" ...  Each run: every kit knob reset to ''
// (as research.dart setEnv), the run's env set, KV cleared, the story prompt decoded, n_tokens greedy.
// Prints "run <i> <spec>" to stderr before each run, so the cpu-prof lines can be told apart.
// KIT_NOREPACK=1 (r8): load without extra buffers (no CPU_REPACK copies), as the app does on the phone.
// r9: a spec starting with 'w' ("w200|...") first runs the app's one-token warm-up ("Hi", chat-templated, 1 token) with
// the run's env, printing "warmup <i>" (the app does it at load and before a run whose module copies differ).
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * knobs[] = {"GGML_CPU_PROF", "GGML_CPU_AB", "GGML_MMID", "GGML_Q3K_X4", "GGML_Q3K_VEC", "GGML_Q3K_ILP",
                               "GGML_Q3K_PF", "GGML_EXPERT_ANON", "GGML_EXPERT_SKIP", "GGML_EXPERT_SKIP_RENORM",
                               "GGML_SSM_INPLACE", "GGML_CONV_RING", "GGML_CPU_AB_WARM", "GGML_CPU_AB_BLOCK",
                               "GGML_HEAD2", "GGML_HEAD2_Q", "GGML_HEAD2_BATCH", "GGML_MM_CHUNK",
                               "GGML_EXPERT_Q4_LAYERS", "GGML_DENSE_WILLNEED", "GGML_Q3K_KERN", "GGML_CPU_BENCH", "GGML_MMID_TOUCH", "GGML_CPU_AB_BAAB",
                               "VOLLIE_SAMPLER_BYPASS"};

static void set_tokens(llama_batch_ext * b, const llama_token * t, int n, llama_pos p0) {
    llama_batch_ext_clear(b);
    for (int i = 0; i < n; i++) {
        const int32_t idx = llama_batch_ext_add_token(b, 0, t[i]);
        const llama_pos pos = p0 + i;
        llama_batch_ext_set_pos(b, idx, &pos);
    }
    llama_batch_ext_set_output_logits(b, n - 1, true);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model threads \"N|ENV=V,...\" ...\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    if (getenv("KIT_NOREPACK") && *getenv("KIT_NOREPACK")) mp.use_extra_bufts = false; // the phone's path (R9: no repack)
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = "Write a short story about a lighthouse.";
    std::vector<llama_token> pt(256);
    pt.resize(llama_tokenize(vocab, prompt.c_str(), prompt.size(), pt.data(), pt.size(), true, true));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_threads = cp.n_threads_batch = atoi(argv[2]);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (int r = 3; r < argc; r++) {
        for (const char * k : knobs) setenv(k, "", 1);
        std::string spec = argv[r];
        const bool warm = !spec.empty() && spec[0] == 'w';
        if (warm) spec.erase(0, 1);
        const size_t bar = spec.find('|');
        const int n = atoi(spec.substr(0, bar).c_str());
        std::string env = bar == std::string::npos ? "" : spec.substr(bar + 1);
        std::string last; // a piece without '=' continues the previous value (GGML_CPU_AB=knob:A:0:1,B:0:1)
        for (size_t s = 0; s < env.size();) {
            size_t e = env.find(',', s);
            if (e == std::string::npos) e = env.size();
            const std::string kv = env.substr(s, e - s);
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) { last = kv.substr(0, eq); setenv(last.c_str(), kv.substr(eq + 1).c_str(), 1); }
            else if (!last.empty()) setenv(last.c_str(), (std::string(getenv(last.c_str())) + "," + kv).c_str(), 1);
            s = e + 1;
        }
        if (warm) {
            fprintf(stderr, "warmup %d\n", r - 2);
            llama_memory_clear(llama_get_memory(ctx), true);
            const std::string hi = "<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
            std::vector<llama_token> ht(64);
            ht.resize(llama_tokenize(vocab, hi.c_str(), hi.size(), ht.data(), ht.size(), false, true));
            set_tokens(batch, ht.data(), (int) ht.size(), 0);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return 1;
            llama_sampler_sample(smpl, ctx, -1);
        }
        fprintf(stderr, "run %d %s\n", r - 2, argv[r]);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);
        set_tokens(batch, pt.data(), (int) pt.size(), 0);
        int pos = 0, nt = (int) pt.size();
        for (int i = 0; i < n; i++) {
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return 1;
            pos += nt;
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            set_tokens(batch, &id, 1, pos); // ponytail: EOG not checked, the kit's 72/200 runs never hit it on this prompt
            nt = 1;
        }
    }
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
