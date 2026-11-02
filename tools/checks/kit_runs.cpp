// Vollie: host stand-in for the research kit's run sequence in ONE process (lab/checks/kit_runs.sh).
// Usage: kit_runs <model> <threads> "<n_tokens>|ENV=V,ENV=V" ...  Each run: every kit knob reset to ''
// (as research.dart setEnv), the run's env set, KV cleared, the story prompt decoded, n_tokens greedy.
// Prints "run <i> <spec>" to stderr before each run, so the cpu-prof lines can be told apart.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * knobs[] = {"GGML_CPU_PROF", "GGML_CPU_AB", "GGML_MMID", "GGML_Q3K_X4", "GGML_Q3K_VEC", "GGML_Q3K_ILP",
                               "GGML_Q3K_PF", "GGML_EXPERT_ANON", "GGML_EXPERT_SKIP", "GGML_EXPERT_SKIP_RENORM"};

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
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
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
        const size_t bar = spec.find('|');
        const int n = atoi(spec.substr(0, bar).c_str());
        std::string env = bar == std::string::npos ? "" : spec.substr(bar + 1);
        for (size_t s = 0; s < env.size();) {
            size_t e = env.find(',', s);
            if (e == std::string::npos) e = env.size();
            const std::string kv = env.substr(s, e - s);
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
            s = e + 1;
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
