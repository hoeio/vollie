// c3_head_b2check (memo-09 T1b): is a batch-2 decode step bit-identical to two batch-1 steps?
// For each prompt: run A = prompt, then n greedy steps one token per decode, keeping every row's logits.
// Then B = the same prompt in a fresh sequence, then the same tokens two per decode ([y0,y1], [y2,y3], ...), both
// positions output. Every B row is compared with A's row at the same position: bytes (memcmp), max |diff|, argmax.
// Model-generic; which CPU path runs (ggml repack, KleidiAI) is decided by the libllama it is linked against.
//   c3_head_b2check <model.gguf> <prompts.txt> <n_steps> [n_threads=4]
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::string unescape(const std::string & s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) { char c = s[++i]; o += c == 'n' ? '\n' : c == 't' ? '\t' : c; }
        else o += s[i];
    }
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model prompts n_steps [n_threads]\n", argv[0]); return 1; }
    const int n_steps = atoi(argv[3]) & ~1, n_thr = argc > 4 ? atoi(argv[4]) : 4;
    std::vector<std::string> prompts;
    { std::ifstream f(argv[2]); std::string l; while (std::getline(f, l)) prompts.push_back(unescape(l)); }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = n_thr; cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    llama_batch b = llama_batch_init(512, 0, 1);
    auto add = [&](llama_token t, int pos, bool out) {
        int i = b.n_tokens++; b.token[i] = t; b.pos[i] = pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = out;
    };
    long rows = 0, same = 0, same_argmax = 0; double worst = 0;
    for (size_t pi = 0; pi < prompts.size(); pi++) {
        llama_chat_message msg = {"user", prompts[pi].c_str()};
        std::vector<char> buf(prompts[pi].size() * 2 + 512);
        int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int) buf.size());
        std::string text(buf.data(), n);
        std::vector<llama_token> tok(text.size() + 8);
        int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), true, true);
        if (nt > 400) continue;
        // A: batch 1
        llama_memory_clear(mem, true);
        b.n_tokens = 0; for (int i = 0; i < nt; i++) add(tok[i], i, i == nt - 1);
        if (llama_decode(ctx, b)) return 1;
        std::vector<std::vector<float>> la;  // la[k] = logits of the row fed y_{k-1} (k >= 1)
        std::vector<llama_token> y;
        for (int k = 0; k <= n_steps; k++) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            int best = 0; for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[best]) best = v;
            if (k > 0) la.emplace_back(lg, lg + n_vocab);
            if (k == n_steps) break;
            y.push_back(best);
            b.n_tokens = 0; add(best, nt + k, true);
            if (llama_decode(ctx, b)) return 1;
        }
        // B: batch 2 with the true tokens
        llama_memory_clear(mem, true);
        b.n_tokens = 0; for (int i = 0; i < nt; i++) add(tok[i], i, i == nt - 1);
        if (llama_decode(ctx, b)) return 1;
        long p_same = 0;
        for (int k = 0; k < n_steps; k += 2) {
            b.n_tokens = 0; add(y[k], nt + k, true); add(y[k + 1], nt + k + 1, true);
            if (llama_decode(ctx, b)) return 1;
            for (int q = 0; q < 2; q++) {
                const float * lg = llama_get_logits_ith(ctx, q);
                const std::vector<float> & ref = la[k + q];
                bool eq = memcmp(lg, ref.data(), sizeof(float) * n_vocab) == 0;
                int ba = 0, bb = 0; double md = 0;
                for (int v = 0; v < n_vocab; v++) {
                    if (lg[v] > lg[ba]) ba = v;
                    if (ref[v] > ref[bb]) bb = v;
                    md = std::fmax(md, std::fabs((double) lg[v] - ref[v]));
                }
                rows++; same += eq; p_same += eq; same_argmax += ba == bb; worst = std::fmax(worst, md);
            }
        }
        fprintf(stderr, "prompt %zu: %ld of %d rows bit-identical\n", pi, p_same, n_steps);
    }
    printf("B2CHECK rows %ld bit_identical %ld argmax_equal %ld max_abs_diff %.6g\n", rows, same, same_argmax, worst);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model);
    return 0;
}
