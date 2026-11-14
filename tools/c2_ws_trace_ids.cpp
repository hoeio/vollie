// Logs selected expert ids and their mixture weights per layer and token (prefill batch + greedy decode).
// Campaign 2, memo-18: the --ids input of tools/c2_ws_trace.py. Lines: "P|D tok layer norm id:w ...".
// Fixed against memo-13's scratch version: ffn_moe_topk is a view of the argsort, rows nb[1] apart; reading it as
// contiguous gave wrong prefill ids for every token after the first. Use the prefill lines (P) only: with the eval
// callback set, campaign 1's runtime can compute the one-token graphs differently (fusions split), so the decode
// sets come from the page trace (tools/c2_ws_trace.cpp). Build as c2_ws_trace.cpp.
// usage: c2_ws_trace_ids model.gguf rendered_prompt.txt n_gen out.txt [forced_reply.txt]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>

static FILE * out; static const char * phase = "P"; static int tok0 = 0;
static std::map<int, std::vector<int32_t>> ids;   // layer -> ids [k*n_tok]
static int kk = 0, ntok = 0;

static int layer_of(const char * n) { const char * d = strrchr(n, '-'); return d ? atoi(d + 1) : -1; }

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    bool topk = strncmp(n, "ffn_moe_topk-", 13) == 0;
    bool w = strncmp(n, "ffn_moe_weights-", 16) == 0 || strncmp(n, "ffn_moe_weights_norm-", 21) == 0;
    if (ask) return topk || w;
    int il = layer_of(n);
    std::vector<char> buf(ggml_nbytes(t));
    if (topk) { // a view of the argsort: rows are nb[1] apart, not ne[0]*4 (memo-13's trace read it as contiguous)
        kk = t->ne[0]; ntok = t->ne[1];
        ids[il].resize(kk * ntok);
        for (int j = 0; j < ntok; j++) ggml_backend_tensor_get(t, ids[il].data() + j * kk, j * t->nb[1], kk * sizeof(int32_t));
        return true;
    }
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    if (topk) {
    } else { // weights [1, k, n_tok] f32; norm variant overrides the earlier line
        const float * f = (const float *) buf.data();
        bool norm = strncmp(n, "ffn_moe_weights_norm-", 21) == 0;
        for (int j = 0; j < ntok; j++) {
            fprintf(out, "%s %d %d %d", phase, tok0 + j, il, norm ? 1 : 0);
            for (int s = 0; s < kk; s++) fprintf(out, " %d:%.6g", ids[il][j*kk + s], f[j*kk + s]);
            fprintf(out, "\n");
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: trace model rendered.txt n_gen out.txt\n"); return 1; }
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string prompt = ss.str();
    int n_gen = atoi(argv[3]); out = fopen(argv[4], "w");
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 512; cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = cp.n_threads_batch = 3; cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(m, cp);
    const llama_vocab * v = llama_model_get_vocab(m);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(v, prompt.c_str(), prompt.size(), toks.data(), toks.size(), /*add_special*/ true, /*parse_special*/ true);
    toks.resize(n);
    fprintf(stderr, "prompt tokens %d, first:", n); for (int i = 0; i < 6 && i < n; i++) fprintf(stderr, " %d", toks[i]); fprintf(stderr, "\n");
    fprintf(out, "# prompt_tokens %d\n", n);
    llama_batch b = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, b)) return 2;
    phase = "D"; tok0 = 0;
    std::string text;
    std::vector<llama_token> forced;
    if (argc > 5) { std::ifstream ff(argv[5]); std::stringstream s2; s2 << ff.rdbuf(); std::string ft = s2.str();
        forced.resize(4096); int nf = llama_tokenize(v, ft.c_str(), ft.size(), forced.data(), forced.size(), false, false); forced.resize(nf);
        fprintf(stderr, "forced tokens %d\n", nf); }
    for (int i = 0; i < n_gen; i++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        int nv = llama_vocab_n_tokens(v), best = 0;
        for (int j = 1; j < nv; j++) if (lg[j] > lg[best]) best = j;
        if (i < (int) forced.size()) best = forced[i];
        char piece[256]; int pl = llama_token_to_piece(v, best, piece, sizeof piece, 0, true); if (pl > 0) text.append(piece, pl);
        llama_token t = best; tok0 = i;
        if (llama_decode(ctx, llama_batch_get_one(&t, 1))) return 3;
    }
    fprintf(stderr, "text: %s\n", text.c_str());
    fclose(out);
    return 0;
}
