// c3_head_dump: greedy chat decode with libllama, dumping the last hidden state (post output-norm, the vector the
// output head reads) at every decoded position, plus the greedy token it produced. Model-generic: the chat template
// is the model's own (llama_chat_apply_template), BOS follows the file's add_bos flag.
//
// Input: one user message per line; "\n" and "\\" are escaped as in C. Lines are prompts 0..N-1 in file order.
// Output: <out>.rows  int32[4] per row {prompt, step, token, pos}; step 0 is the last prompt token, token is the
//                     greedy token produced from that position (argmax, lowest id on ties)
//         <out>.h16   fp16[n_embd] per row, same order (rows of one prompt are contiguous only when n_par == 1)
//         <out>.skip  prompt indices skipped (longer than max_prompt tokens)
//         <out>.ptok  per prompt int32 {prompt, n} then n prompt token ids (chat-formatted, as decoded)
//         <out>.topk  int32[64] per row, the 64 best token ids of the row's logits, best first
//         <out>.gap   float32 per row, the largest logit minus the second largest (near-ties)
// Knobs (E-T23, ebony-joins-r12 §2.1 and r13 §2.2); all unset = the tool as it was, the four files above byte-identical:
//   C3_HEAD_RENDERED=1  the lines are prompts already rendered by the app's template; tokenized with special tokens
//                       parsed, BOS added only if the text does not start with the vocabulary's BOS text (and then only
//                       if the file's add_bos flag allows it): the app's rule
//   C3_HEAD_ONEBATCH=1  the whole prompt in one decode, its last row written as step 0 (the app's feeding); n_par 1 only
//   C3_HEAD_KV=q8_0     K and V cache Q8_0 with flash attention on (the app's setting); f16 or unset = default
// Build: see tools/c3_head_build.sh. Usage:
//   c3_head_dump <model.gguf> <prompts.txt> <out> <n_gen> <n_par> [n_threads=8] [max_prompt=1024]
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

static std::string unescape(const std::string & s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            o += c == 'n' ? '\n' : c == 't' ? '\t' : c;
        } else o += s[i];
    }
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model prompts out n_gen n_par [n_threads] [max_prompt]\n", argv[0]); return 1; }
    const int n_gen = atoi(argv[4]), n_par = atoi(argv[5]);
    const int n_thr = argc > 6 ? atoi(argv[6]) : 8, max_prompt = argc > 7 ? atoi(argv[7]) : 1024;

    std::vector<std::string> prompts;
    { std::ifstream f(argv[2]); std::string l; while (std::getline(f, l)) prompts.push_back(unescape(l)); }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("C3_HEAD_NGL") ? atoi(getenv("C3_HEAD_NGL")) : 0;  // 0: CPU, as the phone; 99: Metal (training dumps only)
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model), n_vocab = llama_vocab_n_tokens(vocab);

    const int per_seq = max_prompt + n_gen + 16;
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (per_seq * n_par);
    cp.n_batch = cp.n_ubatch = (uint32_t) std::max(max_prompt, n_par);
    cp.n_seq_max = (uint32_t) n_par;
    cp.n_threads = cp.n_threads_batch = n_thr;
    cp.kv_unified = getenv("C3_HEAD_UNIFIED") ? atoi(getenv("C3_HEAD_UNIFIED")) != 0 : true;
    cp.no_perf = true;
    if (getenv("C3_HEAD_KV") && !strcmp(getenv("C3_HEAD_KV"), "q8_0")) {
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else if (getenv("C3_HEAD_KV") && strcmp(getenv("C3_HEAD_KV"), "f16")) { fprintf(stderr, "C3_HEAD_KV: f16 or q8_0\n"); return 1; }
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.embeddings = true;  // toggling llama_set_embeddings between decodes crashed in decode at this base; prompt tokens are then all output (extra head pass, harmless)
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    const bool rendered = getenv("C3_HEAD_RENDERED") && atoi(getenv("C3_HEAD_RENDERED"));
    const bool onebatch = getenv("C3_HEAD_ONEBATCH") && atoi(getenv("C3_HEAD_ONEBATCH"));
    if (onebatch && n_par != 1) { fprintf(stderr, "C3_HEAD_ONEBATCH needs n_par 1\n"); return 1; }
    const std::string bos_text = llama_vocab_bos(vocab) >= 0 ? llama_vocab_get_text(vocab, llama_vocab_bos(vocab)) : "";

    FILE * frows = fopen((std::string(argv[3]) + ".rows").c_str(), "wb");
    FILE * fh = fopen((std::string(argv[3]) + ".h16").c_str(), "wb");
    FILE * fskip = fopen((std::string(argv[3]) + ".skip").c_str(), "w");
    FILE * fptok = fopen((std::string(argv[3]) + ".ptok").c_str(), "wb");
    FILE * ftopk = fopen((std::string(argv[3]) + ".topk").c_str(), "wb");
    FILE * fgap = fopen((std::string(argv[3]) + ".gap").c_str(), "wb");
    const int TOPK = 64;

    struct Slot { int prompt = -1, step = 0, pos = 0; llama_token next = 0; };
    std::vector<Slot> slots(n_par);
    llama_batch b = llama_batch_init(std::max(max_prompt, n_par), 0, 1);
    std::vector<ggml_fp16_t> h16(n_embd);
    int next_prompt = 0, n_rows = 0, n_done = 0;

    auto emit = [&](int s, int i) {  // write the row of slot s from output i of the last decode, advance the slot
        Slot & sl = slots[s];
        const float * lg = llama_get_logits_ith(ctx, i);
        const float * em = llama_get_embeddings_ith(ctx, i);
        int best = 0;
        for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[best]) best = v;
        ggml_fp32_to_fp16_row(em, h16.data(), n_embd);
        int32_t rec[4] = {sl.prompt, sl.step, best, sl.pos};
        fwrite(rec, sizeof(rec), 1, frows);
        fwrite(h16.data(), sizeof(ggml_fp16_t), n_embd, fh);
        {   // top-64 token ids of this row's logits, best first (for last-logit rescoring)
            std::vector<int32_t> ids(n_vocab);
            for (int v = 0; v < n_vocab; v++) ids[v] = v;
            std::partial_sort(ids.begin(), ids.begin() + TOPK, ids.end(), [&](int x, int y) { return lg[x] > lg[y] || (lg[x] == lg[y] && x < y); });
            fwrite(ids.data(), sizeof(int32_t), TOPK, ftopk);
            const float gap = lg[ids[0]] - lg[ids[1]];
            fwrite(&gap, sizeof(float), 1, fgap);
        }
        n_rows++;
        sl.step++; sl.pos++; sl.next = best;
        if (llama_vocab_is_eog(vocab, best) || sl.step >= n_gen) {
            n_done++;
            if (n_done % 50 == 0) { fprintf(stderr, "done %d prompts, %d rows\n", n_done, n_rows); fflush(frows); fflush(fh); }
            sl.prompt = -2;  // refilled after this step: start() decodes, which would overwrite the outputs read above
        }
    };

    auto start = [&](int s) -> bool {  // load the next prompt into slot s; false when none left
        while (next_prompt < (int) prompts.size()) {
            int pi = next_prompt++;
            std::string text;
            if (rendered) text = prompts[pi];
            else {
                llama_chat_message msg = {"user", prompts[pi].c_str()};
                std::vector<char> buf(prompts[pi].size() * 2 + 512);
                int n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int) buf.size());
                if (n > (int) buf.size()) { buf.resize(n + 1); n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int) buf.size()); }
                if (n < 0) { fprintf(stderr, "template failed\n"); exit(1); }
                text.assign(buf.data(), n);
            }
            const bool add_special = !rendered || bos_text.empty() || text.compare(0, bos_text.size(), bos_text) != 0;
            std::vector<llama_token> tok(text.size() + 8);
            int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), add_special, true);
            if (nt < 2 || nt > max_prompt) { fprintf(fskip, "%d\n", pi); continue; }
            if (pi == 0) fprintf(stderr, "prompt 0 formatted (%d tokens):\n%s\n---\n", nt, text.c_str());
            llama_memory_seq_rm(mem, s, -1, -1);
            b.n_tokens = 0;
            const int nf = onebatch ? nt : nt - 1;   // onebatch: the last prompt token too, its row is step 0
            for (int i = 0; i < nf; i++) {
                b.token[i] = tok[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = onebatch && i == nf - 1;
            }
            b.n_tokens = nf;
            if (llama_decode(ctx, b)) { fprintf(stderr, "prompt decode failed\n"); exit(1); }
            { int32_t hdr[2] = {pi, nt}; fwrite(hdr, sizeof(hdr), 1, fptok); fwrite(tok.data(), sizeof(llama_token), nt, fptok); }
            slots[s] = {pi, 0, nt - 1, tok[nt - 1]};
            if (onebatch) { emit(s, nf - 1); if (slots[s].prompt == -2) continue; }   // done at step 0: next prompt
            return true;
        }
        slots[s].prompt = -1;
        return false;
    };

    for (int s = 0; s < n_par; s++) start(s);
    for (;;) {
        std::vector<int> act;
        for (int s = 0; s < n_par; s++) if (slots[s].prompt >= 0) act.push_back(s);
        if (act.empty()) break;
        b.n_tokens = (int) act.size();
        for (int i = 0; i < (int) act.size(); i++) {
            Slot & sl = slots[act[i]];
            b.token[i] = sl.next; b.pos[i] = sl.pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = act[i]; b.logits[i] = 1;
        }
        if (llama_decode(ctx, b)) { fprintf(stderr, "step decode failed\n"); return 1; }
        for (int i = 0; i < (int) act.size(); i++) emit(act[i], i);
        for (int s : act) if (slots[s].prompt == -2) start(s);
    }
    fprintf(stderr, "finished: %d prompts, %d rows, n_embd %d, n_vocab %d\n", n_done, n_rows, n_embd, n_vocab);
    fclose(frows); fclose(fh); fclose(fskip); fclose(fptok); fclose(ftopk); fclose(fgap);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
