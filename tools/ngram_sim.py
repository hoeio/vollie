"""Prompt-lookup (n-gram) speculative decoding acceptance, simulated on a greedy token sequence.
Deterministic given the tokens: at each step the draft looks for the last n tokens in prompt+output so far
(most recent match first) and proposes the next d tokens after the match; accepted = longest prefix equal to
the actual greedy continuation. Reports tokens produced per model evaluation (verify of d+1 tokens).
usage: python3 -I ngram_sim.py <prompt_ids.txt> <output_ids.txt> [n=3] [d=4]"""
import sys
def ids(p):
    return [int(x) for x in open(p).read().replace(',', ' ').replace('[', ' ').replace(']', ' ').split()]
prompt, out = ids(sys.argv[1]), ids(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
d = int(sys.argv[4]) if len(sys.argv) > 4 else 4
seq = prompt[:]
i = 0; evals = 0; accepted_total = 0; drafted_evals = 0
while i < len(out):
    # draft
    draft = []
    if len(seq) >= n:
        key = seq[-n:]
        for j in range(len(seq) - n - 1, -1, -1):
            if seq[j:j + n] == key:
                draft = seq[j + n:j + n + d]
                break
    evals += 1
    acc = 0
    while acc < len(draft) and i + acc < len(out) and draft[acc] == out[i + acc]:
        acc += 1
    if draft:
        drafted_evals += 1
    accepted_total += acc
    step = acc + 1  # accepted drafts + the model's own token
    seq.extend(out[i:i + step]); i += step
print(f'n={n} d={d}: out={len(out)} tokens, model evals={evals}, tokens/eval={len(out) / evals:.2f}, '
      f'evals with a draft={drafted_evals} ({drafted_evals / evals:.0%}), accepted drafts={accepted_total}')
