#!/usr/bin/env python3
"""memo-12 round 11 (Bonnie): what KleidiAI's re-quantisation of a Q6_K head (int8, one scale per row = max|w| / 127,
kleidiai.cpp repack) adds to the head's error, per model. *host*, file arithmetic only, no model run.
For a sample of rows: dequantise the served Q6_K row (gguf-py), re-quantise as KleidiAI does, and report
  crest   = max|w| / rms(w) per row (median, 90th, 99th percentile)
  e_rel   = rms(requantised - served) / rms(served), per row and pooled (the added error, relative to the row)
Run: lab/venv-c3/bin/python3 -I -B bonnie_head_requant.py <repo root>
"""
import sys, os
import numpy as np
root = sys.argv[1]; sys.path.insert(0, os.path.join(root, "lab/llama.cpp-c3/gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T
FILES = [("OLMoE-Mi (c2; R61: +0.03 % with the head in KleidiAI)", "lab/olmoe/olmoe-Mi.gguf", "output.weight"),
         ("Granite-Mi (c2; R61: +0.57 %)", "lab/granite/granite-Mi.gguf", "token_embd.weight"),
         ("Yi-1.5-6B", "lab/c3/models/yi15-6b-q4_0.gguf", "output.weight"),
         ("Llama-2-7B", "lab/c3/models/llama2-7b-q4_0.gguf", "output.weight"),
         ("DeepSeek-7B", "lab/c3/models/deepseek7b-q4_0.gguf", "output.weight")]
rng = np.random.default_rng(0)
for label, path, name in FILES:
    p = os.path.join(root, path)
    if not os.path.exists(p): print(f"{label}: file missing"); continue
    t = next((x for x in GGUFReader(p).tensors if x.name == name), None)
    if t is None or t.tensor_type != T.Q6_K: print(f"{label}: {name} not found as Q6_K ({None if t is None else t.tensor_type.name})"); continue
    k, n = int(t.shape[0]), int(t.shape[1]); rb = k // 256 * 210
    raw = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(n, rb)
    idx = np.sort(rng.choice(n, size=min(n, 6000), replace=False))
    w = dequantize(np.ascontiguousarray(raw[idx]), T.Q6_K).reshape(len(idx), k).astype(np.float64)
    mx = np.abs(w).max(1); rms = np.sqrt((w ** 2).mean(1)); ok = rms > 0
    sc = mx / 127.0; q = np.clip(np.rint(np.where(sc[:, None] > 0, w / np.where(sc[:, None] > 0, sc[:, None], 1), 0)), -127, 127)
    err = q * sc[:, None] - w; e_row = np.sqrt((err ** 2).mean(1)); crest = mx[ok] / rms[ok]
    print(f"{label}: rows {n} x {k}, sampled {len(idx)}")
    print(f"   crest max/rms: median {np.median(crest):.1f}, p90 {np.quantile(crest, .9):.1f}, p99 {np.quantile(crest, .99):.1f}, max {crest.max():.1f}")
    print(f"   added error / row rms: median {np.median(e_row[ok] / rms[ok]):.4f}, p90 {np.quantile(e_row[ok] / rms[ok], .9):.4f}, pooled {np.sqrt((err ** 2).mean() / (w ** 2).mean()):.4f}")
