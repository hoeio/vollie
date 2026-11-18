# Campaign 3 memo index

| memo | date | title | verdict |
| [memo-02](memo-02-host-screens.md) | 2026-11-19 | Host screens H1, H3, H6, H7 (*host*) | Column input sparsity open on both models (Mistral 40 %, Qwen2.5 35 %); exact column kernel 2.6x sdot per touched weight (*host*), so the phone's read rate decides. Sublayer skip and per-token FFN skip closed. RWKV-7 90.5 % exact zeros. Q4_0 within 1.02 to 1.04 of Q8_0. |
|---|---|---|---|
| [memo-01](memo-01-feasibility.md) | 2026-11-18 | Is 15 tok/s on a dense 7B-class model within reach? (joint round, Bonnie and Ebony) | 15 tok/s on a body that reads 7.1 B or more weights per token is out of reach with every priced method, P < 0.05 (*estimated*). Best priced combination 5 to 9 tok/s (*estimated*). Gemma 4 E4B opening 7.5 to 14 tok/s (*estimated*). Host test list H1 to H7. Nothing measured. Drafts in [memo-01-work/](memo-01-work/). |
| [memo-03](memo-03-accelerators-e4b.md) | 2026-11-18 | Accelerators, bus headroom and Gemma 4 E4B (joint round, Bonnie and Ebony) | No path reaches 15 tok/s. Strict dense best about 12 tok/s, E4B 12.7 to 13.8 at its ceilings (*estimated*). GPU and Hexagon closed in stock form; CPU+GPU split and multi-stream GEMV ranked first. Device probes P1, P2, X1, X2 planned. Nothing measured. Drafts in [memo-03-work/](memo-03-work/). |
