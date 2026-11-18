# c3 memo-03: accelerators, bus headroom, Gemma 4 E4B, and a re-price of strict dense (joint round 2)

Date: 2026-11-18. Authors: Bonnie and Ebony (`claude-fable-5-2`). Merged by the lead (`claude-opus-5-5`). Drafts and
cross-reviews: [memo-03-work/](memo-03-work/). Nothing was measured on the phone. The only new *host* facts are
GGUF header reads (first 32 MB of each file) and reads of the base tree. Everything else is published or *estimated*.

## 1. Verdict

memo-01 stands: **no priced path reaches 15 tok/s**, neither on a strict dense 7B body nor on Gemma 4 E4B.

- **Bus limit.** 15 tok/s on 4.0 GB per token needs 60 GB/s. That is above the nominal 51.2 GB/s bus, and Ebony
  puts the reachable ceiling at about 45 GB/s.
- **Bonnie's best case on strict dense.** About 12 tok/s: 11.9 to 12.7 at 40 % sparsity, a 1.2 × kernel and
  r ≥ 35 GB/s.
- **E4B.** 12.7 to 13.8 tok/s at the bus and `sdot` ceilings, and P(15) of 0.03 to 0.08.

## 2. Ranked levers for a strict dense 7B body (Ebony; 4.0 GB per token; CPU alone 4.5 to 5.7 tok/s, *estimated*)

| rank | lever | tok/s | note |
|---|---|---|---|
| 1 | Concurrent CPU + GPU split of the decode mat-vecs | 5.6 to 8.6 | P 0.3; numerics change; large build |
| 2 | Multi-stream GEMV on the CPU | 5 to 7.4 | P 0.3; bit-identical; uses the same bus headroom as 1, so the two do not add |
| 3 | Per-token FFN-sublayer skip | 4.7 to 6.4 | numerics change; no new kernel |
| 4 | A55 helper readers, a 4th dense thread, 2 MB pages | 4.6 to 6.2 | |
| 5 | GPU alone with a purpose-written kernel | 3.5 to 6 | |
| – | Stock OpenCL backend on the GPU alone | 1 to 2.5 | closed: upstream lists phone Adreno 6xx as unsupported |
| – | Hexagon 780 (v68) | 1.2 to 2.8 | closed: v68 is HVX-only, at 0.27 to 0.49 of the CPU (*published*); QNN needs a converted file |

**Column sparsity** (memo-01 Y3/Y4) is re-priced by Bonnie §9 as a function of the read rate r:
- 6.8 to 8.8 tok/s at r = 19 to 24 GB/s;
- 11.9 to 12.7 tok/s at its best corner.

## 3. Gemma 4 E4B (provisional; held for the owner)

**Bytes and band.**
- 4,644 M weights are multiplied per token (*host*, header). The 2,819 M PLE table is a lookup and can stay on
  storage.
- Published Q4_0 files read 2.83 to 2.99 GB per token; an all-Q4_0 mix reads 2.61 GB.
- The band is 6.7 to 13.8 tok/s for r from 19 to 51 GB/s. A CPU + GPU split moves it to 8.3 to 12.9 printed.
- The base `620d2225c` already runs the model, with lazy PLE reads.

**Quality risk (*published*).**
- The stock base model at Q4_K_M has 3.24 × BF16 perplexity; at Q5_K_M, 1.17 ×.
- Only Google's QAT file is near F16 at four bits. New owner question: does a QAT file count as "stock weights"?
- Ebony thinks the fragility may sit in one tensor group. A per-group ladder is proposed, with a Q8_0 PLE table.

**Residency.** 2.61 GB is at the edge of the 2.67 GB held in R58 (*device*).

**Carry-over from campaign 2.**
- Pin, the exec map and kv clear carry over.
- Touch carries over only with a tensor-range page set that leaves out the PLE table.
- The two-stage head only buys a 6- or 8-bit head at Q4_0 speed.

## 4. Corrections

- memo-01 §7 on campaign 1's GPU numbers was wrong. A dated note was added there.
- "The CPU saturates the DRAM" was never measured. A same-bus Adreno 730 reads 39 GB/s (*published*).

## 5. Lead's decisions

- **H6 kill line.** It is set as a function of P2's measured read rate, following Ebony: the kernel is killed if
  its cost per touched weight exceeds break-even at the measured r (about 3.5 × at 19 GB/s, about 1.7 × from
  39 GB/s). Bonnie's fixed 1.6 × is recorded as the conservative alternative.
- **Price columns.** Columns above 45 GB/s are reported as above the reachable ceiling.
- **Device probe set.** P1 (resident capacity), P2 (dense read rate and `sdot` ceiling, power saving off), X1
  (bus headroom: two engines and streams reading together) and X2 (GPU read rate) go together into one probe
  kit. They are about 15 phone-minutes in all and are run in the next phone window. Tom builds it after the host
  screens.
- **Host additions for Tom.**
  - X4: a per-token FFN-skip screen, added to H7.
  - The Gemma 4 E4B per-group quantisation ladder, but only with the owner's QAT answer pending, and only if disk
    allows.
- **R20 (draft speculation).** It is reopened only if P2 reads under about 28 GB/s (Bonnie §11).
- **Goal.** Unchanged at 15 tok/s and not raised.

## 6. New owner questions (collected; not asked until the R68 JSON arrives)

1. Does Google's QAT file of Gemma 4 E4B count as "stock weights"? This is added to the E4B and RWKV-7 questions
   in STATUS.
