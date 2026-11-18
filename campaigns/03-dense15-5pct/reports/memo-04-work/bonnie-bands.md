# c3 memo-04 work file, Bonnie: prediction bands and decision rules for the device probe kit (P1, P2, P2hot, X1, X2)

Agent model: `claude-fable-5-2` (Bonnie). Date 2026-11-19. Repository HEAD when written: `c310fb6`. Draft for the
lead; verdicts and the pre-registration text are the lead's. **Written before the probe kit exists and before any
probe line has been read. R68 (the first power-saving-off session with references) has not been run either, so
no dense rate with power saving off exists in the repository.**

Labels: *device* = a stored campaign 1 or 2 phone measurement, named by R-number; *host* = laptop;
*published* = someone else's measurement; *recalled* = from memory, not checked today; *estimated* = my
arithmetic or judgement. **Every band and centre below is *estimated*.** A band is my 80 % interval, a centre my
median; a value outside the band is scored as a miss of this draft.

Probe design priced: [memo-02 §6](../memo-02-host-screens.md). Decision context: [memo-01](../memo-01-feasibility.md),
[memo-03 §5](../memo-03-accelerators-e4b.md).

## 0. Summary

1. **P2 centre: 24 GB/s through the stock Q4_0 kernel on three A78s (band 19 to 30); 27 GB/s plain read (20 to
   36).** Power saving off is priced as a small gain (about +3 to +6 % from the clock), not a new regime:
   campaign 1's dense rate was flat between 2,112 and 1,555 MHz (*device*, R27, R29).
2. **P1 centre: 3.1 GB held (band 2.6 to 3.8).** P(3.5 GB step held) 0.35, P(4.0) 0.15. The most likely result
   is that the resident-capacity rule kills the two lead Q4_0 bodies (4.04 and 4.12 GB) on this probe alone.
3. **H6:** at the centre the break-even kernel cost is 2.7 x (Mistral) and 2.5 x (Qwen2.5), right at the host
   kernel's 2.6 x. The probe most likely lands in the zone where the A78 column-kernel row decides (P 0.75).
4. **R20:** P(r < 28 GB/s) is 0.8, so the reopen condition is most likely met; the expected gain at the centre is
   only about 1.15 x.
5. **CPU + GPU:** centre sum 30 GB/s, 1.25 x the CPU alone (band 0.95 to 1.60); P(sum ≥ 1.30 x) 0.35; P(the GPU
   block prints no usable number) 0.2.
6. **Six design points to settle before the build** (§7): no 3.3 GB and no 4.2 GB step in P1; no X1 + 2 A78 cell
   in P2; the 256 KB hot slice does not fit an A55's L2; a 2 s cell may sit inside the bus clock's ramp; the
   probe has no column-kernel row; `x1gpu` does not say which GPU kernel runs.
7. **None of this moves the campaign's verdict:** the best outcome of every rule below prices under 10 tok/s on a
   4.0 GB body (§6).

## 1. Inputs

| input | value | label, source |
|---|---|---|
| dense rate in decode, 3 threads | 18.9 (Q6_K head, R29), 19.7 (R16), 21.5 (R20), 22.5 to 22.8 (R31), 24.3 GB/s (0.0411 ms/MB, R25) | *device*; power saving most likely on (owner, 2026-11-18); logged clock 2,112 MHz |
| the same at 2,112 and at 1,555 MHz | head op 6.7 / 6.5 ms against 6.9 / 6.9 ms; dense `MUL_MAT` × 1.08 to 1.10 for a clock × 0.736 | *device*, R29, R27. So about a quarter of the dense time follows the clock (*estimated*) |
| 4 threads against 3 | 22.7 against 19.7 GB/s (× 1.15) | *device*, R13/R14 (`reports/04_campaign.md` §1) |
| stock Q4_0 4x4 gemv, cache-hot, 1 thread | A78 at 2,112 MHz: 78.58 ns per 1,536 weights = **0.0512 ns per weight** (0.108 cycles); X1 at 2,841: 43.23 ns = **0.0281 ns per weight** (0.080 cycles) | *device*, R41 (r13b, verified core) and R37 (r11; the core was identified as the X1 afterwards, read once) |
| A55, same kernel | no verified bench exists (R40's "A55" lines were A78s, R41 correction) | – |
| caps with power saving off | 2,419 (cpus 4 to 6) / 2,841 (cpu 7) | *device*, R67 |
| slow first tokens after idle | tokens 1 to 12 at 52 to 57 ms against 47 steady, no faults, same logged clock; a bus-clock ramp is the leading reading | *device*, kit r15 data re-read in the R48 entry (`reports/03_final.md`); cause not shown |
| resident file set | largest held: 2,672 MB (R58, OLMoE's whole working set, so not a measured limit); `Cached` before runs 1.8 to 3.2 GB (R55 to R67); campaign 1's 3.1 GB of mapped experts did not stay resident (R9) | *device* |
| anonymous memory | ran at 2.27 GB of repack copy; killed at 3.93 GB (R9); the app's own 0.5 to 0.7 GB (c2 memo-23); free swap 0 to 144 MB in most sessions | *device* |
| exec mapping | file MB lost per second of gate wait 1.8 against 8.7 (Granite), 0.7 against 3.5 (OLMoE) | *device*, R67; idle waits, not capacity under load |
| flash re-read | 0.7 to 2.7 ms per MB | *device* fits, c2 memo-18 |
| GPU on this phone | Adreno 660, compiler E031.38, segfaults on some kernels; one dense run about 35 % below the CPU, an effective 8.6 to 9.6 GB/s with the fast kernels disabled | *device*, campaign 1 (memo-03 Ebony §1) |
| GPU sum-read on other chips | Adreno 730 39 GB/s on a 51.2 GB/s bus; Adreno 750 63 on 77; Adreno 690 51 to 58 | *published*, clpeak (arXiv 2410.03613v5; gist of 2024-03-17) |
| two engines together | 59.1 of a nominal 76.8 GB/s on the 8 Gen 3; applied here: about 39, ceiling about 45 | *published* (HeteroLLM, arXiv 2501.14794), scaled *estimated* (memo-03) |
| host column kernel | 2.63 x `sdot` per touched weight at 60 % kept, 2.55 x at 65 % | *host*, memo-02 §4 |
| bytes per token | Mistral 4.037 GB, Qwen2.5 4.119 GB; 0.5625 bytes per weight in the 4x4 layout | *host*, memo-02 §1 |
| cache sizes | X1 1 MB L2, A78 512 KB L2, A55 128 KB L2, 4 MB shared L3 | *recalled*; Tom reads `/sys/devices/system/cpu/cpu*/cache/` in the probe header |

Searched today and **not found**: a measured main-memory read rate of the Snapdragon 888 per core or in total
(STREAM, lmbench; one site has a Snapdragon 888 graph as an image only), and a clpeak result for the Adreno 660.
The CPU bands therefore rest on campaign 1's device rates, the GPU bands on other chips.

Compute rates derived from the benches (*estimated*: scaled by clock; in situ = × 0.87, R41's kernel share):

| core set, clocks | hot, G weights/s | hot, GB/s of Q4_0 | in situ, GB/s (`c_b`) |
|---|---|---|---|
| 1 A78 at 2,419 | 22.4 | 12.6 | 11.0 |
| 2 A78 | 44.8 | 25.2 | 21.9 |
| 3 A78 | 67.2 | 37.8 | 32.9 |
| X1 at 2,841 | 35.5 | 20.0 | 17.4 |
| X1 + 2 A78 (the set memo-01 and memo-03 priced) | 80.3 | 45.2 | **39.3** |
| X1 + 3 A78 | 102.7 | 57.8 | 50.3 |
| 3 A78 at 2,112 (campaign 1's clock) | 58.6 | 33.0 | 28.7 |

## 2. P2 and P2hot bands

Reasoning for the centres:

- With power saving on, the stock kernel moved 22 to 24 GB/s in the later, cleaner rounds (R25, R31) and 19 to 20
  in the earlier ones. A microbench has no graph, barriers or waits, so it should sit at the upper end: 23 to 24.
- The dense time was nearly clock-flat, so the higher caps (× 1.145 on the A78s) are worth +3 to +6 %, not +15 %.
- Three A78s without the X1 (campaign 1's threads were not pinned to a known set) should read a little less than
  a set with the X1 in it.
- The plain read has no arithmetic in the loop, so it removes the quarter of the time that followed the clock:
  read ≈ 1.0 to 1.3 × the kernel's rate.
- Upside tail: power saving may also have held the memory bus clock down (unknown; no kit has logged it).
  This is the only route to 30 GB/s or more, and I give it 0.15.

### P2, `c3-probe: p2 kern=<read|q4x4> cores=<set> gbs=` (cold 1 GB arena)

| cores | read, GB/s: centre (band) | q4x4, GB/s: centre (band) | q4x4 `ns_per_w`: centre (band) | hot compute cap of the set, GB/s |
|---|---|---|---|---|
| 1 A78 | 10 (7 to 14) | 9 (6.5 to 12) | 0.063 (0.047 to 0.087) | 12.6 |
| 2 A78 | 19 (13 to 26) | 17 (12 to 23) | 0.066 (0.049 to 0.094) per thread | 25.2 |
| **3 A78** | **27 (20 to 36)** | **24 (19 to 30)** | 0.070 (0.056 to 0.089) per thread | 37.8 |
| X1 alone | 13 (9 to 19) | 12 (8 to 18) | 0.047 (0.031 to 0.070) | 20.0 |
| X1 + 3 A78 | 30 (22 to 40) | 27 (21 to 35) | – | 57.8 |
| 3 A78 + 4 A55 | 31 (22 to 42) | 28 (20 to 36) | – | 37.8 + A55s |

`ns_per_w` per thread = 0.5625 × threads ÷ GB/s. If the line prints wall time per weight of the whole cell
instead, divide by the thread count before comparing.

Ratios predicted, which are firmer than the levels:

| ratio | centre (band) | from |
|---|---|---|
| q4x4 ÷ read, 3 A78 | 0.90 (0.75 to 1.00) | R27's clock experiment |
| 3 A78 ÷ 1 A78, q4x4 | 2.6 (2.0 to 3.0) | sublinear, as 4 against 3 threads was |
| (X1 + 3 A78) ÷ 3 A78, q4x4 | 1.13 (1.00 to 1.30) | × 1.15, *device*, R13/R14 |
| (3 A78 + 4 A55) ÷ 3 A78, q4x4 | 1.15 (0.95 to 1.35) | A55 at 3 GB/s each at most by compute; R18's A55 loss was in situ, with joins |
| `mhz` before and after each cell | 2,419 on cpus 4 to 6 and 2,841 on cpu 7 in at least 80 % of cells, P 0.8 | R67. If the cap reads 2,035 to 2,150, the session is in campaign 1's clock state: every q4x4 centre falls by about 4 % and every P2hot centre scales by the clock |

### P2hot, `c3-probe: p2hot core=<a78|x1|a55> ns_per_w=`

Predicted in cycles per weight, so that the band holds at whatever clock the line logs: ns per weight = cycles ÷
logged GHz.

| core | cycles per weight: centre (band) | ns per weight at the expected clock | basis |
|---|---|---|---|
| A78 | 0.105 (0.095 to 0.118) | **0.0434 at 2,419 MHz** (0.039 to 0.049); 0.050 at 2,112 | R41: 0.108 at K 1,536; rows of 4,096 amortise the per-row step a little better |
| X1 | 0.078 (0.070 to 0.090) | **0.0275 at 2,841 MHz** (0.025 to 0.032) | R37: 0.080, one reading |
| A55 | 0.33 (0.22 to 0.48) | **0.18 at 1,804 MHz** (0.12 to 0.27) | no device number; in-order core, about one third of the A78's weights per cycle (*estimated*, wide). The 256 KB slice exceeds the A55's 128 KB L2 (*recalled*), so this line is an L3 number unless the slice is 64 KB (§7) |

Three-thread compute rate that follows from the centres (X1 + 2 A78, in situ): 0.87 × 0.5625 × (1/0.0275 +
2/0.0434) = **40.4 GB/s** (band 35 to 45). Memo-03's 39.3 used K 1,536.

## 3. P1 bands (resident capacity)

Definition proposed for scoring, to be fixed before the run: a step **is held** in an arm when, over passes 6 to
20, the median pass rate is at least 0.80 × that arm's median at the smallest step, and the median major faults
per pass are at most 50. **Capacity C** = the largest held step such that every smaller step is held too.

Why I expect a cliff and not a slope: cyclic sequential reading is the worst case for an LRU; once the set
exceeds what the phone keeps, the page needed next is always among the oldest. A pass that re-reads a fraction f
of N from flash at 0.7 to 2.7 ms per MB (*device* fits) runs at 1 ÷ (f × 0.7 to 2.7) GB/s: under 6 GB/s for any
f ≥ 0.25. Held steps read at DRAM speed, 19 GB/s or more. So the 0.80 line is not sensitive.

Why the centre is 3.1 GB: free swap is at or near zero in most sessions (*device*), so other apps' anonymous
memory cannot leave and the page cache is what gives way; `Cached` before runs has never been above 3.2 GB
(*device*); the probe runs without a model loaded, so the app's own anonymous memory is smaller than in a run,
and `lmkd` may kill background apps under pressure, which is the route to 3.5 GB or more.

| N, GB | P(held), `map=r` | P(held), `map=x` | GB/s if held, passes 6 to 20: centre (band) | GB/s if not held | majflt per pass if not held | `SwapFree`, `Cached` |
|---|---|---|---|---|---|---|
| 2.5 | 0.90 | 0.92 | 26 (19 to 36) | – | – | `Cached` ≥ 2.6 GB |
| 3.0 | 0.60 | 0.65 | 26 (19 to 36) | 1.5 (0.4 to 6) | 2,000 to 40,000 | `SwapFree` falls under 100 MB |
| 3.5 | 0.35 | 0.40 | 26 (19 to 36) | 1.2 (0.4 to 5) | 4,000 to 80,000 | `SwapFree` near 0 |
| 4.0 | 0.15 | 0.18 | 26 (19 to 36) | 1.0 (0.3 to 4) | 8,000 to 120,000 | near 0; background apps killed |
| 4.5 | 0.05 | 0.07 | 26 (19 to 36) | 0.9 (0.3 to 3) | 10,000 to 150,000 | near 0 |

- **Capacity C: centre 3.1 GB, band 2.6 to 3.8.** With the designed steps the most likely readings are "3.0 held,
  3.5 not" (P 0.25) and "2.5 held, 3.0 not" (P 0.30).
- Held rate = 0.95 × the P2 read rate of the same three cores (file pages and the anonymous arena both sit on
  4 KB pages; R40 read `anon_huge_kb=0`). Pass 1 of every step: 8 to 15 GB/s (first-touch mapping faults);
  not scored.
- **Exec arm:** I predict the same capacity step as the read arm in 4 of 5 steps (P 0.75) and at most one step
  higher. R67's effect was on idle loss; under continuous reading both mappings are referenced every 150 ms.
  If the exec arm holds a step that the read arm does not, that is new information and worth more than the
  centre.
- Order effect: the second arm of a step runs with the file already cached. Score the arms in the order
  written into the pre-registration, and alternate the order between steps if time allows.
- P(the kit process is killed during P1) 0.05; P(the write fails for space) is not priced: R67 left 9.4 GB free
  with both campaign 2 models copied in (*device*), and 4.5 GB is needed.
- **A real decode holds less than C.** It adds the KV, compute buffers and the app's 0.5 to 0.7 GB (*device*):
  usable body ≈ C − 0.3 GB (*estimated*, band 0.1 to 0.6), and C − 4 GB if the repack copy is not removed (Y1).

## 4. X1 bands (streams per thread, 3 A78)

Gain = `gbs` at k streams ÷ `gbs` at 1 stream, same cores, same session.

| streams per thread | gain: centre (band) | P(gain ≥ 1.15) |
|---|---|---|
| 1 | 1.00 by definition; its level should equal the P2 read cell of 3 A78 within ± 8 % (different region size) | – |
| 2 | 1.05 (0.95 to 1.25) | 0.22 |
| 4 | 1.05 (0.88 to 1.35) | 0.27 |

- For headroom: a fourth core added 15 % (*device*, R13/R14), and a GPU on the same nominal bus reads 39 GB/s
  (*published*). So the three-thread rate is at least partly a per-core limit.
- Against a gain: the A78's prefetcher already keeps one sequential stream far ahead; more streams help only if
  the per-core limit is the number of misses in flight for one stream. Four streams also cost four TLB walks per
  page step.
- **The gain a multi-stream GEMV can use is min(gain, `c_b` ÷ r)**: with `c_b` 32.9 (3 A78) and r 24, at most
  1.37 x; the read-loop gain is an upper bound for the kernel.

## 5. X2 bands (GPU)

| line | centre (band) | basis |
|---|---|---|
| block completes and prints all lines | P 0.80 | E031.38 has segfaulted on kernels before (*device*); a `float4` sum is the simplest kernel there is, the Q4 kernel less so |
| `x2 kern=read gbs=` | 30 (15 to 42) | other Adrenos read 0.76 to 0.85 of their nominal bus by clpeak (*published*), which would be 39 to 43 here; lowered for the older GPU, the driver, and a 2 s cell inside the GPU's clock ramp |
| `x2 kern=q4 gbs=` (a plain, untuned kernel) | 10 (3 to 22) | campaign 1's whole-token 8.6 to 9.6 GB/s effective with the fast kernels disabled (*device*); Ebony's 14 to 24 is for a tuned kernel |
| `x1gpu`, GPU kernel = read: `cpu_gbs` | 20 (14 to 27), i.e. 0.83 × the CPU alone (0.6 to 1.0) | the CPU loses part of the bus |
| same: `gpu_gbs` | 12 (4 to 24) | |
| same: `sum_gbs` | **30 (22 to 40)**; ratio to the 3-A78 q4x4 cell **1.25 (0.95 to 1.60)** | Ebony's 29 to 40; two-engine ratio of HeteroLLM gives about 39 |
| `x1gpu`, GPU kernel = q4: `sum_gbs` | 27 (20 to 36); ratio 1.12 (0.90 to 1.45) | the untuned kernel is the limit, not the bus |

P(sum ÷ CPU alone ≥ 1.30 with the read kernel) 0.35; ≥ 1.15: 0.60; under 1.05: 0.20.

## 6. Decision rules (drafts; each fixed before the run)

Definitions used by all rules, from the probe's own lines:

- **r** = `p2 kern=q4x4`, 3 threads (the X1 + 2 A78 cell if Tom adds it, else `cores=3 A78`), GB/s.
- **c_b** = 0.87 × 0.5625 × Σ (1 ÷ `p2hot ns_per_w`) over X1 + 2 A78, GB/s. Predicted 40.4 (35 to 45).
- **R** = c_b ÷ r: how many times compute outruns bytes. Predicted 1.68 (1.3 to 2.2).
- Token model (*estimated*, memo-03 §9): stock = max(B ÷ r, B ÷ c_b) + O; column kernel at kept fraction d and
  kernel cost k: max(b × B ÷ r, k × d × B ÷ c_b) + O, with b = 1 − 8/9 × (1 − d) (the scales are always read).
  Mistral d 0.60, b 0.644; Qwen2.5 d 0.65, b 0.689 (*host*, memo-02 §3).

### Rule A: the read rate that sets H6's line

The lead's line (memo-03 §5): the column kernel is killed if its cost per touched weight exceeds break-even,
**k\*(r) = max(R, 1) ÷ d**. With c_b = 39.3 (*estimated*; use the measured c_b on the day):

| r, GB/s | 16 | 19 | 20 | 22 | 24 | 26 | 28 | 30 | 33 | 36 | ≥ 39.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k\*, Mistral (d 0.60) | 4.09 | 3.45 | 3.27 | 2.98 | 2.73 | 2.52 | 2.34 | 2.18 | 1.98 | 1.82 | 1.67 |
| k\*, Qwen2.5 (d 0.65) | 3.78 | 3.18 | 3.02 | 2.75 | 2.52 | 2.33 | 2.16 | 2.02 | 1.83 | 1.68 | 1.54 |
| token gain at the host's k = 2.6, Mistral / Qwen2.5 | 1.55 / 1.45 | 1.33 / 1.22 | 1.26 / 1.16 | 1.15 / 1.06 | 1.05 / 0.97 | 0.97 / 0.89 | 0.90 / 0.83 | 0.84 / 0.78 | 0.76 / 0.70 | 0.70 / 0.65 | 0.64 / 0.59 |
| token gain at k = 2.0 | 1.55 / 1.45 | 1.55 / 1.45 | 1.55 / 1.45 | 1.49 / 1.37 | 1.36 / 1.26 | 1.26 / 1.16 | 1.17 / 1.08 | 1.09 / 1.01 | 0.99 / 0.92 | 0.91 / 0.84 | 0.83 / 0.77 |

(Gains without O; O of 6 to 12 ms pulls every gain 3 to 6 % towards 1.)

The probe has no column-kernel row on the A78, so it cannot apply the line by itself. Draft rule in three zones,
with the host-to-A78 transfer of k taken as 2.0 to 3.3 around the host's 2.6 (*estimated*):

- **Pass line: k\* ≥ 3.0 on both models** (R ≥ 1.95; r ≤ 20.2 GB/s at c_b 39.3). Even the top of the transfer
  band leaves a gain; the host kernel gives 1.16 x or more on both. Action: Tom builds the kernel (Y3) and its
  A78 bench row in the next kit.
- **Kill line: k\* < 2.0 on both models** (R < 1.20 on Mistral and < 1.30 on Qwen2.5; r > 32.8 and > 30.2 at
  c_b 39.3; scored on Mistral's line, r > 32.8, for "both"). No kernel inside the transfer band breaks even.
  Y3, Y4 and Y5 close with a device number.
- **Between the lines** (20.2 < r ≤ 32.8): nothing is built. The next kit carries one cell, memo-02's column
  kernel on a pinned A78 at density 1.0, 0.65 and 0.60, hot and cold, and the lead's line is applied to the
  measured k: killed if k > k\*(r) on either model's density, and by my conservative alternative also killed if
  the token gain max(R, 1) ÷ max(b × R, k × d) is under 1.15 on both models.
- One model above a line and one below: reported per model; the lever stays open on the model that passes
  (COMMON_RULES, generality), and the result is "did not generalise".

My probabilities: pass zone 0.15, kill zone 0.10, between 0.75.

### Rule B: reopening R20 (a drafted second token in one batch of two)

Model (*estimated*, memo-03 Bonnie §11): a batch of two costs max(B ÷ r, 2 B ÷ c_b), so its gain over one token
is (1 + a) × min(1, R ÷ 2) less the drafter's time (taken as 5 % of the step), a = the chance the drafted token is
the greedy one. Break-even at a = 0.5 is R = 1.40, which is r = 28 GB/s at c_b 39.3: the lead's threshold.

- **Kill line: r ≥ 0.71 × c_b (28 GB/s at c_b 39.3).** R20 stays closed; the batch of two cannot pay at a = 0.5.
- **Pass line: r ≤ 0.61 × c_b (24 GB/s at c_b 39.3).** Expected gain at least 1.15 x at a = 0.5. R20 is reopened
  as a **host** item only: (1) the drafted token's greedy-match rate a on the fixed prompt set, per model, with a
  drafter that is not a second publisher file unless the owner allows one; (2) greedy identity of a batch of two
  against a batch of one, at `-ub 1` and `-b 512`. Kill inside the reopened item: a < 0.4 on either model.
- **Between** (24 to 28 at c_b 39.3): reopened on paper only; I re-price with the measured r and c_b and the
  lead decides. No code.
- A caution carried from R20 (*device*): on Granite a five-token batch cost 1.8 to 2.1 x the dense time of one
  token at power saving on, where this model says 3.4 x. The dense line was diluted by the draft's small graphs,
  so it does not settle the model; it is the reason item (2) is run before anything is built.

My probabilities: r < 28: 0.80; r ≤ 24: 0.50.

### Rule C: resident capacity

"Held" and C as defined in §3; the better of the two map arms counts, since the exec map is already in the
runtime (R67).

- **Kill line: C < 3.3 GB.** With the designed steps that reads "the 3.5 GB step is not held in either arm"
  (the steps cannot tell 3.0 from 3.3; §7). No Q4_0 body of the class fits (smallest: Yi-1.5-6B, 3.4 GB,
  *estimated* from the header). Strict dense continues only as host research; the phone subject becomes
  Gemma 4 E4B if the owner admits it, and only if its own line below holds.
- **Pass line for the two lead candidates: the 4.5 GB step is held** (Mistral needs 4.04, Qwen2.5 4.12, plus
  about 0.3 GB for the run; the 4.0 GB step is under both). Y1 (kernel layout in a mapped file, no anonymous
  copy) becomes the first build.
- **3.5 held, 4.5 not:** the two lead candidates do not fit at Q4_0 as they are. Open paths: Yi-1.5-6B (3.4 GB)
  if 3.5 is held with the run's 0.3 GB to spare, which a held 3.5 step does not show; a lower-bit served file
  (H2); or sparsity that shrinks the resident set (H5). Reported per model.
- **Gemma 4 E4B line (provisional subject):** the 3.0 GB step held → the 2.61 GB campaign mix fits with margin
  and the published 2.83 to 2.99 GB files are at the edge; 3.0 not held and 2.5 held → only the mix, with no
  margin shown; 2.5 not held → the subject fails on residency.
- Not decided by P1: anonymous or driver-pinned capacity (the stock repack copy; a GPU share of 1.6 GB on a 7B
  body). A CPU + GPU split needs a pinned-memory arm before it is priced for residency.

My probabilities: kill (3.5 not held in either arm) 0.60; lead-candidate pass (4.5 held) 0.07.

### Rule D: headroom for the CPU + GPU split (memo-03 rank 1)

S = `x1gpu sum_gbs` ÷ r_cpu, where r_cpu is the P2 q4x4 cell of the same three cores run alone in the same
session. Price of the split (*estimated*): token = φ B ÷ (S r) + (1 − φ) B ÷ r + J + O, with φ the split share of
the bytes (0.81 if only the FFN products and the head are split, 97 joins on Mistral; 1.0 for all 225), and J =
20 to 100 µs per join (*published* range, memo-03).

| needed S for | FFN + head split (J 1.9 / 5.8 / 9.7 ms) | every product split (J 4.5 / 13.5 / 22.5 ms) |
|---|---|---|
| break-even, r = 24 | 1.01 / 1.04 / 1.08 | 1.03 / 1.09 / 1.15 |
| token gain 1.15 x, r = 24 | 1.22 / 1.27 / 1.32 | 1.20 / 1.28 / 1.37 |
| token gain 1.15 x, r = 19 / r = 28, central J | 1.25 / 1.28 | 1.25 / 1.30 |

- **Kill line: S < 1.15 with the read kernel on the GPU, or `x2 kern=read` < 10 GB/s after one retry with
  `-cl-opt-disable`, or the block prints no number.** Under 1.15 the split is at or under break-even once joins
  are paid. The GPU lane closes for campaign 3 with a device number (Ebony's line, memo-03 draft §5; I agree).
- **Pass line, rank 1 stays open: S ≥ 1.30 with the read kernel, and S ≥ 1.15 × the best X1 stream gain.** The
  combined read must exceed the CPU-only read by **30 %** for a 15 % token gain at central join cost, and it must
  beat the bit-identical multi-stream GEMV (rank 2, same headroom, no numerics change, a tenth of the build) by
  15 %, or rank 2 takes its place. Action: I price the split kernel with the measured rates; nothing is built
  before that.
- **Between (1.15 ≤ S < 1.30):** headroom exists but the split does not pay for its build. Rank 1 is parked; if
  X1's gain is at least 1.15, rank 2 (multi-stream GEMV) is built instead (Ebony's X1 open line); otherwise both
  are parked.
- If S passes with the read kernel but the q4 kernel is under 10 GB/s: "bus headroom shown, GPU kernel not";
  rank 1 stays parked until a GPU kernel reaches 10 GB/s on the phone.
- Bus-limit reading (Ebony's X1 close line, kept): if both (3 A78 + 4 A55) ÷ 3 A78 ≤ 1.10 and S < 1.15, the bus
  is the limit at about r, and ranks 1, 2 and 4 close together.

My probabilities: kill 0.50 (S < 1.15: 0.40; no number or GPU read under 10: 0.2, overlapping); pass 0.25;
between 0.25. P(the untuned q4 kernel is under 10 GB/s) is 0.5 by §5, which is why it parks the lever and does
not kill it.

### What the best outcome of every rule is worth (*estimated*)

Rule D at S = 1.6 and r = 24: 4.04 GB ÷ 38 GB/s + 6 + 9 ms = 121 ms, **8.3 tok/s steady**. Rule A in its pass zone
(r = 20, k = 2.0, Mistral): 4.04 × 0.644 ÷ 20 + 9 = 139 ms, **7.2 tok/s**. Rule B on top at a = 0.5 and R = 2:
1.4 x. No combination of pass results reaches 15 tok/s on a 4.0 GB body; memo-01's verdict does not depend on
this probe. The probe decides which of the open levers are worth a build, and whether any 7B body fits at all.

## 7. Design points to settle before the build (for Tom, through the lead)

1. **P1 steps do not resolve the lines they are scored against.** 3.3 GB lies between 3.0 and 3.5; the lead
   candidates (4.04, 4.12 GB) lie between 4.0 and 4.5. Proposed steps, same count and time: **2.7, 3.0, 3.4, 4.2,
   4.6 GB** (2.7 = Gemma mix and R58's held set; 3.0 = published Gemma files; 3.4 = the 3.3 line and Yi; 4.2 =
   both lead candidates; 4.6 = Llama-3.1-8B and Qwen3-8B). If the steps change, §3's table is read at the nearest
   step and Rule C's lines become "3.4 not held" (kill) and "4.6 held" (pass); P(held) by interpolation: 2.7
   0.80, 3.4 0.40, 4.2 0.10, 4.6 0.04.
2. **P2 has no X1 + 2 A78 cell**, which is the three-thread set every price in memo-01 and memo-03 uses. One more
   2 s cell per kernel.
3. **P2hot's 256 KB slice is not L2-resident on an A55** (128 KB L2, *recalled*) and is half of an A78's L2. Use
   64 KB for the A55, or print the cache sizes and label the A55 line "L3".
4. **A 2 s cell may sit partly inside the bus clock's ramp** (R48 entry: about 0.7 s of slow tokens after idle,
   *device*; cause not shown). Print the rate per 250 ms slice, or run 0.5 s unscored before each cell. If slices
   are printed, I predict the first 0.5 s at 0.75 to 0.95 of the last second, P 0.5 that it is under 0.95.
5. **No column-kernel row.** Rule A's middle zone (P 0.75) needs one more kit only for that row. Adding memo-02's
   kernel on one pinned A78 (density 1.0, 0.65, 0.60; hot and cold; about 30 s) to this kit would let the lead's
   line be applied in one phone window. Prediction if added: k on the A78 at 60 % kept **2.6 (2.0 to 3.3)**.
6. **`x1gpu` should name its GPU kernel.** Run both: read (bus headroom, scored by Rule D) and q4 (what a split
   would use).
7. Smaller: log `/sys/class/devfreq/*/cur_freq` if readable (Ebony's X3), since the bus clock is the unlogged
   variable behind the upside tail of §2; record which three cpus P1's readers ran on.

## 8. Where I may be wrong

- The level of every CPU rate rests on campaign 1's in-situ rates under a power-saving state that was never
  recorded. If power saving held the bus clock down, r is 30 or more and Rules A and B flip to their kill sides.
- The A55 row has no device basis at all.
- P1 has no precedent: no kit has read a file larger than 2.67 GB to steady state, and `lmkd`'s behaviour under
  a foreground reader is a guess. The centre could be off by 0.7 GB either way.
- GPU bands come from other chips and a different compiler generation.
- The in-situ factor 0.87 was measured on the expert op, not on a dense product; the token model is a plain
  maximum, and R27's data says the true curve is rounder (about a quarter of the dense time followed the clock
  while bytes were binding). Near R = 1 to 1.5 the model overstates every gain by up to about 10 %.

## 9. Sources

- Repository: `reports/03_final.md` (R13/R14, R16, R20, R25, R27, R29, R31, R37, R40, R41, R48),
  `reports/04_campaign.md` §1 and §2, `campaigns/02-25toks-5pct/reports/README.md` (R58, R67),
  `campaigns/02-25toks-5pct/STATUS.md` (session log, 2026-11-18, power saving),
  `campaigns/02-25toks-5pct/knowledge.md`, memo-01, memo-02, memo-03 and their work files.
- Web, 2026-11-19 (search summaries; no measured number for this phone found):
  <https://zs3.pages.dev/bandwidth> (a Snapdragon 888 graph, image only, not read);
  <https://gist.github.com/woachk/3fa9916bb12667bb11320ad62bc6c81f> (clpeak, Adreno 690: 57.94 GB/s `float4`);
  <https://www.cpu-monkey.com/en/cpu-qualcomm_snapdragon_888> (51.2 GB/s nominal).
- Published figures quoted through memo-03's drafts (not re-opened today): arXiv 2410.03613v5, arXiv 2501.14794.
