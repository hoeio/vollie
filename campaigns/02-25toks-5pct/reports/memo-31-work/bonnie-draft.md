# Memo 31 working draft (Bonnie): is a 25 tok/s claim round on OLMoE priced well enough to run?

Agent model: claude-fable-5-2

Date 2026-11-18. Author: Bonnie (pricing). Draft for the joint memo-31 with Ebony; not committed by me.
**Nothing here ran on a phone and no host benchmark ran.** Everything is short Python over the stored raw report
`device/r27_2026-11-18.json` (and `r22`, `r23`, `r25`, `r26` for history), plus arithmetic. Scripts:
`memo-31-work/bonnie_table.py` (per-run table), `memo-31-work/bonnie_ppass.py` (pass probability; has a self-check).
Labels: *device* = read or computed from a stored report; *estimated* = arithmetic, resampling or an in-sample fit.
**Every fit and every probability is in-sample, post hoc, from one session (R67), and not pre-registered; the R-log
verdicts stand as written.** I price; the lead decides.

Conversion (*device* identity, memo-29 §1): tok/s = 965.6 / (decode ms per token). 25.0 tok/s = 38.62 ms.
Near 25, 1 ms per token = 0.66 tok/s.

## 0. Answer

1. **P(pass) of a 10-run claim today is about 0.55 if the phone is in R67's clock state, and about 0.4 overall**
   (*estimated*; range 0.2 to 0.55). The lead's judgement "not yet sufficient" is right, with numbers.
2. **The clean OLMoE run sits on the line, not above it:** 7 clean scored runs are 24.98 to 25.56 tok/s, mean
   38.26 ms ± 0.35 (*device*), that is a margin of 0.36 ms (0.24 tok/s) over 38.62. Every non-clean run fails.
3. **Three separate things make a run non-clean in R67, and T1 can now tell them apart** (§2): (a) another
   process on the big cores (3 of 16 scored runs, 17.6 to 19.1 tok/s); (b) a 91 s gate timeout, after which the
   model is still in the page cache but **unmapped from the process** and 165 to 350 MB of the app's memory is in
   swap (5 of 16; pinned ones 23.2, 23.7, 25.1); (c) file pages really missing (2 of 16, both unpinned).
4. **The whole 25 rests on the clock:** R67 is the first of 24 stored raw reports of both campaigns in which the
   big cores ran above 2,227 MHz for a session (*device*). At a 2,150 cap the same build gives 23.2 tok/s (one warm-up, *device*);
   at 2,035 about 22.0 (*estimated*). The cause of the new state is not known. This is the largest risk and it
   is not priced by any bootstrap of R67.
5. **Margin needed** (conditional on the R67 clock state and R67's tail rates, *estimated*): −1.5 ms per token
   for P ≥ 0.8, −2.5 to −3.0 ms for P ≥ 0.9, if only the level moves. If the timeout runs are repaired instead
   (Y1 b), P is 0.80 with no level change and 0.90 with −0.7 ms.
6. **Ranked levers:** (1) keep the clock state (know its cause): worth 5 to 6 ms, all or nothing; (2) `MASS=900`,
   −1.3 ms (−0.7 to −2.0), built, inside the budget; (3) Y1 b extended to "map, not only cache", −1.4 ms on the
   31 to 38 % of runs that follow a timeout; (4) strict pin Y4, −0.3 ms (0 to −0.7); (5) thread priority against
   the contention episodes, unverified.
7. **Recommended next round: a rehearsal, not the claim** (§5): two short sessions in claim format (10
   consecutive plain runs per configuration), one with and one without the deep-sleeping list, with a block that
   puts the animated screen back, `MASS=900` and Y1 b as arms. Go to the claim if the rehearsal's own numbers
   meet a written rule (§5.3).

## 1. The stored runs

OLMoE off, R67 (*device*, `r27`, scored runs; P/p = pin, X/x = exec; ms = decode ms per token; "mapped" = Rss of
the model mapping from `smaps_before`, see §2.2; "rq" = run-queue wait per token, mean of the three workers):

| pos | cell | tok/s | ms | wait s | lost in wait MB | mapped / resident MB | app swap MB | rq ms | decode majors | class |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | PX | 17.63 | 54.8 | 51 | 69 | 2,708 / 2,959 | 95 | 4.8 | 61 | contention |
| 16 | px | 18.89 | 51.1 | 80 | 207 | 2,207 / 2,639 | 200 | 0.4 | 675 | pages missing (188 MB), unpinned |
| 17 | pX | 19.05 | 50.7 | 0 | 12 | 2,505 / 2,615 | 95 | 3.3 | 3 | contention |
| 18 | Px | 23.18 | 41.6 | 91 T | 836 | 492 / 2,448 | 165 | 0.4 | 4,246 | timeout |
| 19 | px | 25.02 | 38.6 | 45 | 0 | 2,583 / 2,673 | 97 | 0 | 10 | clean |
| 20 | Px | 25.37 | 38.1 | 55 | 0 | 2,584 / 2,673 | 96 | 0 | 5 | clean |
| 21 | PX | 25.48 | 37.9 | 0 | 0 | 2,584 / 2,673 | 96 | 0 | 1 | clean |
| 22 | pX | 18.24 | 52.9 | 66 | 418 | 1,439 / 2,522 | 221 | 3.4 | 59 | contention (and unpinned) |
| 27 | Px | 25.07 | 38.5 | 92 T | 409 | 1,479 / 2,570 | 125 | 0 | 51 | timeout |
| 28 | pX | 24.32 | 39.7 | 91 T | 636 | 999 / 2,569 | 262 | 0 | 105 | timeout |
| 29 | px | 25.21 | 38.3 | 35 | 38 | 2,579 / 2,799 | 101 | 0 | 7 | clean |
| 30 | PX | 25.56 | 37.8 | 0 | 0 | 2,578 / 2,787 | 100 | 0 | 1 | clean |
| 31 | pX | 25.08 | 38.5 | 35 | 0 | 2,580 / 2,787 | 99 | 0 | 0 | clean |
| 32 | PX | 24.98 | 38.7 | 0 | 0 | 2,580 / 2,787 | 99 | 0 | 0 | clean |
| 33 | Px | 23.67 | 40.8 | 91 T | 1,324 | 27 / 2,550 | 320 | 0 | 189 | timeout |
| 34 | px | 19.58 | 49.3 | 91 T | 1,609 | 92 / 2,190 | 351 | 0.1 | 614 | timeout + 538 MB missing, unpinned |

Every run: cap `2419/2841` before and after, heat at start 0.63 to 0.77 (*device*).

Clean class (7 runs: gate reached or zero wait, at most 38 MB lost, no run-queue wait): **38.26 ms ± 0.35 (sd),
25.24 tok/s ± 0.23, range 24.98 to 25.56** (*device*). Per-worker CPU time 35.0 to 35.7 ms per token.
Clean runs do not depend on the cell: three are p runs whose workers happened to stay on cpus 4-7.

Granite num, same session (*device*): the 6 runs without contention and with the model mapped are 40.9 to
42.1 ms (22.9 to 23.6 tok/s). Granite needs −2.3 to −3.5 ms more for 25; it is not a claim candidate now. That is a
result, not a reason to drop it.

## 2. What explains the slow runs (T1's first use)

### 2.1 Contention from another process (new; *device*)

In pos 15, 17 and 22 each worker waited 3.3 to 5.8 ms per token in the run queue and took 122 to 508 involuntary
context switches per run (clean runs: 0.00 ms and 0 to 7 switches). Their on-CPU time also rose from 35 to 44-47 ms
per token, which is what a barrier does when one of three workers is descheduled (the other two spin). Missing
pages do not explain these runs (18 to 151 MB missing, 3 to 61 decode majors). The app's own other threads are not
the competitor: process CPU minus worker CPU in the decode is 140 to 270 ms in every run, slow or clean
(*device*). So another process ran on cpus 4-7. All three workers were hit about equally, so the competitor used
more than one big core. The episodes cluster in time (pos 15 to 17, then pos 22 to 24; the profiled run pos 23
has 2,382 worker involuntary switches and pos 36 has 461), and block b3's scored runs have none. Which process:
not visible from inside the app. Rate: 3 of 16 OLMoE scored runs (19 %); on Granite 1 of 16 heavily (pos 3: 9.9 ms
per worker, 9.42 tok/s) and 3 mildly (pos 5, 39, 40: 1.6 to 1.7 ms per worker, all three also unmapped). **This happened with the deep-sleeping bench condition on.**

Pinning does not prevent it: pos 15 is a P run with every worker on cpus 4-7.

### 2.2 After a gate timeout the model is cached but not mapped (new; *device*)

**T1's `smaps` data is usable.** The R-log says "model Rss reads 0". The model row is empty because the kit does
not recognise the internal-storage path; the mapping's numbers are in the `file_rest` row (2,580 MB in clean
runs; 27 to 92 MB when nearly everything is unmapped, so other file mappings are at most about 27 MB). Fix: match
the internal path. No new session is needed to read R67.

| OLMoE run state at start (*device*) | mapped MB | resident (mincore) MB | app swap MB | ms per token |
|---|---|---|---|---|
| clean (7) | 2,578 to 2,584 | 2,673 to 2,799 | 96 to 101 | 37.8 to 38.7 |
| after a 91 s timeout, pinned (pos 18, 27, 33) | 492, 1,479, 27 | 2,448 to 2,570 | 165, 125, 320 | 41.6, 38.5, 40.8 |
| after a timeout, unpinned (pos 28, 34) | 999, 92 | 2,569, 2,190 | 262, 351 | 39.7, 49.3 |

So during a 91 s wait the kernel unmaps 1.1 to 2.5 GB of the model from the process and swaps out 30 to 250 MB
more of the app; the hold's `posix_fadvise` brings the file pages back **into the cache**, not into the page
table. The run then pays minor faults to map them (5,500 to 7,000 decode minors against 3,300 to 4,400) and
swap-ins (pos 18: 4,246 decode majors with 3.6 MB read from storage, so they are swap-ins). This is the closing
rule of memo-29 T1, second branch: "resident ≫ mapped at run start → the hold must go through the mapping".
Fit over the 11 OLMoE scored runs without contention and without missing pages (pos 16 and 34 left out):
1.28 ms per token per GB cached-but-unmapped, r 0.89, intercept 38.01 (*estimated*, in-sample; with pos 16 and
34 in: 1.9 ms per GB, r 0.38). Granite, 12 runs with summed run-queue wait under 1 ms: 0.67 ms per GB, r 0.35:
Granite pays little for unmapped pages (it touches fewer distinct expert pages per
token). **Generalisation: the unmapping happens on both models (Granite mapped 24 to 225 MB after timeouts
against 1,950 to 1,970 clean); the cost shows on OLMoE.**

Penalty of a timeout on a pinned OLMoE run: +3.3, +0.2, +2.5 ms (mean +2.0, n 3, *device*).

### 2.3 Why some waits are zero (it is the gate's first poll, not the exec mapping)

A zero-length wait happens only directly after a run whose own wait ended below the gate (pos 8, 10, 12, 17, 21,
23, 30, 32, 45, 46: 10 of 10, *device*). The headroom forecast lags: the first poll, milliseconds after the
previous run, still reads 0.63 to 0.65, and 5 s later it would read 0.7 to 1.0 (every non-zero wait's series
does). So with r16's gate as built a waited run is often followed by a back-to-back run (not always: pos 20
and 31 waited 55 and 35 s after such runs). The four OLMoE PX
runs sat in that second slot three times; that is the Williams order, not an effect of exec. This settles the
lead's caveat in the R67 entry on the side of "wait length, not the mapping" for the zero waits. It also matters
for a claim: a third to a half of a claim block's runs will be back-to-back runs, which lose no pages and were the
fastest runs of R67 (25.48, 25.56, 24.98; Granite 23.55, 23.06, 23.53, 22.89), provided the clock is not capped.
It is r16's gate unchanged, but the pre-registration should say so in words.

### 2.4 What does not explain the tail

- **The cap:** every OLMoE scored run read `2419/2841` before and after.
- **Heat at start:** contention runs started at 0.64 to 0.65.
- **A55 placement alone:** p runs with a worker on cpus 0-3 are pos 16, 22, 34 (all slow for another reason as
  well); P's measured gain (+1.94 ± 0.69 tok/s, R67) stands, but in this session the three clean p runs equal the
  clean P runs (38.3 to 38.6 against 37.8 to 38.7 ms).

## 3. P(pass)

Rule (CAMPAIGN.md, draft): median of 10 plain runs ≥ 25.0, so the mean of the 5th and 6th run. All *estimated*.

### 3.1 Conditional on R67's clock state (cap 2419/2841) and R67's tail rates

| model | assumption | P(pass) |
|---|---|---|
| bootstrap, the 8 pin-on runs | runs independent, tomorrow = R67 | 0.56 |
| bootstrap, the PX cell (4) | same; n 4, three of them back-to-back runs | 0.61 |
| bootstrap, all 16 | includes unpinned runs: a lower bound for a pinned claim | 0.34 |
| mixture: clean 0.50 (38.26 ± 0.33 ms), timeout 0.31 (drawn from 23.18, 23.67, 25.07), contention 0.19 | class rates from the 16 scored runs, values from pinned runs | 0.53 |

About **0.55**. It is this low although the median of the 8 P runs is 25.03, because (a) only half the runs are
clean, and (b) a clean run is above 25.0 with probability about 0.85, not 1: two of seven are 24.98 and 25.02.
Not modelled: contention comes in episodes (two or three runs in a row), which makes the outcome more
all-or-nothing and pulls P toward 0.5; a claim block starting right after warm-ups starts with a long wait
(pos 15, 27); campaign 1's claim also began with two slow runs (14.09, 18.47, knowledge.md §6).

### 3.2 The clock state is the larger unknown

- History (*device*): of the 24 stored raw reports of both campaigns, `mhz` readings of cpus 4-7 above 2,227 MHz
  exist in r27 (205 of 384 readings) and as 4 and 3 single readings in r5 and r12 (I did not check which phase). In r22 to r26
  the `cap=` field (`scaling_max_freq` of policy 4 and 7, `lab/app-c2/lib/research.dart:284`) read `2035/2035` in
  154 of 192 runs. Campaign 1's 20.51 was measured at 2,112 / 2,035.
- It is not the headroom value: R67 pos 4 to 6 ran uncapped at heat 0.74 to 0.80; r23 was capped at 2,035 at heat
  0.67 to 0.74. It is not battery temperature alone: r26 pos 5 to 12 were capped at 25.5 to 27.1 °C, R67 uncapped
  at 25.8 to 27.6. All board sensors were lower in R67 than in r26's capped runs, but none separates them
  (best AUC 0.95, a modem PA zone; `xo-therm` median 35.0 against 38.5 °C).
- What changed on the same day: kit r27's static dark screen (r22 to r26 drew a spinner every frame for the
  whole session, memo-28 §2.2), no OpenCL/Adreno libraries loaded, the pool, and the deep-sleeping list.
  Also the gate was reached again (heat 0.63 to 0.65; r23 and r26 plateaued at 0.72). A cooler idle phone fits
  all of these; which one caused it is not resolved by one session.
- My probability that the next session with the same kit **and** the bench condition is uncapped: 0.7 (0.5 to
  0.85). Without the bench condition: 0.5 (0.3 to 0.7). These are judgements, not measurements. The argument
  for "not chance": 23 sessions at or below 2,227, then an uncapped one on the day two idle-power causes were removed.
- If capped: at `2150/2150` the one OLMoE datum is warm-up pos 26, 41.6 ms, 23.21 tok/s (*device*); at 2,035,
  scaling worker CPU time by the clock, about 43.8 ms, 22.0 tok/s (*estimated*). **P(pass | capped) ≈ 0**, also
  with every lever of §4 at its centre (22.0 → about 23.5).

### 3.3 Overall

| claim round | P(uncapped) | P(pass given uncapped) | P(pass) |
|---|---|---|---|
| today's build, bench condition on | 0.7 | 0.55 | **0.4** (0.25 to 0.5) |
| today's build, bench condition off | 0.5 | ≤ 0.45 (more timeouts and contention, as R66) | **≤ 0.25** |

## 4. Margin needed, and the levers priced

### 4.1 Margin (conditional on the uncapped state, *estimated*, `bonnie_ppass.py`)

| shift of every run, ms per token | mixture 0.50 / 0.31 / 0.19 | bootstrap of the 8 P runs |
|---|---|---|
| 0 | 0.53 | 0.56 |
| −0.5 | 0.64 | 0.70 |
| −1.0 | 0.70 | 0.74 |
| −1.5 | 0.82 | 0.87 |
| −2.0 | 0.84 | 0.87 |
| −3.0 | 0.95 | 0.98 |

**P ≥ 0.8 needs about −1.5 ms; P ≥ 0.9 needs −2.5 to −3.0 ms** (the level at which runs after a timeout pass
too), while 19 % contention runs stay lost. The other axis is the class mix:

| clean / timeout / contention | shift 0 | −0.3 | −0.7 | −1.3 |
|---|---|---|---|---|
| 0.50 / 0.31 / 0.19 (R67) | 0.53 | | 0.64 | 0.79 |
| 0.60 / 0.21 / 0.19 (a third of timeout runs repaired) | 0.68 | | | 0.88 |
| 0.69 / 0.12 / 0.19 (60 % repaired) | 0.80 | 0.89 | 0.90 | 0.94 |
| 0.75 / 0.12 / 0.13 (and a third of the contention gone), shift −1.6 | | | | 0.99 |

To make the unconditional P ≥ 0.8, P(uncapped) itself has to be ≥ 0.85 to 0.9: **no lever below does that; only
knowing the cause of the clock state does.**

### 4.2 Levers (ms per token on OLMoE at the uncapped state; bytes where a lever moves bytes; all *estimated* unless marked)

| # | lever | price, centre (band) | acts on | bytes | basis | status / cost |
|---|---|---|---|---|---|---|
| L0 | **clock state 2419/2841 instead of 2035** | −5.5 (−5 to −6.6); 2,150 → uncapped −3.3 (*device*, n 1) | every run | 0 | clean floor 44.9 ms at cap 2,035 (R64/R66, memo-29) against 38.26 here (*device*, confounded with r27's instrument changes); worker CPU 38.8 ms at 2,150 against 35.3 | not a lever we hold; the claim depends on it. Test in §5 |
| L1 | **`MASS=900`** (OLMoE skip) | −1.3 (−0.7 to −2.0); +0.9 tok/s (+0.5 to +1.4) | every run; also reads fewer pages after a timeout | 13.39 % of expert bytes skipped (*host*, R51) | op time −1.9 to −2.3 ms (R64) and −3.9 (R60) *device* at cap 2,035, net −1.1 to −2.0 after the dense excess (memo-29 §1.2), scaled by 38.3 / 44.9 for the clock. Cross-check: 13.39 % of an expert op of about 24 ms is 3.2 ms; realised share 55 to 60 % (memo-29 §6.6) minus 0.4 dense excess = −1.4. tok/s in three rounds: +1.71 ± 0.70, +0.15 ± 1.00, +2.02 ± 1.09 | built; PPL 12.0927 / 12.0976 (*host*, R61) against 12.3003; spends 0.38 of 0.586. Parked by R63 §4 "until the pre-run memory state is under control": lead's reading |
| L2 | **Y1 b, as "map and swap-in before the run"**: at the end of the wait, touch one byte per page of the model **through the mapping** for the pages resident at the last run end, and touch the app's own hot anonymous ranges | −1.4 on a run after a timeout (0 to −3.3); over all runs −0.5; moves 30 to 60 % of timeout runs to clean | 31 to 38 % of runs (timeouts) | maps 1.1 to 2.5 GB that is already cached (no storage read); swaps 30 to 250 MB back in | §2.2: penalty +2.0 ms mean (n 3), 1.3 ms per GB unmapped (r 0.89, n 11) | Tom S to M. Cost per wait: about 640 k page touches, 0.1 to 0.4 s CPU outside the timed window (*estimated*; the hold already spends 1.4 to 2.6 s after a timeout, *device*). Bit-identical. Generic: the unmapping is on both models |
| L3 | **strict pin Y4** (`VOLLIE_CPUMASK=0x80,0x10,0x20` or similar; thread 0 already sits on cpu 7 in 28 to 32 of 33 graph starts) | −0.3 (0 to −0.7) clean; unknown sign in a contention episode | clean runs | 0 | clean runs: the two with workers that stayed put (share ≥ 0.94) are 37.8 and 38.5 ms; slope over the 7 clean runs about −1.3 ms per unit of "stay share", weak (n 7) | runtime supports it; app parameter only. Ebony's memo-29 objection (a strictly pinned worker cannot dodge a competitor) is now testable with `c2-sched` |
| L4 | **thread priority for the three workers** (nice below 0) | on a contention run 0 to −10; converts perhaps a third of them; over all runs counts as contention share 0.19 → about 0.13 | 19 % of runs | 0 | §2.1: 10 to 15 ms of run-queue wait per token summed over workers | **not verified that an app may lower nice**: the Android docs say `setThreadPriority` throws when the priority is not permitted and that apps should not use −19; one NDK thread reports scheduler calls refused without `CAP_SYS_NICE` (sources below). A 5-line probe in the kit (call `setpriority`, read it back) settles it. Rule 5 is not touched: it is our own thread |
| L5 | identify the competitor | 0 by itself | | | §2.1 | kit: per run, the delta of `/proc/loadavg` and of `/proc/stat` per cpu if readable (it was not for vmstat), and the app's own per-thread `stat`. If unreadable, the owner looks at the running services after a slow run |
| – | exec mapping (X) as it is | 0 on the level (R67: −0.71 ± 1.53 tok/s) | | | §2.3: its zero waits were the gate's pairing | keep on (R67 verdict); its loss-per-second reading stands |
| – | plateau gate / shorter timeout | would remove most timeouts | | | R64: speed not shown | protocol, not runtime; a claim keeps r16's gate unless the owner decides |

Sums for a clean run (*estimated*): 38.26 − 1.3 (L1) − 0.3 (L3) = **36.7 ms, 26.3 tok/s** (band 35.6 to 37.6 ms,
25.7 to 27.1). With L1 + L2 at their centres the conditional P(pass) is about 0.94; with L1 alone 0.79; with L2
alone 0.80.

Second model (rule: each lever on two models, reported per model): L2, L3, L4 are model-generic and go on
Granite in the same session. L1 is OLMoE's setting of the skip rule that Granite already runs (skip 130, pass in
R62 and R64). Granite at its clean 41.4 ms with L3 is still about 41 ms (23.5 tok/s) (*estimated*).

## 5. Round design

### 5.1 Next round: a rehearsal in claim format (not a claim), kit r28

Purpose: replace my 0.7 and 0.55 by measurements. Two sessions, same kit, same day part if possible:
session A with the deep-sleeping list, session B without it (phone as the owner normally has it, background apps
closed by hand as before R66).

Per session, per model, blocks of **10 consecutive plain runs** exactly as a claim block would be (r16 gate, hold,
kv, pin P, exec X, 3 threads, internal file), in this order (*proposal*):

| block | OLMoE | Granite | decides |
|---|---|---|---|
| 1 | base (off) | base (num) | the claim-format median and its class mix; cap state with and without the bench condition |
| 2 | base + L2 (map and swap-in at wait end) | same | whether runs after a timeout become clean; mapped MB at run start is the mechanism endpoint |
| 3 | `MASS=900` + L2 | base + L2 + L3 (strict pin) | L1's level shift; L3 on Granite (and on OLMoE in session B, swapped) |
| 4 | base, **animated screen back on** (r26's spinner), 6 runs | same | whether the cap returns to 2,035: the cause test for L0 |

Riders: the `smaps` path fix; L4's permission probe (no effect on the runs); L5's counters; first poll of the
gate logged with the time since the previous run's end. About 46 runs per model per session; if that is too
long, block 2 is dropped on Granite first (the mechanism endpoint needs no speed).

Pre-registered predictions I would write (all *estimated*): block 1 OLMoE median 24.9 tok/s (23.0 to 25.4) when
uncapped, 22.0 (21 to 23) when capped; cap `2419/2841` in session A with p 0.7 and in session B with p 0.5;
block 4 capped at 2,035 with p 0.5; L2: mapped MB at run start after a timeout ≥ 2,300 (today 27 to 1,479) and
the timeout penalty below +1.0 ms (today +2.0); `MASS=900`: −1.3 ms (−0.7 to −2.0) against block 2's clean runs;
strict pin −0.3 (+0.3 to −0.7).

### 5.2 The claim round itself (to fix before it is run)

- **One attempt.** The claim is the median of the 10 scored plain runs of one pre-named block; no run is
  dropped for being slow, capped or contended. A fail is scored as a fail; a second claim round needs a declared
  change of the build or the protocol, not a better day.
- **Which session is the claim:** the owner said the claim must also hold without the bench condition. Proposal:
  the claim session is the one **without** the deep-sleeping list; a session with it is reported beside it as
  the bench result. (If the lead prefers the reverse, the "without" session still has to pass for the goal.)
- **N and positions:** N = 10 per model. Both models run a claim-format block in the same session and both are
  reported; only a model whose pre-registered prediction is ≥ 25 is called a claim (today that can only be
  OLMoE). Order fixed beforehand; my proposal G then O, with the lead saying in the file that the later block
  starts on a cooler phone (R67's first block began at cap 1,555 after the storage hash). Two warm-ups before
  each block as in r26/r27, not scored, as campaign 1 did.
- **Fixed beforehand:** APK and library hashes; the env of every run; pin, exec, kv, hold; the numerics setting
  and its host PPL at `-b 512` and `-ub 1` if `MASS=900` is in (12.0927 / 12.0976 ≤ 12.3003, and the head is
  not claimed); the gate (r16, 0.65, 90 s) with the first-poll behaviour of §2.3 written down; void rules
  (hashes, charger) only; text identity against R67's `f8a9fdc4` for off (a new reference text for the num arm).
- **Recorded, never used to exclude:** cap before and after, heat series, mapped / resident / swap at start,
  `c2-sched`, class of each run.
- **Not allowed:** choosing the block after seeing it, a best-of, restarting a block.

### 5.3 A written go rule for the claim (proposal for the lead)

Go when, in the rehearsal, the configuration to be claimed has, **in the session without the bench condition**:
median of its 10 runs ≥ 25.7 tok/s (37.6 ms, a 1 ms margin), at least 7 of 10 runs ≥ 25.0, and cap `2419/2841`
in all 10; and the same configuration in the other session has median ≥ 25.0. With those met I would put
P(pass) at 0.8 or higher (*estimated*: the mixture gives 0.88 to 0.94 for 0.6 to 0.7 clean with −1.3 ms, times
about 0.9 for a clock state seen in three sessions out of three). If block 1 is capped in either session, the
answer for now is "25 is reached only in a clock state we do not control", said with these numbers, and the
research goes to the cap's cause.

## 6. Open, unverified, and things I would not sign

- The clock state's cause (§3.2). The 0.7 and 0.5 are judgements.
- The competitor of §2.1; whether it is tied to the first r27 session (6.2 GB copied and hashed 15 to 25 minutes
  before block b2) and would not recur.
- That an app may give its threads a negative nice on this phone (L4).
- L1's price is carried from sessions at cap 2,035 and scaled by the clock; the dense excess of the skip arm
  (memo-29 §1.2) is assumed to scale as well.
- L2: that touching through the mapping keeps the pages mapped until the run starts (the touch must be the last
  thing before the run; R56's keep-warm lost on timing). The 0.6 repair share is a guess from three runs.
- All class rates are from 16 runs of one session.

## 7. Sources

- Repository: `device/r27_2026-11-18.json` (and r22, r23, r25, r26; `reports/device/r1` to `r16` for the clock
  history); R62, R64, R66, R67 entries; memo-28, memo-29, memo-30; `lab/app-c2/lib/research.dart:284` (`cap=`).
- Android `Process.setThreadPriority` (throws if the priority is not permitted; URGENT_AUDIO "should not
  normally" be used by apps): https://developer.android.com/reference/android/os/Process (search summary,
  2026-11-18).
- NDK list, scheduler calls from an unprivileged APK refused without `CAP_SYS_NICE`:
  https://groups.google.com/g/android-ndk/c/PoGdYmAUbho (search summary; about `pthread_setschedparam`, not
  `setpriority`).
- ART thread priorities: https://android.googlesource.com/platform/art/+/android17-release/runtime/thread_priorities.md
- Snapdragon 888 nominal clocks 2.42 GHz (A78) and 2.84 GHz (X1):
  https://www.anandtech.com/show/16463/snapdragon-888-vs-exynos-2100-galaxy-s21-ultra/4 . A search for a
  documented Samsung cap at 2,035 MHz found nothing (2026-11-18).

## 8. Review of Ebony's draft (`ebony-draft.md`, read after §0 to §7 were written)

**Where we agree, found independently:** R67's 25 is a clock result and a capped session fails whatever else is
done; the tail is contention plus pages; `smaps`' model row is in `file_rest` and T1 is readable now; P(pass |
uncapped) 0.35 to 0.6 (hers) against 0.55 (mine); pin does not prevent contention; next round is a rehearsal in
claim layout, twice (with and without the deep-sleeping list), not the claim.

**What I take from her and correct in mine:**
1. **Ask the owner about power saving before anything is built.** Her candidate "limit CPU speed to 70 %" gives
   0.7 × 2,841 = 1,989 → next X1 step 2,035, and one value for both clusters; the owner was in the battery
   settings on 2026-11-18 (deep-sleeping list). I had not thought of it. If the answer is "yes, it was switched
   off", my P(uncapped) of 0.7 / 0.5 becomes about 0.9 in both sessions and the overall P(pass) today about 0.5.
   Against it, as she says: 2,150 appears in r23 and r26. It costs one question.
2. **Contention follows a model switch** (her §1.2: runs 3 to 5 after a load that replaced the other model, in
   three of three switches; none in b3's scored runs). I listed the clustering but not the switch. Consequence
   for my §5.2: my proposed order "G then O" puts the claim block right after a switch. I withdraw that order;
   her T-C (layout O G O G, score contention by position after a load) should decide it, written before the
   session.
3. Her T-A cap probe (six minutes, no model, spinner on / off, load on / off) is cheaper than my block 4 and
   separates more candidates. Replace my block 4 by T-A.

**Where I disagree, with numbers:**
1. **"The claim is decided by q and by the cap, not by the clean level" (her §4.1) is too strong.** Her two-part
   model puts every tail run at 20.0 tok/s. Three of the four slow pin-on runs are not there: 23.18, 23.67
   (and 25.07 passes). With the tail drawn from the real values, −1.3 ms moves P from 0.53 to 0.79 and −3.0 to
   0.95 (§4.1), because the median of 10 is the mean of the 5th and 6th run and a 24.6 next to a 26.0 passes.
   Margin and tail rate are both worth buying; `MASS=900` should be an arm of the rehearsal, not "only after
   that".
2. **"Resident-but-unmapped is cheap in the decode; the decode cost follows the missing MB" (her §1.4).** Among
   the four timeout runs without missing-page trouble the missing MB does not order the penalty (pos 27: 251 MB
   missing, +0.2 ms; pos 18: 150 MB, +3.3 ms), while unmapped GB does (1.1, 1.6, 2.0, 2.5 GB → +0.2, +1.4, +3.3,
   +2.5 ms; 1.28 ms per GB, r 0.89 over 11 runs, in-sample). n is 4 and the two are confounded with swap (125 to
   320 MB), so neither reading is shown. No design consequence: her L1 and my L2 both touch through the mapping
   and fix either; the endpoint should record mapped, missing and swap so the next session separates them.
3. **Touch every 10 to 15 s (hers) against once at the end of the wait (mine).** R56's keep-warm with 10 s passes
   lost. A pass over 640 k pages on a little core is several times slower than on a big one (*estimated* 0.3 to
   1.2 s), so 6 to 9 passes are 2 to 10 s of A55 CPU per wait, which can keep the headroom above 0.65 and lengthen
   the very waits it protects. I would run both as levels (end-only; periodic + end) and score the wait length
   as well as the loss.
4. **Her rehearsal puts a 2 × 2 inside the 10-run blocks.** Then no block is a claim-format sample of one
   configuration (2 or 3 runs per cell), and the go rule "at most 2 of 10 below 25.0" has nothing to be counted
   on. I would keep whole blocks of one configuration (§5.1), and accept fewer factors per session.
5. **Go rule.** Hers: at most 2 of 10 below 25.0 in both rehearsals, 9 of 10 uncapped. Mine: median ≥ 25.7 and
   7 of 10 at or above 25.0 in the session without the bench condition. Hers has no margin term; at today's
   clean level two of seven clean runs are within 0.02 of the line, so "at most 2 below" can fail on clean runs
   alone. Proposal for the merged memo: both conditions (at most 3 of 10 below 25.0 **and** median ≥ 25.5).
6. **Her L5 (quiet gate)** selects the phone state by a second gate; I would not carry it toward a claim at all
   (COMMON_RULES 5 in spirit, and campaign 1 had no such gate). She already marks it as the owner's.

**Not checked by me:** her `init.rc` quote (she flags it herself), the llama.cpp PR states, MobileMoE.
