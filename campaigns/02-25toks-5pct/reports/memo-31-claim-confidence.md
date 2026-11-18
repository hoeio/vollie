# memo-31: is a 25 tok/s claim round on OLMoE ready? (joint round, Bonnie and Ebony)

Date: 2026-11-18. Authors: Bonnie (`claude-fable-5-2`) and Ebony (`claude-fable-5-2`), each with a review of the
other; merged and decided by the lead (`claude-opus-5-5`). Drafts, reviews and scripts:
[memo-31-work/](memo-31-work/) (`bonnie-draft.md` §8 and `ebony-draft.md` §7 are the cross-reviews). Not an
R-entry. Every number is from R67 (`device/r27_2026-11-18.json`, *device*) or *estimated* from it; nothing new ran
on a phone.

## 1. Verdict

**No claim round yet.** The owner asked for one only with reasonable confidence. Both researchers put the chance
that a 10-run claim at today's state passes well below that (*estimated*):

| | Bonnie | Ebony |
|---|---|---|
| P(pass), given R67's uncapped clock (2419/2841) | about 0.55 | 0.35 to 0.6 |
| P(uncapped session) | 0.7 (A, bench) / 0.5 (B, without) | about 0.45 |
| P(pass) overall | about 0.4 with the bench condition, at most 0.25 without | not given separately; lower |
| P(pass), capped at 2,035 | about 0 (about 22 tok/s) | about 0 (21 to 22 tok/s) |

The next round is a **rehearsal in claim format** (R68, kit r28), not a claim.

## 2. What the stored runs say (*device*, R67, post hoc)

- **Clean runs sit on the line.** 7 of 16 OLMoE scored runs are clean: 24.98 to 25.56 tok/s, 38.26 ± 0.35
  ms/token, against 38.62 ms needed for 25.0.
- **Tail type A, CPU taken from the workers** (3 of 16; 17.6 to 19.1 tok/s). 315 to 461 ms of run-queue wait per
  decode and 700 to 1,100 involuntary switches, almost no faults. It is not the app's own threads, exec, pin or
  heat. All three sat in runs 3 to 5 after a model switch. Who preempts is unknown.
- **Tail type B, after an 80 to 91 s gate timeout** (5 of 16; pinned 23.2, 23.7, 25.1). 150 to 540 MB missing at
  run start. The model is cached but unmapped, and the app is partly swapped. The exec mapping alone did not
  keep idle pages mapped. The zero-length waits in X runs come from the gate's first poll, not from exec
  (Bonnie §2.3), so R67's exec wait-loss verdict stands as written.
- **T1 "smaps reads 0" is a path mismatch, not missing data.** The model's Rss is in `file_rest` and equals
  `RssFile`, so R67 is readable now.
- **Clock cap.** In r13 to r26, cpu 7 never ran above 2,150. Campaign 1's 20.51 tok/s was measured under the
  2,035 cap. R67 is the only stored session at 2419/2841. Recorded temperatures do not explain it: r26 was capped
  at cooler readings than r27 had while uncapped. The cause is unknown. Ebony's candidates (judgement):
  unrecorded-sensor hysteresis 0.30, Samsung power saving's 70 % limit 0.25, removal of the spinner and OpenCL
  0.20, deep-sleeping list 0.10. The runtime cannot raise `scaling_max_freq`, and ADPF cannot either.

## 3. Margin needed (Bonnie, given uncapped, *estimated*)

- P ≥ 0.8 needs −1.5 ms/token. P ≥ 0.9 needs −2.5 to −3.0 ms.
- Repairing 60 % of timeout runs gives 0.80 with no level change, and 0.90 with −0.7 ms.
- `MASS=900` plus Y1 b at their centres give about 0.94. Ebony thinks Bonnie's conditional numbers run high,
  because her price per GB unmapped is confounded with missing pages.

## 4. Lead's decisions on the disagreements

- **P(uncapped):** carried as a range, 0.45 to 0.7. Either end rules out a claim now.
- **Lever comparisons:** levers are compared **within quads, inside a session** (Ebony), not between blocks
  (Bonnie's layout). A claim-format block of 10 plain runs is added only to measure the median and the tail
  mix descriptively.
- **Strict pin Y4:** dropped for now. Bonnie prices it at −0.3 (0 to −0.7) and Ebony at about 0; it is not worth
  a factor.
- **Spinner cause test:** it moves into the cap probe (T-A), not into a block of model runs.
- **Numerics in the claim:** whether the claim configuration includes `MASS=900` is fixed in the claim's
  pre-registration. If it does, it is a declared numerics claim inside the budget (12.0927 / 12.0976 *host*, R61,
  against 12.3003), and the parked status under R63 §4 is lifted only by R68 showing the pre-run memory state
  under control (T1 readable, type B repaired).
- **Go rule for a claim round** (merged from both, fixed now):
  - In the R68 session **without** the deep-sleeping list, at an uncapped clock in all 10 runs, the configuration
    to be claimed must have a median ≥ 25.7 tok/s (a 1 ms margin) and at most 2 of 10 runs below 25.0.
  - In the session with the list, the same configuration must have a median ≥ 25.0.
  - If block 1 is capped in either session, the report says "25 is reached only in a clock state we do not
    control", with these numbers, and the research goes to the cap's cause.
- **The claim itself, when it comes:**
  - One attempt, N = 10 per model, in a pre-named block.
  - No run is dropped for being slow, capped or contended.
  - The order is fixed beforehand, and the claim block does not start right after a model switch.
  - It runs in the session without the bench condition. A fail is a fail.

## 5. TEST list (to Tom; riders in kit r28 unless marked)

| # | test | decides |
|---|---|---|
| T-A | **Cap probe screen, no model.** Logs once per second for six minutes: `scaling_max_freq` and `scaling_cur_freq` (policies 0, 4, 7), headroom, thermal status, power-save flag, battery % and temperature. The spinner alternates on and off every 60 s; the second half adds a three-thread spin on cpus 4 to 7. The owner runs it a few times over two days (morning, evening, after use, with and without the list). | whether an uncapped session can be expected, and whether the app controls it |
| T-B | **Who preempts.** Per run: worker nice, policy and cgroup; `/proc/stat` per-cpu and `/proc/loadavg` deltas around the decode if readable; `c2-sched` per token in profiled runs; power-save flag and thermal status per run. | whether priority (try L2) can help type A |
| T-C | **Position after a model switch.** Type A scored by position in the claim-format blocks (O G O G). | whether a claim block must start later after a switch |
| T-D | **smaps fix:** match the model mapping by inode; parser falls back to `RssFile`. | done when model Rss = `RssFile` ± 50 MB |
| T-E | **Does the official benchmark screen animate?** Desk check of the app code. | whether campaign 1's plain run carries the spinner cost |

## 6. TRY list (to Tom; ranked)

| rank | lever | OLMoE, ms/token (*estimated*) | label | design in R68 |
|---|---|---|---|---|
| 1 | **Touches through the exec mapping during the wait (Y1 b).** A helper on a little core reads one byte per page of the wanted set every 10 to 15 s of the gate wait and once at its end. No advice calls. | clean runs 0; after a 90 s wait 0 to −3; q −0.1 to −0.3 | bit-identical | factor in a 2 × 2 per quad, both models |
| 2 | **Worker priority:** `setpriority(tid, −10)` at pool start, value and errno read back; ADPF hint session as a second arm only if cheap. Permission is unverified (one probe settles it). | type-A runs 0 to −13; clean runs 0 | bit-identical | second factor of the 2 × 2, both models |
| 3 | **`MASS=900` on OLMoE** (built; spends 0.38 of 0.586 PPL). | −1.3 (−0.7 to −2.0) | numerics | its own claim-format block on OLMoE in each session; Granite runs its own skip setting (num) |
| later | outside-graph crumbs (Y5, Y6), −0.3 to −0.9; quiet gate (changes the protocol; owner's word); hedged chunks (only if L2 closes and type A stays) | | | not in r28 |

Conditions kept on: pin (0010), exec mapping (0011), kv clear, static screen, no OpenCL, internal model copies.

## 7. R68 outline (the pre-registration follows the owner's answers)

- Two sessions with the same kit: A with the deep-sleeping list, B without it (background apps closed by hand).
- Per session and per model:
  1. A claim-format block of 10 plain runs, the configuration as R67's P X cells. It gives the median, the tail
     mix, the cap state and T-C.
  2. Quads of touch × priority (8 against 8 per factor).
- OLMoE also gets a claim-format block with `MASS=900` plus whichever of touch and priority the host checks
  allow.
- Before R68: the owner runs T-A two or three times, and answers whether power saving (70 % CPU limit) or a
  performance profile changed between r26 and r27.

## 8. What generalised and what did not (R67 re-read)

- **Generalised:** gate-timeout unmapping (type B) is on both models. Touch and priority are model-generic and
  are tested on both.
- **Did not:** the 25 line is OLMoE only. Granite's clean runs are about 41 ms/token (about 23.5 tok/s,
  *estimated* with any lever above), and that is reported, not dropped.
