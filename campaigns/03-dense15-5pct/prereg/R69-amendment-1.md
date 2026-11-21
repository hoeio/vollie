# R69 amendment 1: decision rules under campaign 3 Amendment 1 (12 tok/s, three reselected models)

Written and committed **before any R69 data exists** (2026-11-21: no c3 probe JSON in `reports/device/`, kit r29
not yet run). The original pre-registration (`R69-probe-read-rate-capacity-gpu-device.md`, commit `458102c`) is
not edited. If the owner's R69 run turns out to predate the push of this file, this amendment is void and R69 is
scored as originally written; any re-reading under the new goal is then a dated post-hoc note, labelled as such.

- **Date / author:** 2026-11-21, clyde (lead), agent model `claude-opus-5-5`. Other agents' models as in the
  original.
- **Trigger:** CAMPAIGN.md Amendment 1 (owner, 2026-11-21): goal 12 tok/s on each of exactly three dense
  7B-class models of different families, all three must pass; subjects reselected (memo-06).
- **Why an amendment is needed:** R69 loads no model, so its measurements and bands are still valid. But Rules A,
  B and C, and §6, name the two old subjects (Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct), their kept
  fractions and the 15 tok/s goal. The three new subjects may be chosen after R69 runs.

## What does not change

- The build, the APK hash, the device state, the void rules, §1 (what is measured), §2 (no budget spent) and
  every prediction in §3. They are scored as written.
- Rule D (CPU + GPU split, multi-stream GEMV, bus limit). It depends only on S, the X1 gain and the A55 ratio.
- Rule B's kill and reopen lines (r ≥ 0.71 × c_b; r ≤ 0.61 × c_b). They depend only on r and c_b.
- Rule C's steps and thresholds (3.4 GB and 4.6 GB, the "held" definition, the better arm counts).

## What changes

1. **Rule A, scoring at R69.** R69 scores Rule A at two **reference densities**, d = 0.60 and d = 0.65 (the
   *host* frontiers of Mistral and Qwen2.5, memo-02 H3). The formula, the 1.15 gain line and the kill line
   k > k\*(r) are unchanged. The result is reported as "column kernel at d 0.60 / d 0.65". It is per model for
   Mistral or Qwen2.5 only if that model is chosen as a subject.
2. **Rule A, per new subject (derived later, formula fixed now).** For each chosen subject s whose served file is
   Q4_0:
   - d_s is its kept fraction from an H3 screen on that served file. It must fall inside the +5 % margin at both
     `-b 512` and `-ub 1`.
   - k_s is R69's P2col column-kernel k at the nearest measured kept fraction: 0.60 if d_s ≤ 0.625, else 0.65.
   - The pass, kill and "open" verdicts follow Rule A with R69's r and c_b.
   - If d_s < 0.55 or d_s > 0.70, the verdict is "not scored from R69" and needs a device row at d_s.
   - If the served file is not Q4_0, Rule A does not apply: R69 measured only the Q4_0 kernel. The lever is
     re-priced for that file's kernel.
   - The derived verdict is written in that subject's freeze record, before its first device round.
   - The generality clause stays: one subject passing and another failing is "did not generalise", and the lever
     stays open on the subjects that pass.
3. **Rule B, inside the reopened host item.** "Killed inside if the match rate is under 0.4 on either model"
   becomes "**on any of the three subjects**".
4. **Rule C, consequences.** The steps are scored as written. "The two lead candidates" is replaced by "a Q4_0
   body of 6.0 B or more multiplied weights (about 3.4 GB or more)".
   - **C < 3.4 GB:** no such Q4_0 body is resident.
     - Each subject's served file is then chosen with its resident estimate (file + KV at ctx 512 + compute
       buffers) at most C. This adds a condition to memo-06 §1 eligibility item 4; it does not replace it.
     - Strict dense continues on the device with such files, not only as host research. This replaces the old
       consequence, which was written for fixed Q4_0 subjects.
   - **4.6 GB held:** Y1 (kernel layout in a mapped file) is the first build, as written.
   - **In between:** served files are priced against C, as written. "Reserves Yi-1.5-6B" is replaced by "the
     reserves recorded in memo-06 §6".
5. **§5 and §6, goal.** Every "15 tok/s" is read as "12 tok/s on each subject".
   - §5's statement (best outcome under 10 tok/s on a 4.0 GB body, *estimated*) stands.
   - §6's "15 tok/s is reported as not reachable on this hardware" becomes: "12 tok/s is reported as not reachable
     for every subject whose priced token time at the measured r and c_b exceeds 83.3 ms (memo-06 §5 lines),
     with the measured r, c_b and C".
6. **New derived quantities, reported with the result (no lever decision attached).** They are computed from
   R69's numbers and labelled *estimated* (from *device* inputs):
   - the 12 tok/s **byte line** B12 = r × (83.3 − O) ms;
   - the 12 tok/s **weight line** W12 = (c_b ÷ 0.5625 bytes per weight) × (83.3 − O) ms;
   - both at O = 6, 8 and 12 ms. These feed memo-06 §5's kill line K0.

## Owner action

None beyond the original: run kit r29 as written. As in the original, there is no tag (owner's wish); the commit
hash and push time of this file show that it came before the data.
