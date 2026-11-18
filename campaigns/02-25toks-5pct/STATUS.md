# Campaign 2 status and hand-off

**Read this first, update it last.** Every session appends a line to the session log below and refreshes the
sections above it. Last updated: 2026-11-18 (retroactive prereg tags recorded).

## State in one paragraph

**As of 2026-11-16 (R62, *device*):** Granite skip 130 + head2 passes (+2.37 ± 0.69 tok/s; 17.65 tok/s at three threads) and is frozen; OLMoE's num arm is not shown in two rounds and its repaired head saves nothing on the phone; four threads lose on both models; every gate wait timed out in R60 and R62 and no cell is near 25. The state before a run (heat, pages) decides the number.

**As of 2026-11-15 (R60, *device*):** page hold through the gate wait works on both models (+4.7 tok/s OLMoE, +5.3 Granite, pass) but costs 5 s of CPU per wait; Granite's skip + head is +1.44 ± 1.28 tok/s over all quads (unresolved; +2.4 in both hold quads); OLMoE's num factor is void because the untied head did not run on the phone (skip alone +1.71 ± 0.70). Best state of the round: Granite 18.96, OLMoE 19.25 tok/s (num with hold). Not near 25.

**As of 2026-11-14 evening (R52 to R58, *device*):** the paragraph below is the state before the device rounds. Best measured state: OLMoE 19.70 tok/s median (21.03 when resident before the run), Granite-Mi 17.19, both cool, both without a numerics change (R58). Speed is decided by page residency and heat, not by kernels (R55 to R58).

Campaign 2 asks whether 25 tok/s is reachable on the Snapdragon 888 phone with a frozen 7B-class model and a
+5 % perplexity budget. **The first measurements exist (R50, *host*): one served file is frozen for OLMoE and
one for stock Granite**, both the campaign-1-style mix with an imatrix (cards in `models/`). Budgets: OLMoE
12.3003 (served 11.7146, F16 10.9420), Granite 9.6879 (served 9.2266, BF16 8.7816). LFM2.5's screen is finished (R50 addendum): the rule names the mix without imatrix (28.9005), but its BF16
reference (32.9649) is worse than every quantized file and is not credible, so no ceiling is established, no card
is frozen, and by memo-03's stop-rule reading no phone round is planned for it. **No device run and no device pre-registration exist.** The chosen Granite file
has the same types and bytes per token as campaign 1's (806 MB), so campaign 1's costing carries over
(*estimated*); OLMoE reads 590 MB per token with 16 layers and no recurrent state, but two campaign-1 levers do
not transfer as they stand: the two-stage head does nothing on OLMoE and expert skip has a different meaning
there ([reports/memo-02-lever-applicability.md](reports/memo-02-lever-applicability.md)).

## Decisions already made by the owner

- Goal: *"On the SD888, with the 7B model untouched, can runtime optimisation reach 25 tok/s within at most
  +5 % perplexity?"* Campaign 1's answer (yes at 20 tok/s, +2 %) is taken as settled.
- The project is expected to be very long; thoroughness over speed.
- Prefer models people know, as far as feasibility allows.
- Repository: one repository, campaign structure (campaign 1 frozen, campaign 2 added beside it).
- Dense 7B models are excluded because the bandwidth and compute arithmetic rules them out
  ([feasibility.md](feasibility.md) §1), not by preference.
- **2026-11-12, session 3** (full text in [CAMPAIGN.md](CAMPAIGN.md), "Owner decisions"):
  1. Model set (a): Granite-4.0-H-Tiny, OLMoE-1B-7B, LFM2.5-8B-A1B.
  2. "7B-class" = total parameters of about 7 to 9 B.
  3. Metric: the benchmark's definition, unchanged; no second metric.
  4. Budget: one total +5 % over the served file's stock PPL; hard ceiling F16 × 1.20.
  5. **Stock weights, not abliterated.** Campaign 1's Granite file (d6-e6) is therefore not a campaign-2 file
     and its PPL table and budget value (10.0854) do not apply; a stock-weights Granite file is measured anew.
  6. Served file per model is fixed after a host screen (publisher Q4_0 against a campaign-1-style mix), by a
     rule written before the screen (memo-01, part C).
  7. Base or instruct: no owner preference; the lead chose Instruct (OLMoE `0125-Instruct`).
  8. LFM2.5 stays for research use only (licence "lfm1.0" is not open source), third in priority.

## Open decisions (need the owner)

None at the moment. (2026-11-15, answered: option (a). R61, *host*: on a KleidiAI build stock is OLMoE
11.7132 / 11.7159 and Granite 9.2795 / 9.2732 (`-b 512` / `-ub 1`); both num settings are inside the existing
budgets and inside same-build stock × 1.05. Budgets unchanged: 12.3003 and 9.6879.)

Hosting (2026-11-13): the two mixes are in the public repository `HoeioUser/vollie2` at commit `e893e8a6`
(server-side sha256 checked); the publisher's Q4_0 is taken from AI2's repository at its pinned commit. Tags: the
owner does not want them; the order of pre-registration and run is shown by commit hashes and push times
(R51: `ab7875e`, amendment 1: `e16630b`).

Answered 2026-11-13: push to `main` directly, no PR; the publisher's Q4_0 is a second OLMoE served file
(CAMPAIGN.md, decisions 9 and 10); the contact e-mail and the private path were removed from campaign-1 files.

## Next actions, in order

**Waiting on the owner (2026-11-16):** kit r24 handed over; R63 pre-registered
([prereg](prereg/R63-gate-rule-and-numerics-two-models-device.md), kit and fault analysis in
[memo-23](reports/memo-23-kit-r24.md)). APK `lab/dist-c2/vollie-c2-1.24.0+24-research-r24.apk` (sha256
`6a655ca7…ffa5beba`). When the report comes back: `python3 -I tools/c2_research_parse.py <json>`, score as
written, store the raw report. New reading to carry: most major faults of a gated run look like zram swap-ins
of the app's own 0.5 to 0.7 GB of anonymous memory (memo-23 §1, inferred; R63 tests it); if confirmed, that is
the next lever. Not started: memo-21 item 2 (routing prior simulation).

**Latest (R62, kit r23, 2026-11-16, *device*; [R-log entry](reports/README.md)):** Granite num passes
(+2.37 ± 0.69 tok/s, 17.65 against 15.95 at three threads) and its setting is frozen. OLMoE: the head now runs
on the phone (0006 works) but saves nothing measurable, and num minus off is +0.15 ± 1.00, not shown for the
second round. Four plain threads lose on both models (−5.1 and −3.9 tok/s) and are closed. The cheap hold costs
2 s CPU and starts runs at 93 to 94 % of the wanted MB (prediction missed, fail rule not met). All 32 gates
timed out again. Best cells 17.65 (Granite num) and 16.85 (OLMoE off): **not near 25**. Next, in order:
1. **The state before the run is the problem to solve first**, on both models: why the gate (0.65 in 90 s)
   is not reached in R60 and R62 but was in R58 (compare heat series, clocks and caps of r21, r22, r23 and
   campaign 1's r16; ask the owner what the phone state was), and why a run that starts 93 % resident still
   takes thousands of major faults.
   Owner, 2026-11-16: a charger was probably plugged in at about minute 45; the report shows `plugged` from
   pos 40 (Granite's last block, cap 1,555 from pos 42). Blocks 1 to 3 were unplugged and timed out too.
   Granite's four-thread kill leans on that block (dated note in the R62 entry). Next kit: pause or void on
   `plugged`.
   Owner, 2026-11-16: background apps were closed in R62, so neither the banking app nor the charger explains
   the timeouts. **Reading of the heat series (lead, 2026-11-16, *device* data, not pre-registered):** in R60
   and R62 the headroom falls to a plateau of about 0.72 within about 40 s of a wait and stays there (median of
   the minimum 0.722 and 0.719; lowest value of either session 0.665); on 2026-11-14 (r16 control, r20, r21,
   evening) the gate at 0.65 was reached in a median of 35 s or at once. The clock is uncapped at 0.72 (2,112 /
   2,035 MHz). So on these two mornings the gate waited 50 s past the plateau for nothing and lost pages
   meanwhile (median lost per wait 240 MB with r22's hold, 683 MB in r23). Candidate cause: the phone's idle
   headroom floor differs by place or time of day (ask the owner). Candidate fix for research kits: end the
   wait when the headroom has stopped falling (plateau rule) or at 0.65, whichever comes first; a claim round
   keeps r16's gate unless the owner decides otherwise.
2. OLMoE: why the skip arm takes more decode faults than off (5,310 against 1,139 median); whether `MASS=900`
   is worth keeping; the untied head is kept out of the next kit's claims (about 0 ms on the phone).
3. memo-21 item 2 (routing prior, host trace simulation): **not started yet**; item 1 only as the
   barrier-free helper, microbench first.
4. Say to the owner with numbers where the goal stands (R58 best 19.70 / 17.19 without numerics; R60 and R62
   below that; fault-free arithmetic in memo-21: OLMoE about 24.4, Granite 23 to 24 tok/s at best).

**Latest (R60, kit r22, 2026-11-15, *device*; [R-log entry](reports/README.md)):** every gate wait ran the full 90 s. Hold: pass on both
models (resident before the run +842 MB OLMoE, +1,165 MB Granite; tok/s +4.72 ± 1.70 and +5.28 ± 0.98), cost
predictions missed (5.2 s CPU per wait, 20 repeated read-backs). num: Granite unresolved over four quads
(+1.44 ± 1.28), +2.3 and +2.5 tok/s in the two hold quads, skip counter 39.6 % as predicted; OLMoE **void**: no
`c2-head-run` line, head op time unchanged, so `VOLLIE_HEAD2` did not act in the `armv8.2_2` module. Level
predictions missed low. Next, in order:
A. **Tom:** why the untied head (`patches/llama.cpp-c1-olmoe/0004`) does not run on the phone build (first
   candidate: the Q6_K head's layout in the armv8.2 module is not one of plain / 8x4 / 8x8, so the patch
   declines); fix in a new patch, keep identity with the knob unset, and make the kit print a head line even
   when the head declined (with the reason).
   **Cause found 2026-11-15 (Tom, *host*, reproduced):** the phone module is built with KleidiAI
   (`android/build_cpu.sh`, `GGML_CPU_KLEIDIAI=ON`, as in campaign 1); KleidiAI takes OLMoE's Q6_K `output.weight`
   into its own buffer (int8, one scale per row), so 0004 declines every head op. Granite's tied head is not
   taken. Every host PPL so far (R50, R51, R53, R59, the budgets) was run on builds **without** KleidiAI, so
   **the phone's stock numerics have never been measured on the host**: with KleidiAI the per-chunk values of
   the first 4 chunks move by up to about 0.5 % (OLMoE; Q4_0 attention tensors and the head). Patch
   `patches/llama.cpp-c1-olmoe/0006-c2-head2-kleidiai.patch` (uncalibrated on the phone): `VOLLIE_HEAD2_REPACK=1`
   at model load keeps the head out of KleidiAI (this changes the off arm's head numerics back to Q6_K, a
   declared change), and the head line now reports declines with a reason. Identity with knobs unset on 4
   chunks at `-b 512`, `-ub 32`, `-ub 1`, with and without KleidiAI. **Open decision for the owner below.**
   **R61 done (2026-11-15, *host*):** the settings for the next kit stand: OLMoE `VOLLIE_HEAD2_REPACK=1` at
   load in both arms + `MASS=900` + head 1024; Granite skip 130 + head2 2048. Still to do for A: the app's load
   setting for `VOLLIE_HEAD2_REPACK`, declared in the next pre-registration as a change of the off arm.
B. **Hold, cheaper:** one read-back shortly before the run (or far fewer passes) instead of one per poll;
   target under 1 s CPU. Then hold is part of the protocol in both arms, not a factor.
C. **Pre-register R61:** num against off on both models with hold on in all quads (the hold quads of R60 are
   the prior: Granite about +2.4 tok/s, OLMoE skip alone about +2.4 ± 1.2), plus the repaired OLMoE head.
D. **Answered 2026-11-15 (owner):** Toss was running in the background during R60 (candidate cause of the
   gate timeouts and page loss, not shown). Kit r22 is not rerun; the next kit (R61) is run with background
   apps closed, and the phone state before the session is written into the pre-registration's state condition.
F. **Unconventional levers (owner, 2026-11-15: explore such methods freely):**
   [memo-21](reports/memo-21-unconventional-levers.md), all gains *estimated*. Shortlist: (1) helper threads on
   the idle cores that take expert row chunks without joining barriers (bit-identical; test: threads 3 against 4
   per run in the next kit); (2) routing bias toward experts the runtime used in the last N tokens (declared
   numerics; host trace simulation and PPL first); (3) LUT kernel for the Q3_K experts, exact or lossy (needs a
   runtime-derived re-laid-out expert file of 2.6 to 2.8 GB: owner decision on storage); (4) helper warming L3
   for the next dense op; (5) thread pinning, `mlock` (the limit is 64 MiB, not 64 KB), overlap after sampling.
   Finding: the expert op is Q3_K kernel time (about 2 times the dense byte rate), so kernel work pays only
   there. Not yet pre-registered; nothing measured.
E. Still open from before: LFM2.5 as third device subject (proposal to the owner); the two host builds'
   different OLMoE `-ub 1` values. The raw kit report is stored (`reports/device/r22_2026-11-15.json`);
   the summary matches it and the repository parser reproduces the scoring.

**Latest (R58, kit r21, 2026-11-14 evening, *device*):** cool phone, `Cached` 2.4 to 3.2 GB (the "high" state;
the low state did not occur). No numerics change on either model. Knob-off medians: **OLMoE 19.70 tok/s**
(resident-before runs 21.03, best 22.28, fault-free tokens about 42 ms), **Granite-Mi 17.19** (tokens 53 to
57 ms, no skip and no head2). `GGML_PREFILL_COLD`: Granite +0.09 ± 0.50, OLMoE +0.93 ± 2.74 tok/s, unresolved
on both; **the lever is closed by R58's kill rule** (read bytes were not measurable, `/proc/self/io` is
unreadable in the app). Slow runs are those after a 45 to 91 s gate wait that loses file pages. Next, in order:
0. **Done 2026-11-15 (R59, *host*):** both levers are on campaign 1's runtime (`patches/llama.cpp-c1-olmoe/0003`, `0004`), identical with knobs unset; inside budget at `-b 512` and `-ub 1`: OLMoE `MASS=900` + head 1024 (12.1053 / 12.0946), Granite skip 130 + head2 (9.5173 / 9.5076); Granite 145 fails at `-ub 1`. Next: the device round (kit r22, R60) with these two settings.
1. **Superseded by item 0 (was: R59 to pre-register):** carry the c2 skip rule (`VOLLIE_SKIP_MASS`, model-generic) and the untied
   two-stage head from `patches/llama.cpp-c2/0001`, `0009` onto campaign 1's runtime as new patches in
   `patches/llama.cpp-c1-olmoe/`; host identity and PPL at `-b 512` and `-ub 1` for OLMoE (`MASS=900`) and for
   Granite-Mi (campaign 1's skip 117 and head2 must be re-screened on this file; its budget is 9.6879); then a
   two-model ABBA kit. *Estimated* OLMoE: 42 ms to about 36 to 38 ms per token when resident.
2. **Keep the pages through the gate wait** (kit or runtime side, measured on both models in the same kit):
   R56's keep-warm was tested only in a low-cache state; here the cache is there and the wait still drops pages.
3. Add a way to read I/O in the app (the report's `read_bytes` is -1); major faults per phase are the fallback.
4. LFM2.5 as third device subject: propose storage (about 10 GB for three files) and run time to the owner.
5. Open check: the two host builds' different OLMoE `-ub 1` values (8.9550 and 8.8302, 4 chunks).

**Owner, 2026-11-14 (after memo-18):** agreed to solve residency on campaign 1's base first. And: campaign 2 has
two purposes, speed and trust through many models. OLMoE is not demoted to a control; every lever is measured
on at least two models in one session and reported per model; LFM2.5 is to be brought back into the device work
when storage and run time allow (a third 3.8 GB file; propose it to the owner with the numbers). The gate for
batch-dependent numerics (both `-b 512` and `-ub 1`) is agreed. Kit r21 (two models, `GGML_PREFILL_COLD`) is
being built by Tom; pre-register it as R58 from memo-18's predictions before handing it over.

**Latest (memo-18, 2026-11-14, *host* traces and an in-sample simulator, tok/s *estimated*):** corrected working
sets: Granite 1.97 to 1.99 GB (decode 1.11 GB), OLMoE 2.67 GB (decode 2.16 GB); memo-13's figures were wrong
(tracer bug). At equal page cache Granite is the better model for this phone: about 19.8 tok/s at 2.2 GB of
cache against 11.5 for OLMoE with today's runtime; below about 2.2 GB no residency policy brings OLMoE to 20.
Recommended first lever: `GGML_PREFILL_COLD` (mark rarely routed expert slices cold during the prefill; advice
only, bit-identical, both models), predicted Granite 10.7 to 13.4 and OLMoE 11.6 to 13.2 at the residency measured
on 2026-11-14. Next kit: two models in one session (Granite, OLMoE, OLMoE, Granite blocks), knob alternated,
read bytes split into prefill and decode. Open rules question: a skip rule that behaves differently in batched
prefill and single-token decode should be gated at both `-b 512` and `-ub 1`.

**Latest (R57, same-day control, 2026-11-14):** campaign 1's own app and kit, unchanged, printed a median of
**11.46 tok/s** on Granite at the claim positions today (20.51 on 2026-11-08), on a cool phone at 2,112 MHz.
OLMoE the same day: 11.25. So OLMoE equals Vollie 1 at equal state, and **Vollie 1's 20.51 is not reproduced
today**: the phone gave the app 1.8 to 2.3 GB of page cache instead of 2.2 to 3.0 GB. The campaign's first
problem is therefore residency under about 2 GB of page cache, for both models; R56's reading that the
shortfall is specific to OLMoE is corrected in the R57 entry. Next: price the ways to shrink or order the
resident set (host traces first), starting from what campaign 1 learned in R9, R12, R23, R29 and R49's cold
runs, and keep measuring both models in one session so the day cancels out.

**Latest (R56, kit r20, 2026-11-14):** with campaign 1's protocol and runtime, on a cool phone at 2,112 MHz, OLMoE
printed a median of 11.25 tok/s at the ten claim positions (campaign 1's Granite: 20.51). Fault-free OLMoE
tokens take about 43 to 51 ms (Granite: about 46), but the first ten tokens of every run fault, because the
phone holds at most 2.39 GB of the 2.73 GB the benchmark touches (Granite needed about 2.0 GB). Keep-warm gave
nothing and is closed. **The open problem is a residency shortfall of about 0.35 to 0.5 GB for OLMoE; kernels
are not the gap.** Candidates, all to be priced before any is built: order of page use after the prefill
(re-read the experts the first generated tokens will need before the timed decode starts), a smaller resident
set (which expert slices this benchmark's decode really needs: 2,138 MB by trace), no repack copy (about 225 MB
of anonymous memory back to the cache), and, as an owner decision only, a smaller expert quantization. Also
still useful: the campaign-1 app's own kit on the same day, as the control for Granite's residency.

**Read first: the owner's standing instruction of 2026-11-14, "start from Vollie 1"** (CLAUDE.md, rules). Memo-16
shows why: campaign 1's own first run in r19's thermal state was 12.27 tok/s; its 20.5 came after the phone had
cooled and the cache had filled, with runs back to back. Kit r19 waited 90 s before every run on a phone that
could not cool, which emptied the cache each time. **Next kit (r20): kit r16's protocol reproduced on OLMoE with
campaign 1's runtime as the baseline row, runs back to back after a warm-up when the gate cannot be met, a
keep-warm lever, mincore; new levers (c2 skip rule, untied head) only as deltas on campaign 1's runtime.** The
earlier "no on OLMoE" reading is withdrawn until OLMoE has been measured in the state campaign 1 was measured in.

00. **R52 is in (R-log): OLMoE on the stock runtime printed a median of 3.34 tok/s (*device*), void by hygiene
   and a factor of four below the band.** Cause (memo-13): the benchmark needs about 2.73 GB of the file resident
   (Granite: 2.57 GB, *host* traces) and the phone had 1.6 to 2.35 GB of page cache with no free swap; thermal
   headroom 0.97 to 1.10 capped the clocks at 1,075 to 1,440 MHz; the kit's cool-down gate did not wait (NaN from
   a too-early second thermal read was taken as cool; r18 shares the bug). Campaign 1's fast Granite runs had 2.7
   to 3.0 GB of page cache: it never made the model resident from the runtime. Kit r18 / R54 is withdrawn
   (R54 amendment 1). **Owner, 2026-11-14: use campaign 1's Vollie as well** (its runtime patch, app code and
   lab are to be reused by copying; the sibling folder itself stays unmodified). Campaign 1's 0001 (+ tiny-op)
   builds, loads `olmoe` and is identical to the base on OLMoE with its shipping settings minus the Granite
   numerics (*host*, 4 chunks, memo-13 §7).
   **R55 is in (kit r19, R-log):** campaign 1's runtime on OLMoE printed medians of 4.43 (knobs off), 7.20
   (shipping settings) and 7.51 tok/s (plus read-ahead guess), *device*. No run was resident: 0.4 to 1.4 GB of
   the file in the page cache before a run, at most 2.18 GB after, 2.73 GB needed; the phone also dropped 0.5 to
   1.4 GB of it during each 90 s wait. The fixed gate waited 10 minutes and never saw headroom below 0.774
   (target 0.65); the clock cap was 1,555 MHz throughout. The cleanest tokens ran in 55 to 73 ms at that cap.
   The raw r19 report exists only as pasted text; a transcribed summary is in `reports/device/`. **The goal
   decision is the owner's** (state-conditioned claim, a smaller file, private expert memory, or "no"); the lead's
   reading is in the session report of 2026-11-14. Granite "no" (memo-08 §6) still to be written into the R-log.
   APKs are hosted under `kits/` in `HoeioUser/vollie2`; every device round is an APK out and a JSON back.
0. **R51 is scored** (R-log): on `olmoe-Mi` the largest skip inside the budget is **13.39 % of expert bytes**
   (`MASS=900`, PPL 12.1031, +3.32 %, *host*); `SUM=75` (17.48 %) is over by 0.0167. Prediction hit (band 6 to
   15 %). In ms (*estimated*, memo-03): about 3.9 ms, about 0.8 ms short of 38.0 ms at the central estimate,
   inside the model's uncertainty. **Next round: device baseline of `olmoe-Mi`** (write `prereg/R52-…`; the app
   needs a build that loads this file; prediction on the exact stack 43.5 ms, band 39.1 to 50.4, memo-03).
   **Amendment 1 is void:** the skip knobs did nothing on the Q4_0 file (probably the repacked Q4_0 path, which
   would also mean the campaign-1 memory problem for an all-Q4_0 file on the phone); cause to be established
   before that file is used for anything.
1. **LFM2.5:** done as far as planned (R50 addendum). Optional: re-run its reference with the publisher's F16
   file to learn whether the BF16 path is at fault (about 8.5 h, *estimated*); only worth it if LFM2.5 comes back.
2. `en.txt` sanity perplexity for the chosen files (the text is in campaign 1's private `lab/text/`).
3. **Expert-skip frontier on the host, OLMoE first** (the largest lever there: experts are 346 of 590 MB per
   token): PPL against skipped expert bytes. The skip rule has to be defined for unnormalised top-8 softmax
   weights first (memo-02); Tom writes it into a new patch series, off by default, and the identity checks
   (`-ub 1`, `-ub 32`, `-b 512`) are run with the knob off. For Granite repeat campaign 1's thresholds on the new
   file (115 to 140), because the old PPL table belongs to other weights.
4. New patch series beside `0001`, split by lever. Memo-02 found six levers in `0001` that are **on by default**;
   in the new series every lever is off unless switched on, and each is checked bit-identical on `olmoe` and
   `lfm2moe` before it is used (none ever ran on those architectures).
5. Bonnie: price the levers per model in bytes and ms (*estimated*) from the cards, memo-02 and campaign 1's
   byte rate (0.0411 ms/MB) and per-node costs; give a prediction band for the stock-runtime and full-stack
   steady token of each file.
6. **Stop rule:** if a model's best package inside its budget is worth less than about −6 ms of what it needs
   (*estimated*), write the "no" with numbers into the R-log and do not spend phone rounds on that model.
7. Otherwise write `prereg/R51-<title>.md` (R50 is the host screen; agent models: clyde `claude-fable-5-2`,
   bonnie and tom `claude-opus-5-5`, annie `claude-sonnet-5-5`), commit it, ask the owner to tag it, then ask for
   the phone run. A baseline device run of each chosen file on the stock runtime is the natural first round.

## Things to be careful about

- The owner's +5 % cap was already opened once in campaign 1 (R41 to R43) and the dense-numerics line closed with
  the cap unspent. Do not re-propose the closed items without new evidence: low-rank head screen, shared-expert
  pruning, a hard vocabulary subset, dynamic channel skipping, sub-4-bit dense containers without a kernel.
  List and numbers: [knowledge.md](knowledge.md) §4.
- Expert skip near the cap raises `en.txt` perplexity much more than wikitext-2 (130: +6.1 % versus +3.67 %).
- Single phone runs vary about ±20 %. Small effects need the in-run ABBA A/B and a resolvable σ.
- The first run after memory pressure is slow (cold start, page residency); the claim is about the median of
  warm, gated runs.
- Do not claim a result from a host measurement or an estimate.
- "Stock" has two meanings here: stock weights (the publisher's model, not abliterated) and stock served PPL
  (the served file run by the unmodified runtime). Write which one is meant.
- A weaker served file gets a looser absolute budget (+5 % of a larger number). The guards are the selection
  rule (served ≤ F16 × 1.08) and the hard ceiling (F16 × 1.20); do not pick a file because its budget is roomy.
- llama.cpp issue #29002 (co-batched sequences corrupt the LFM2 short-conv state on CPU) bears on LFM2.5's
  perplexity and identity checks; see memo-01.
- **Where the session runs matters.** Session 3 ran on the owner's Apple M4 Pro (Remote Control), not in a cloud
  container, although the session text described a cloud environment. Check with `sysctl -n
  machdep.cpu.brand_string` and `df -h` before telling the owner to run host work. With the owner's permission the
  lead ran the host screen itself; phone runs remain the owner's.
- Long host runs: start them detached (`nohup caffeinate -i ...`, in a command of their own) and wait in a
  separate command; a background command is killed after two hours together with everything it started. A 16-bit
  pass takes about 2 h (OLMoE) to 7 h (Granite) and about 8.5 h (LFM2.5, *estimated*). The disk had 18 GB free at the
  start; 16-bit references are deleted after use. Memory pressure from other apps slows runs.
- The screen used an **unpatched** build, on purpose: `0001` switches six levers on by default (memo-02), so a
  patched build is not a stock-PPL reference unless those knobs are set to 0.
- Subagents defined in `.claude/agents/` are only listed if the files exist when the session starts. In session 3
  they appeared late; the first research task ran on a general-purpose stand-in given `bonnie.md`
  (`claude-opus-5-5`). After they loaded, all four stated their pinned IDs.

## Session log

| date | session | what was done |
|---|---|---|
| 2026-11-12 | 1 | Read the repository; analysed feasibility; researched candidate models; created the campaign structure (PR #2, merged); wrote this hand-off (`CLAUDE.md`, `STATUS.md`, `knowledge.md`). No measurements. |
| 2026-11-12 | 2 | Pinned the four-agent team in `.claude/agents/` (full model IDs) and described it in `CLAUDE.md`; added the "Agent models" field to the prereg template; added the campaign 1 Phase 0 model history to `candidates.md`. Pushed to `main` at the owner's instruction. No measurements. |
| 2026-11-12 | 3 | Team check: all four agents state their pinned model IDs (`CLAUDE_CODE_SUBAGENT_MODEL` unset). Owner answered the four open decisions plus file variant, served-file procedure and LFM2.5 licence (see above). Web re-check of the three candidates: all supported at `BASE` `620d2225c`. Wrote memo-01 (candidate re-check, host screen plan with commands, selection rule). No measurements. |
| 2026-11-13 | 3 (continued) | The session turned out to run on the owner's M4 Pro; with permission, freed disk (20 non-Granite campaign-1 experiment files in the private `lab/`), built clean `620d2225c`, ran the host screen: **R50**, served files chosen for OLMoE and stock Granite, cards written; LFM2.5 reference still running. `tools/cost.py` on the chosen files. Tom: memo-02 (lever applicability). First measurements of the campaign, all *host*. |

| 2026-11-13 | 3 (continued) | Memo-03 (lever pricing, *estimated*), memo-04 and `patches/llama.cpp-c2/0001` (OLMoE expert skip, off by default, identical on 4 chunks with the knob unset). Pre-registered R51 (`ab7875e`) and amendment 1. Owner: push to `main` without PR; second OLMoE file (publisher Q4_0); e-mail and private path removed from campaign-1 files. Sweeps queued behind the LFM2.5 reference run. |
| 2026-11-13 | 3 (continued) | Served files hosted at `HoeioUser/vollie2` (`e893e8a6`); owner: no tags. LFM2.5 screen finished and appended to R50: reference row suspect (BF16 worse than its quantizations), no card frozen, no phone round planned. R51 sweep started on the laptop. |
| 2026-11-14 | 3 (continued) | R51 scored: f* = 13.39 % of expert bytes inside the budget on `olmoe-Mi` (hit); SUM-against-MASS prediction missed; amendment 1 void (no effect on the Q4_0 file). Next: device baseline pre-registration R52. |
| 2026-11-14 | 3 (continued) | Memo-05 (repack path), R51 amendment 2 pre-registered and started; memo-08 (device predictions; OLMoE now about 22.5 tok/s at the centre, *estimated*); the owner cannot attach the phone, so memo-09/10 and kit r17 (stock-only APK) were built; R52 pre-registered; APK handed to the owner. |
| 2026-11-14 | 3 (continued) | c2 series 0002 to 0009 (exact levers, kit interface, untied head) committed; kit r18 built and host-checked; R53 (head, host) and R54 (lever kit, device) pre-registered; R52 amendment 1; r18 handed to the owner. |
| 2026-11-14 | 3 (continued) | First device result: R52, 3.34 tok/s on the stock runtime, void by hygiene, cause diagnosed (memo-13: page cache, heat, gate bug). Kit r18 withdrawn. Owner: reuse campaign 1's Vollie. |
| 2026-11-14 | 3 (continued) | R55 pre-registered and kit r19 handed to the owner. R51 amendment 2 scored: three hits, f* stands on the no-repack path (olmoe-Mi 13.39 %, Q4_0 file 13.42 %). R53 (head) running on the host; its stock identity row reproduced 11.7146 on the full nine-patch series. |
| 2026-11-14 | 3 (continued) | R55 scored from kit r19: 4.43 / 7.20 / 7.51 tok/s, never resident, clock capped at 1,555 MHz; expected-state prediction hit, ratio prediction missed. Goal decision put to the owner. |
| 2026-11-14 | 3 (continued) | Memo-16 (r16 against r19: campaign 1's first run in r19's state was 12.27 tok/s). Owner's standing rule "start from Vollie 1" recorded in CLAUDE.md. Kit r20 built from r16's protocol (memo-17), keep-warm patch on campaign 1's runtime, R56 pre-registered, APK handed to the owner. Open check: Tom saw a different `-ub 1` value on today's base build than memo-07 recorded (8.8302 against 8.9550, *host*, 4 chunks); find out which build flag differs before the next host identity claim. |
| 2026-11-14 | 3 (continued) | R53 scored (host): untied head K=1024 passes (+0.016 % alone, +3.335 % with skip MASS=900); both identity rows reproduced R50 and R51 on the full nine-patch series. |
| 2026-11-14 | 3 (continued) | R56 scored from kit r20 (cool phone): OLMoE 11.25 tok/s at campaign 1's claim positions against Granite's 20.51; fault-free tokens 43 to 51 ms; never resident (max 2.39 of 2.73 GB); keep-warm closed. The gap to Vollie 1 is residency after the prefill. |
| 2026-11-14 | 3 (continued) | R57: the owner ran campaign 1's kit as a same-day control: Granite 11.46 tok/s at the claim positions (20.51 on 2026-11-08), OLMoE 11.25 the same day. The gap to campaign 1's number is the phone's page cache on the day; residency is the problem for both models. |
| 2026-11-14 | 3 (continued) | Memo-18: corrected working sets (Granite 1.97 to 1.99 GB, OLMoE 2.67 GB), page-cache simulator, first residency lever chosen (`GGML_PREFILL_COLD`). Owner: agreed; campaign 2 is also about trust through many models (all models stay full subjects); gate for batch-dependent numerics agreed. Kit r21 (two models in one session) in progress. |
| 2026-11-14 | 3 (continued) | `GGML_PREFILL_COLD` patch (identical on both models, *host*) and kit r21 (model table, two models in one session, report saved as a file); R58 pre-registered; APK handed to the owner. Open: the two host builds give different `-ub 1` values for OLMoE (8.9550 and 8.8302, 4 chunks); a build-option difference Tom has not pinned down. |
| 2026-11-14 | 3 (continued) | R58 scored (kit r21, two models, cool, high cache): OLMoE 19.70 and Granite-Mi 17.19 tok/s with no numerics change; `GGML_PREFILL_COLD` unresolved on both and closed by the kill rule; OLMoE level prediction missed on the fast side. Residency before the run decides speed; gate waits lose pages. Next: R59, skip and head on campaign 1's runtime for both models. |
| 2026-11-15 | 3 (continued) | R59 (*host*): skip rule and untied head ported onto campaign 1's runtime, identity kept; two-setting gate passed by OLMoE `MASS=900` + head 1024 and Granite skip 130 + head2; Granite 145 fails at `-ub 1`. Kit r22 and R60 next. |
| 2026-11-15 | 3 (continued) | Kit r22 built (num arm per model, fadvise page hold through the gate wait, `0005-kit-lines`, memo-20); R60 pre-registered; APK uploaded for the owner. Nothing of the hold code has run on a phone yet. |
| 2026-11-15 | 4 | R60 scored from kit r22 (pasted report, summary transcribed): hold passes on both models, cost missed; Granite num unresolved over all quads; OLMoE num void (untied head did not run on the phone); levels missed low, every gate wait timed out. Next: head fix, cheaper hold, R61 with hold on throughout. Agent models this session: lead `claude-fable-5-2` only; no subagents used. |
| 2026-11-15 | 4 (continued) | Tom (`claude-opus-5-5`): OLMoE head on the phone is declined because KleidiAI owns `output.weight`; reproduced on the host; patch 0006 (load-time knob, decline reason in the kit line). Finding: all host PPL so far is from builds without KleidiAI, the phone has it on. Owner decision opened; no kit built. |
| 2026-11-15 | 4 (continued) | Owner: R61 option (a) chosen, pre-statement pushed, queue running (Tom). Owner welcomes unconventional methods; Bonnie (`claude-opus-5-5`) wrote memo-21 (priced shortlist, dead list, owner items). |
| 2026-11-15 | 4 (continued) | R61 scored (*host*, KleidiAI build, 10 full-set runs): both num settings inside by both readings; Granite stock +0.57 / +0.50 % with KleidiAI (prediction missed at `-b 512`), OLMoE unchanged. Next: app load setting, cheaper hold, memo-21 items 1, 2, 5, then kit r23 and its pre-registration. |
| 2026-11-16 | 4 (continued) | Kit r23 built by Tom (OLMoE head knob at load, cheap hold in every gated run, threads 3 against 4, memo-22); R62 pre-registered; APK handed to the owner. Never run on a phone: 0006, the new hold rule, the thread sampler. |
| 2026-11-16 | 4 (continued) | R62 scored from kit r23 (raw report stored): Granite num pass and frozen; OLMoE num not shown, head runs but is worth about 0 ms; four threads killed on both; cheap hold misses its predictions; all gates timed out. Next: the pre-run state (gate, faults), OLMoE fault count in the skip arm, goal statement to the owner. |
| 2026-11-16 | 4 (continued) | Owner: apps were closed in R62; a charger was plugged in during Granite's last block (dated note). Gate reading: headroom plateaus near 0.72 on the three morning sessions. Tom: memo-23 (faults are mostly zram swap-ins, inferred; num-arm fault excess not an arm effect) and kit r24 (plateau gate rule as a factor, charger guard, swap counters). R63 pre-registered; APK handed to the owner. |
| 2026-11-16 | 4 (continued) | Owner: Bonnie moved to `claude-fable-5-2`, the lead (clyde, main session) to `claude-opus-5-5`; `.claude/agents/` and `CLAUDE.md` updated. Preregs from R64 on record these IDs. |
| 2026-11-16 | 4 (continued) | Bonnie: memo-24 (anon-memory account per model; swap-in faults confirmed but cheap, 4-6 us each in decode; OLMoE's largest fault block is the app's own KV memset before every prompt, about 3.5 us/fault of TTFT; shortlist led by clearing KV with MADV_DONTNEED, bit-identical by construction; Granite's tied head is already in KleidiAI, contrary to this file). All estimated or read from r22/r23; nothing measured. |
| 2026-11-16 | 4 (continued) | Tom: routing prior (memo-21 item 2) as patch 0007 plus `tools/c2_prior_trace.py`; host study filed as memo-25 (not an R-entry). Fits the budget on both models (OLMoE num+prior 12.2454/12.2435, Granite 9.5487/9.5478, host); residency-only, worth about 0.2-0.7 ms/token on OLMoE and 0 on Granite at R62's state (estimated). Not put in a kit. |
| 2026-11-16 | 4 (continued) | Owner: KV clear patch first; dump deleted (1.3 GB). Tom: patch 0008 (`VOLLIE_KV_DONTNEED=1`, madvise instead of memset in KV and recurrent clear; identity on macOS and Linux aarch64 container, host), kit r25 (r24 plus kv factor, memo-26). R64 pre-registered before handing over the APK. Open: `c2_research_parse.py` needs a kv split before R64 is scored; R63 (kit r24) not run yet, r25 carries its factors. |
| 2026-11-16 | 4 (continued) | Tom: `c2_research_parse.py` reads proto r25 (kv split paired by block x gate x arm, R64 void and text checks, R63 tables pooled and plain-only; r23/r24 output byte-identical). Lead's choices: decode ms/token = c2-io decode graph time / decode graphs; void rule over every non-kv row. Note: plain-only num minus off is one run against one per quad, not an ABBA mean. |
| 2026-11-16 | 4 (continued) | Owner: kit APKs go to the private HF repo `HoeioUser/vollie-kits` from now on (CLAUDE.md). r25 uploaded to `r25/` (APK sha256 1bbbec0c…c4180d4 checked on the server). |
| 2026-11-17 | 4 (continued) | R64 scored from kit r25 (raw report stored): kv factor VOID, every kv clear declined by the maps check (`why=map`, released 0; tagged heap pointers on arm64, inferred); texts equal kv on/off. R63's rules applied: gate exercised, plateau not shown on either model (wait -43 s hit); Granite num +2.68 +/- 0.71 pass; OLMoE num +2.02 +/- 1.09 not shown, third round, MASS=900 parked by R63 §4; vmstat unreadable, swap reading not scorable; levels not scored (cap 2035). Pre-prompt faults were small this day (plain medians 6 and 26). Environment: macOS denied this session access to ~/Documents after midnight; work continued in a fresh clone outside Documents; the Documents checkout needs `git pull` and its stray untracked `reports/device/r25_2026-11-16.json` (campaign 1 folder, copied by mistake) should be deleted. Next: untag the address in 0008, kit r26 (needs the lab build tree in Documents). |
| 2026-11-17 | 4 (continued) | Lead: patch 0009 (untag the buffer address in 0008; `tag=` in the kit line) and `tools/c2_kvclear_tag_test.cpp` (tagged-pointer test passes in a Linux aarch64 container, host). Kit r26 not built: the app copy and build trees are in ~/Documents, which this session cannot read. |
| 2026-11-17 | 4 (continued) | Owner's remote shell is also denied ~/Documents (mv failed). Owner chose kit r25b: the r25 APK with libllama.so rebuilt from 0001-0009 (CPU module and app code r25's byte for byte; same debug key), since the app source is in ~/Documents. R65 pre-registered (OLMoE num runs recorded, not scored, by owner's decision). Built outside Documents in ~/vollie-llama, ~/vb, ~/vollie-r25. |
| 2026-11-17 | 5 | Docs only (no measurement): README, CLAUDE.md, campaigns/README.md and CAMPAIGN.md no longer say campaign 2 is "planning, nothing measured"; README Honest limits now carry the R57 same-day control (11.46 vs 20.51, page cache) and a short "How AI is used" section; the developer e-mail may be public (owner, 2026-11-17). No numbers changed. |
| 2026-11-18 | 4 (continued) | ~/Documents readable again (owner). Checkout synced; stray r25 copy in campaign 1's folder removed. Tom: tree c1r26 (0001-0009), identity of 0008+0009 on both models (macOS and Linux aarch64, host), app patch 0010 (kit r26: r16 gate only, Granite num / OLMoE off, kv ABBA), parser r26, memo-27. R66 pre-registered (supersedes R65 for the phone; R65 dated note fixes its 15-of-16 slip). |
| 2026-11-18 | 4 (continued) | R66 scored from kit r26 (raw report stored): mechanism pass on both (why=ok 8/8, tag=0xb4: tagged-pointer cause confirmed); OLMoE TTFT -704 +/- 149 ms pass; Granite effect not shown; texts equal. Knob stays on in later kits. Owner: background apps restart by themselves during runs (waits lost 0.5-1 GB); asks for an in-app control that closes them. Next: kit r27 with a background-kill factor (Tom), R67. |
| 2026-11-18 | 4 (continued) | Tom: kit r27 (background-app kill) not built: on Android 14+ `killBackgroundProcesses` kills only the caller's own processes (SDK 35 source, ActivityManager javadoc); the phone runs Android 15 (AP3A), so the factor would be null by construction. Options put to the owner: Samsung deep-sleeping list as a recorded bench condition, shell kill via wireless debugging/Shizuku, or runtime levers on the symptom (hold during the wait, less own swap). |
| 2026-11-18 | 4 (continued) | Bench condition from now on (owner): the apps that restarted themselves in R66 are in Samsung's deep-sleeping list on the test phone; recorded as a bench condition (COMMON_RULES 5), not a fix for users; a goal claim must also hold without it. Tom asked to audit the app and runtime code for inefficiency (the app needs only minimal features). |
| 2026-11-18 | 4 (continued) | Joint research round (Bonnie + Ebony, both claude-fable-5-2; Ebony ran as a stand-in with ebony.md's text): memo-29 with TEST list T1-T11 and TRY list Y1-Y12, fits and drafts kept in `reports/memo-29-work/`. Proposals only, nothing measured; the lead's decisions on them are pending and will be written by the lead. |
| 2026-11-18 | 4 (continued) | Tom: memo-28 code audit. Main finding: the thread pin is lost after the gate wait (workers on A55 cores in 39 of 60 OLMoE gated/profiled runs, 0 of 12 warm-ups; inferred cause: Dart retires the pinned pool thread), so R60-R66 gated runs were partly unpinned. Lead's decisions on memo-28/29: kit r27 = runtime patch 0010 (persistent threadpool + per-worker cpumask, off unless set) as the one ABBA factor on both models; kv clear on in every run; memo-28 (a) instrument cleanups, OpenCL out, n_ctx fixed at 4096, static kit screen, declared as instrument changes; riders T1 (state counters, resident vs mapped), T2 (exec-mmap probe), T3 (14-op profile). MASS=900 stays parked until T1 shows the memory state under control (R63 §4). Anon hold demoted (TTFT only). Y1 PROT_EXEC waits for the owner. |
| 2026-11-18 | 4 (continued) | Owner: Y1 (PROT_EXEC mapping) goes into kit r27 now. Order at session start: exec/noexec check on internal storage first; only if it passes, copy each model from the external file (or download if absent), sha256-checked; on failure the session runs from external files with the exec factor void. r27 design: pin and exec as a balanced 2x2 per quad on both models, kv clear on. Tom building. |
| 2026-11-18 | 4 (continued) | Owner created 15 annotated tags on 2026-11-18, retroactively, each on the commit that first added the pre-registration file: `goal1-final` (`686f17d`) and `prereg-R51` to `prereg-R66` except R57 and R59 (no pre-registration file; R59 was a host PPL, R57 a same-day control). The commit date, not the tag date, is the evidence of timing; each tag message says so. Checked against the remote by the lead. `CLAUDE.md` corrected. New pre-registrations still need their tag before a device run. |
| 2026-11-18 | 4 (continued) | Tom: kit r27 built (patches 0010 pool+pin, 0011 exec map; app 0011; parser r27; memo-30); host identity equal for pool, exec and both on both models, macOS and Linux aarch64. R67 pre-registered (pin and exec as a balanced 2x2 per quad, both models). |
