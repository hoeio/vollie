# Campaign 2 status and hand-off

**Read this first, update it last.** Every session appends a line to the session log below and refreshes the
sections above it. Last updated: 2026-11-13 (session 3, continued).

## State in one paragraph

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

None at the moment.

Hosting (2026-11-13): the two mixes are in the public repository `HoeioUser/vollie2` at commit `e893e8a6`
(server-side sha256 checked); the publisher's Q4_0 is taken from AI2's repository at its pinned commit. Tags: the
owner does not want them; the order of pre-registration and run is shown by commit hashes and push times
(R51: `ab7875e`, amendment 1: `e16630b`).

Answered 2026-11-13: push to `main` directly, no PR; the publisher's Q4_0 is a second OLMoE served file
(CAMPAIGN.md, decisions 9 and 10); the contact e-mail and the private path were removed from campaign-1 files.

## Next actions, in order

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
   **Now with the owner: kit r19 / R55** (campaign 1's runtime on OLMoE, three rows, working gate, mincore
   residency). Memo-15 (*estimated*): OLMoE needs `Cached` ≥ about 2.86 GB to be fault-free; one of 19 recorded
   sessions had that; expected result "not resident", `c1_off` 3.5 to 4.5, `c1_ship` 7 to 10 tok/s. The goal is
   therefore conditional on the phone's memory state; the options (state-conditioned claim, a smaller file,
   private expert memory, or "no") were put to the owner and wait for the r19 numbers. Also asked: one run of the
   campaign-1 app's own benchmark as a same-day control.
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
