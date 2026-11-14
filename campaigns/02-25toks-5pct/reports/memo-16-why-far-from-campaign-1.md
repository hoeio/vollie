# Memo 16: why R55 (7.5 tok/s) is far from campaign 1 (20.51 tok/s)

Agent model: claude-fable-5-2 (lead). Date: 2026-11-14. Sources: campaign 1's last kit report
`reports/device/r16_2026-11-08.json` (the R49 session, read with own code) and campaign 2's
`reports/device/r19_2026-11-14_summary.json` (transcribed summary). Every number is *device* unless marked.

## 1. Campaign 1's own report, run by run

The 19 official runs of r16, in order (Granite, campaign 1's runtime):

| run | tok/s | thermal headroom at the gate | A78 MHz at run end | wait before the run | app file pages resident before (MB) | `Cached` before (MB) | free swap (MB) |
|---|---|---|---|---|---|---|---|
| 1 | 12.27 | 0.870 | 1,555 | none | 313 | 2,490 | 1,259 |
| 2 | 14.09 | 0.830 | 2,112 | 90 s | 283 | 2,235 | 191 |
| 3 | 16.84 | 0.766 | 2,112 | 90 s | 690 | 2,372 | 590 |
| 4 | 18.88 | 0.690 | 2,112 | 90 s | 1,754 | 2,678 | 420 |
| 5 | 18.47 | 0.719 | 2,112 | 90 s | 1,514 | 2,724 | 153 |
| 6 | 20.63 | 0.661 | 2,112 | 110 s | 1,868 | 2,877 | 397 |
| 7 to 19 | 17.10 to 20.95 | 0.636 to 0.652 | 2,112 | 0 to 90 s | 1,587 to 2,015 | 2,670 to 2,976 | 113 to 500 |

- **Campaign 1's first run, in the thermal state r19 was in all session (headroom 0.87, cap 1,555 MHz), printed
  12.27 tok/s.** The 20.5 came after the phone had cooled to 0.64 during the kit (about six runs, about ten
  minutes), the clock had gone to 2,112 MHz, the app's resident file pages had grown from 0.3 to 2.0 GB and
  `Cached` from 2.5 to 2.9 GB.
- At 2,112 MHz the five back-to-back runs (wait under 0.1 s) have a median of 20.40 tok/s and a prefill of
  about 3.56 s; the thirteen runs after a wait have a median of 18.88.

## 2. The same quantities in r19 (OLMoE, the same runtime)

| quantity | r16, fast part | r19, all session |
|---|---|---|
| thermal headroom at the gate | 0.636 to 0.652 | 0.81 to 0.87 (10-minute gate: lowest 0.774) |
| big-core clock cap | 2,112 / 2,035 MHz | 1,555 MHz |
| gate | passes, often at once | timed out before every run, so every run follows 90 s of idle |
| model pages in the page cache before a run | 1.6 to 2.0 GB (process RssFile) | 0.4 to 1.4 GB (mincore) |
| `Cached` before a run | 2.67 to 2.98 GB | 1.94 to 2.43 GB |
| free swap before a run | 113 to 500 MB | 0 to 142 MB |
| `Cached` when the kit started | 2,751 MB | 671 MB |
| prefill of the prompt | 3.5 to 4.4 s | 8.3 to 9.2 s (median 8.8) |
| run length | about 5 s | 12 to 21 s |

## 3. Reasons, in the order the data supports them

1. **The phone never cooled, so the clock stayed capped (established).** 1,555 against 2,112 MHz. With memo-15's
   fit for a fault-free Granite token (19.1 + 27.1 × 2112 / f ms) that alone turns 46 ms into 56 ms, 20.5 into
   about 17 tok/s (*estimated*).
2. **The model was not in memory, and every run started cold (established).** In r19 each run came after a
   90-second wait in which the phone took back 0.5 to 1.4 GB of the model's pages; the prompt then had to be
   read from flash again (prefill 8.8 s against 3.6 s) and the decode tokens faulted. Campaign 1's fast runs
   were mostly back to back with the cache warm, because its gate passed.
3. **The phone had less memory to give on this day (established), cause unknown.** Swap was exhausted and
   `Cached` was 0.4 to 0.5 GB lower during the runs, 2.1 GB lower at the start of the kit. What used the memory
   is not in the report.
4. **A loop between 1 and 2 (plausible, not proven).** A non-resident OLMoE run keeps three threads busy for 12
   to 21 s where a resident Granite run took 5 s, so each cycle puts in three to four times the heat; the phone
   then does not cool in 90 s, the gate times out, the wait evicts the cache, and the next run is long again.
   r16 ran the other way round: short runs, cooling, gate passing, cache staying warm.
5. **OLMoE's working set is larger (measured on the host, weight on the phone not established).** 2,725 MB by
   trace against 2,569 MB for Granite; but Granite ran fault-free with about 2.0 GB of file pages resident, so
   the trace overstates what must be resident by an unknown amount (memo-13's open discrepancy).

**Not a reason, as far as the data goes:** the kernels. The cleanest OLMoE tokens in r19 took 55 to 73 ms at the
1,555 MHz cap; memo-15's fit puts a fault-free Granite token at about 56 ms at that clock. The two models cost
about the same per token when nothing faults.

## 4. What would separate the reasons

- **Control on the same day:** the campaign-1 app's own kit (r16 build) on the phone as it is now. If Granite
  also stays near 12 tok/s, the day's state explains the gap; if it reaches 20, reason 5 carries more weight.
- **A kit that measures the warm state instead of destroying it:** when the gate cannot be met, do not wait 90 s
  before every run; run one unscored warm-up and then the scored runs back to back, and report both protocols.
  This is how campaign 1's fast runs happened in practice.
- **Runtime keep-warm (a lever, inside the rules):** keep the experts' pages referenced while the app waits, so
  the system does not take them back. Its heat cost has to be measured.
