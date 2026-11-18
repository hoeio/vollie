# Pre-registrations of campaign 2

One file per round, named `R<n>-<short-title>.md`, copied from [TEMPLATE.md](TEMPLATE.md). The first one is
**R50** (campaign 1 ended at R49). A file here is committed and tagged `prereg-R<n>` before the run it governs;
see [../../README.md](../../README.md).

- [R51-olmoe-expert-skip-frontier-host.md](R51-olmoe-expert-skip-frontier-host.md): OLMoE expert-skip frontier, host.
  R50 was a host screen governed by memo-01 part C, so the first prereg file is R51.

- [R51 amendment 1](R51-amendment-1.md): the same frontier on the second OLMoE file (publisher Q4_0).
- [R51 amendment 2](R51-amendment-2.md): the frontier re-measured on the phone's no-repack kernel path.
- [R52](R52-olmoe-stock-baseline-device.md): OLMoE on the stock runtime, device baseline with kit r17.
- [R53](R53-olmoe-head2-host.md): untied two-stage head on OLMoE, host perplexity.
- [R52 amendment 1](R52-amendment-1.md): the baseline may be taken from kit r18.
- [R54](R54-olmoe-lever-kit-device.md): OLMoE levers on the phone, kit r18.
- [R54 amendment 1](R54-amendment-1.md): kit r18 withdrawn before any run.
- [R55](R55-olmoe-c1-runtime-residency-device.md): OLMoE on campaign 1's runtime with residency measured, kit r19.
- [R56](R56-olmoe-r16-protocol-device.md): OLMoE under campaign 1's kit r16 protocol, warm block, keep-warm; kit r20.
- [R58](R58-prefill-cold-two-models-device.md): prefill-cold on Granite and OLMoE in one session, kit r21.
- [R60](R60-numerics-and-hold-two-models-device.md): skip + two-stage head and page hold through the gate wait, two models; kit r22.
- [R61](R61-kleidiai-host-ppl.md): stock and num perplexity on a host build with KleidiAI (pre-statement, host).
- [R62](R62-numerics-threads-hold-two-models-device.md): skip + head with hold in every run, threads 3 against 4; kit r23.
- [R63](R63-gate-rule-and-numerics-two-models-device.md): plateau gate rule against r16, skip + head; kit r24 (not run; its rules were applied in R64).
- [R64](R64-kv-clear-dontneed-two-models-device.md): KV and recurrent clear with MADV_DONTNEED; kit r25 (kv factor void, tagged pointer).
- [R65](R65-kv-clear-untagged-two-models-device.md): the same on the untagged address; kit r25b (superseded by R66); [dated note 2026-11-18](R65-note-2026-11-18.md).
- [R66](R66-kv-clear-untagged-two-models-device.md): the same on the untagged address; kit r26 (scored in R66 of the R-log).
