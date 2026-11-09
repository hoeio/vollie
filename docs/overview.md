# Overview: how a 7B MoE got to 20 tok/s on a 2021 phone CPU

> **Derived page.** This is a plain-language retelling of [findings.md](findings.md), the [README](../README.md)
> and the R-log in [reports/03_final.md](../reports/03_final.md). It adds no new measurements. If it ever
> disagrees with those, they win. Labels follow [RULES.md](../RULES.md): *device* = measured on the phone,
> *host* = measured on a laptop, *estimated* = arithmetic.

## The question

Can a Snapdragon 888 phone (8 GB, CPU only) print 20 tokens per second from Granite-4.0-H-Tiny, a 7B
mixture-of-experts model, **without changing the model file** and without asking the user to do anything to
the phone? The one quality limit was perplexity within +20 % of the F16 weights, with a smaller +2 % budget
for anything that changes the numbers.

## The story, in the order things were found

1. **It is not memory bandwidth.** The plan assumed reading ~0.9 GB per token at ~13-16 GB/s would give 10-14
   tok/s. The first device run gave 7.4 tok/s, and half of the 135 ms per token was not explained by bytes.
2. **Thread synchronisation.** The model runs about 2,000 small ops per token and the runtime synced all
   threads after every one. On the phone those syncs were roughly half of each token (on a laptop they cost
   almost nothing, which is why the laptop hid it). Running small ops on one thread without syncs took syncs
   per token from 1,909 to 874 and 2-thread speed from 3.2 to 7.0 tok/s (*device*).
3. **The experts do not fit in the page cache.** The 3.1 GB of expert weights are read through mmap, and the
   phone kept only about 2.1 GB of cache. Every token re-read part of them from flash: 354-1,181 major page
   faults per token. Requesting exactly the six routed experts together (prefetch) took 6.1 to 11.8 tok/s and
   cut faults from 708 to 253 per token (*device*).
4. **Then it became compute.** With the reads hidden, the 3-bit expert kernel was compute-bound on the A78
   cores. Several bit-identical kernel and state changes (see [levers.md](levers.md)) took the steady token
   from 91.0 to 63.9 ms.
5. **Measurement noise was as big as the effects.** Single runs on a warm phone vary about ±20 %. From round 5
   every knob is measured as an in-run A/B that alternates the setting inside one generation, so heat, clock
   and cache hit both sides alike.
6. **Spending the +2 % budget.** Skipping experts with a small router weight (threshold 117) and a two-stage
   output head cost +1.91 % perplexity together. Skip alone was worth −7.90 ms per token at threshold 118
   (*device*, in-run A/B). The budget was fixed in advance and not raised after results.
7. **The result.** Round 16: a median of 20.51 tok/s over 10 plain runs of the official 32-token benchmark
   (*device*, R49). A cold start after memory pressure still prints 12.3-18.5.

## How the evidence is kept honest

- Predictions and pass rules are written into the R-log **before** each device run and scored as written,
  misses included. R49's prediction (median 18.0-20.0) missed high.
- Every speed number carries a label, and host numbers are never presented as phone numbers.
- Page faults, clocks and memory are logged next to every timing, because they turned out to explain most
  surprises.
- Results are appended to the log; history is not rewritten. Corrections appear as dated notes (for example
  the R49 correction note about why slow runs are slow).

## What is still open

- **Cold start:** the first run after memory pressure misses 20; the cause is page residency at the start of
  the run.
- **Generalisation:** one device and one model so far; showing that the method carries over is the stated next
  step. Reports from other phones are welcome: see [run-on-your-phone.md](run-on-your-phone.md#share-your-result).

## Where to go next

| If you want to | read |
|---|---|
| see each lever with its numbers | [levers.md](levers.md) |
| see what was tried and dropped | [negative-results.md](negative-results.md) |
| find the entry for a specific R-number | [reports/README.md](../reports/README.md) |
| look up a term | [glossary.md](glossary.md) |
| use or change a runtime knob | [runtime-knobs.md](runtime-knobs.md) |
