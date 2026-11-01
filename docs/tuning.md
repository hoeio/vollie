# Tuning on the device

What works best differs per phone, per heat state and per free memory, so Vollie measures on the
device instead of shipping a table. Its "Optimize for this phone" speed test, as of app 2.2.5:

1. **Load once on the CPU.** Thread counts change in place (`llama_set_n_threads`, via
   `patches/llamadart`); a reload re-reads the model and rebuilds the repacked copy, which took most
   of a minute per candidate on the SD888.
2. **Only thread counts that fit the big cores** (`fastCoreIds`), when core types are known. A thread
   on a little core holds every sync: on the SD888, 5 threads (4 big + 1 A55) took minutes.
3. **Cool down before every timed run**: wait (≤ 90 s) until Android's thermal headroom forecast is
   under 65%, and record it with the result. Back-to-back tests had halved every number, the GPU's
   included, which no setting can do.
4. **Knob sweeps first**, at the last test's writer thread count: `GGML_TINY_OP` 0/65536/262144, then
   `GGML_EXPERT_IO` 0/1/5/13. Each run is a ~120-token prompt and 32 written tokens; the pick is the
   lowest time for a 500-token prompt + 200-token reply. Major page faults per token are recorded
   for every run.
5. **Then thread counts** (with the picked knobs), then GPU backends last, so a slow GPU run doesn't
   heat the phone for the CPU runs.
6. **Pick** with `Timing.fastest`: the best writer and the best reader thread count, each scored by
   its best run, the sweep's winning run included (it ran at exactly those settings). Single runs
   on a warm phone vary about ±20%.
7. **Profile**: two replies with `GGML_CPU_PROF` (the pick, then everything off) and one
   `GGML_CPU_AB=q3k` reply, each with fault counts, memory (`memInfo`) and core clocks (`cpuMhz`).
   "Copy speed test details" puts all of it on the clipboard to share.

The picked values are saved and applied with `setenv` at the next start, before the model loads.

## Lessons from running it

- Measure on the device. The M4 never showed the sync cost (0.4 µs per sync) or the page-cache
  thrash (24 GB of RAM), the two largest effects found.
- Count what the kernel counts: major page faults turned "the experts are slow" into "the experts
  are re-read from flash", which pointed at prefetch rather than kernels.
- Compare inside one generation when the difference is small; separate runs drift with heat.
- Fix the environment as given: the phone's free memory and heat are what users have.
