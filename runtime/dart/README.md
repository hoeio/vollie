# vollie_runtime (Dart)

App-side helpers Vollie uses to serve llama.cpp on Android, without Flutter or app code:

| call | does |
|---|---|
| `setTinyOps`, `setExpertIo`, `setProfiling`, `setAb`, `setNativeEnv` | turn on the patched CPU module's knobs (`docs/runtime-knobs.md`); ggml reads them before each graph, so they change without a reload |
| `fastCoreIds`, `performanceCores` | find the big cores from `/proc/cpuinfo` part numbers, scheduler capacity or max clocks |
| `pinInferenceThreads` | pin llama.cpp's worker threads (Dart `DartWorker` threads) to those cores; call before each generation |
| `pageFaults`, `memInfo`, `cpuMhz` | read major/minor faults, memory and core clocks, to tell flash re-reads and throttling from real slowdowns |
| `stderrTo` | keep llama.cpp's log (and the profiler lines) in a file on Android |
| `Timing.fastest` | pick the setting from timed runs: best writer + best reader thread count, each scored by its best run |

```yaml
dependencies:
  vollie_runtime:
    git: {url: https://github.com/hoeio/vollie.git, path: runtime/dart}
```

How Vollie strings these together on a device is in `docs/tuning.md`. `dart test` covers the core
detection (Snapdragon 888 part numbers) and the selection rule.
