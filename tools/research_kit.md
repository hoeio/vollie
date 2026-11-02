# Research kit (round 3) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.8+31, kit `r3`; round 2 was
2.2.7+30 / `r2`, results in `reports/03_final.md` R20). It measures the round-3 probes
(`reports/05_compute_less.md` §3 S5(0)) on the phone as it is, and puts one JSON report on the clipboard
("Copy research results"). `tools/research_parse.py` turns that JSON into `results.csv` rows and R-log text.

What changed in the round-3 build (CPU module, every probe default off): `GGML_Q3K_ILP=1` (the Q3_K x4 expert
dot with the 4 rows side by side and no serial accumulator chain; bit-identical), `GGML_Q3K_PF=1` (prefetch
of the next 4 rows' block inside the x4 loop; bit-identical), `GGML_EXPERT_ANON=<N>` (the N most used experts
of each expert tensor copied into anonymous memory, window-LFU over 64 tokens, ≤ 4 promotions per token, the
prompt fills it, lives across requests; same bytes, so identical output). The speculation phase is gone
(closed in R20) and `GGML_CPU_PROF=2` now counts only the main model's 2-8-token graphs.

## What one run does (about 12 minutes, no settings to change)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 2 | official benchmark prompt, 32 tokens, saved settings | — |
| `ab` | 6 | in-generation A/B over a 72-token story: `Q3K_VEC:0:1`, `Q3K_ILP:0:1`, `Q3K_PF:0:1`, `Q3K_ILP:0:1`, `Q3K_PF:0:1`, `EXPERT_SKIP:0:<t>` (the threshold placed on the full wikitext-2 set, R23; not bit-exact) | `GGML_CPU_AB=knob:…` |
| `anon` | 5 | separate rows (the cache is state): story 200 without the cache, story 200 with N = 16 cold (fills while it runs), story 200 warm (same prompt: best case), official 32 warm, then `EXPERT_ANON:0:16` A/B on the warm cache (one cache, filled on both sides; B reads the copies, A the file) | `GGML_EXPERT_ANON`, `GGML_CPU_AB` |

Known r2 kit flaws, fixed: the `PROF=2` mix (phase dropped; the profiler now keys on the main model's token
embedding); the saved knobs still come from the speed test's picks (`saved` in the JSON) — compare rows within one run.

Every run waits for thermal headroom < 0.65 (≤ 90 s) like the speed test, and records the headroom
it started at, core clocks before/after, `/proc/meminfo` + the process's RssAnon/RssFile, major/minor
page faults, bytes read from storage (`/proc/self/io`, absent where Android denies it), token counts, time to
first token, duration, the module's `cpu-prof` lines, and the first 80 characters of the reply (same text
for every bit-exact knob).

The knobs are reset and the model reloaded normally afterwards (an empty `GGML_EXPERT_ANON` frees the cache).

## Result file (JSON, one object)

```
kit, app, started, finished, stopped_early   strings/bool
header        [str]  Vollie's test header (chip, RAM, context, CPU parts, clocks, fast cores, features, GPU, setting)
saved         {tiny, expert_io, write_threads, read_threads, context}   the speed test's picks the run started from
kernel, thp   str    /proc/version, /sys/kernel/mm/transparent_hugepage/enabled
mem_start, mem_end   str   DeviceProfile.memInfo(): avail= cached= swapfree= anon= file= swap= (MB)
notes         [str]  e.g. "aborted: …"
runs          [run]
run:
  phase, name, threads, env {VAR: value}          what was set (env values '' = off)
  heat                 thermal headroom at start (1.0 = severe throttling)
  mhz_before/after     per-core MHz
  mem_before/after     memInfo() strings
  wall_ms, tokens, prompt_tokens, cached_prompt_tokens, ttft_ms, duration_ms
  decode_tps           Vollie's own number: (tokens-1)*1000/(duration_ms-ttft_ms)
  prefill_tps
  majflt, minflt       page faults during the run (process-wide)
  io_read_bytes        bytes read from storage during the run (process-wide)
  prof                 [str] cpu-prof / cpu-bench / repack-release lines the module printed during the run
  text                 first 80 chars of the reply
  mem_before_load      (release/after load) memInfo() before the reload
  error                exception text if the run failed
```

Reading a `cpu-prof` line: `… graphs=32 tokens=32 … us/token=98176 compute=74768 wait=23016 nodes=1963 barriers=818 mmid_thr=22387/23027/22477 | MUL_MAT_ID 120:34214+16861 …` —
`mmid_thr` = µs each thread spent computing inside the expert mat-muls per token (spread = imbalance);
after `|`, per op type: count : µs computing + µs waiting at the sync after it (thread 0). `skip=60/1%` =
expert-skip threshold / share of routed experts skipped, both on that side; `kavg` = experts kept per layer and token on that side.
`x4`, `ilp`, `pf`, `mmid` = kernel and no-barrier knobs in effect. `anon=16/68%/11.2` = cache slots per expert
tensor on that side (0 = not read) / routed experts of decode tokens found cached / tensor copies per decode token
(Granite: 3 tensors per expert, ~0.34 MB each); `anon_mb` = anonymous memory the cache holds.
`majflt`, `minflt` per run divided by `tokens` = faults per token (the parser prints both).

## 실행 안내 (한국어)

**준비:** Vollie 2.2.8+31 설치(APK, `vollie-2.2.8+31-research-r3.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다. 전화기 설정은
아무것도 바꾸지 마세요(다른 앱도 그대로). 이번 회차는 추가 다운로드가 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭.
2. 약 12분 동안 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
   `anon` 단계에서 앱이 메모리를 약 650 MB 더 씁니다(끝나면 돌려줍니다).
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 30-60 KB). 같은 내용이 앱 파일 폴더에 `research-r3-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
