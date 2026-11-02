# Research kit (round 2) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~10 min)"** button in Vollie's Settings (app 2.2.7+30, kit `r2`;
round 1 was 2.2.6+29 / `r1`, results in `reports/03_final.md` R18). It measures the round-2 build
(`reports/04_campaign.md` §7) on the phone as it is, and puts one JSON report on the clipboard
("Copy research results"). `tools/research_parse.py` turns that JSON into `results.csv` rows and R-log text.

What changed in the round-2 build: the CPU module now defaults to `GGML_MMID=3` + `GGML_Q3K_X4=1`
(R18: −8.3 %/token, bit-identical), and the app ships a rebuilt `libllama.so` (`lab/granite-rollback.patch`)
so a Granite-H target can roll a rejected draft back — round 1's speculation runs all failed on that.

## What one run does (about 10 minutes, no settings to change)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 2 | official benchmark prompt, 32 tokens, saved settings (3 threads, tiny-op, expert I/O) — now with the new defaults | — |
| `ab` | 4 | in-generation A/B over a 72-token story: new defaults vs off (regression check), then tiny-op 65,536 vs 262,144 three times | `GGML_CPU_AB=knob:…` |
| `spec` | 9 | reload with 4 recurrent rollback slots; warm-up on free text with 4 draft tokens (a rejection must occur — a note is added if not); official 32 without a draft, then with 1 / 2 / 3 / 4 draft tokens; story 64 without, then with 2 / 4. Draft = Granite 4.0 350M Q4_0 (downloaded once, 228 MB, SHA-256 checked). Draft runs set `GGML_CPU_PROF=2`: the profile line covers the 2-8-token verify graphs only (`us/token` there = µs per graph, `tokens` = tokens they covered) → batched-verify cost | `ModelParams.speculativeRollbackTokenMax`, `SpeculativeDecodingConfig.draftSimple`, `GGML_CPU_PROF=2` |

Round-2 results: `reports/03_final.md` R20 (speculation measured and closed; tiny-op 262,144 default). Known r2 kit flaws: `GGML_CPU_PROF=2` lines also count the draft's 2-5-token catch-up graphs (and, with rollback slots, some 1-token target graphs), so the verify cost is read from `spec.eval_ms / attempts`; the saved knobs drifted between rounds (expert I/O 5 → 1) because the speed test re-ran — a kit should pin what it is not testing.

`decode_tps` of a draft run is the **effective** rate (every accepted token counts; greedy, so the text is the
model's own). Round 1's barrier, threads and release phases are gone: answered (`03_final.md` R18).

Every run waits for thermal headroom < 0.65 (≤ 90 s) like the speed test, and records the headroom
it started at, core clocks before/after, `/proc/meminfo` + the process's RssAnon/RssFile, major/minor
page faults, bytes read from storage (`/proc/self/io`, absent where Android denies it), token counts, time to
first token, duration, the module's `cpu-prof` lines, and the first 80 characters of the reply (same text
for every bit-exact knob).

The knobs are reset and the model reloaded normally afterwards. The draft model file stays next to
the model (`granite-4.0-350m-Q4_0.gguf`, 228 MB); delete it from the app's files to reclaim space.

## Result file (JSON, one object)

```
kit, app, started, finished, stopped_early   strings/bool
header        [str]  Vollie's test header (chip, RAM, context, CPU parts, clocks, fast cores, features, GPU, setting)
saved         {tiny, expert_io, write_threads, read_threads, context}   the speed test's picks the run started from
kernel, thp   str    /proc/version, /sys/kernel/mm/transparent_hugepage/enabled
mem_start, mem_end   str   DeviceProfile.memInfo(): avail= cached= swapfree= anon= file= swap= (MB)
notes         [str]  e.g. "draft model unavailable: …", "aborted: …"
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
  spec                 {draft_tokens, accepted, attempts, verify_tokens, replay_tokens, draft_ms, verify_ms, eval_tokens, eval_ms}  (speculation runs)
  mem_before_load      (release/after load) memInfo() before the reload
  error                exception text if the run failed
```

Reading a `cpu-prof` line: `… graphs=32 tokens=32 … us/token=98176 compute=74768 wait=23016 nodes=1963 barriers=818 mmid_thr=22387/23027/22477 | MUL_MAT_ID 120:34214+16861 …` —
`mmid_thr` = µs each thread spent computing inside the expert mat-muls per token (spread = imbalance);
after `|`, per op type: count : µs computing + µs waiting at the sync after it (thread 0). `skip=60/1%` =
expert-skip threshold / share of routed experts skipped. `x4`, `mmid` = kernel and no-barrier knobs in effect.
With `GGML_CPU_PROF=2` (draft runs) `graphs` counts verify graphs of 2-8 tokens, `tokens` their total, and `us/token` is µs per graph.

## 실행 안내 (한국어)

**준비:** Vollie 2.2.7+30 설치(APK, `vollie-2.2.7+30-research-r2.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다. 전화기 설정은
아무것도 바꾸지 마세요(다른 앱도 그대로). Wi-Fi 연결은 필요합니다(드래프트 모델 228 MB를 한 번 받습니다).

1. Vollie → 설정(Settings) → **"Run research kit (~10 min)"** 탭.
2. 약 10분 동안 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
   모델을 다시 읽는 구간(`reloading…`)이 1분 정도 걸립니다. 1회차에서 받은 드래프트 모델(228 MB)이 남아 있으면 다시 받지 않습니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 30-60 KB). 같은 내용이 앱 파일 폴더에 `research-r2-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
