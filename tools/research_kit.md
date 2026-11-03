# Research kit (round 5) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.10+33, kit `r5`; round 4 was
2.2.9+32 / `r4`, results in `reports/03_final.md` R24). It measures the round-5 runs
(`reports/05_compute_less.md` §7.5-7.6) on the phone as it is, and puts one JSON report on the clipboard
("Copy research results"). `tools/research_parse.py` turns that JSON into `results.csv` rows and R-log text.

What changed in the round-5 build (CPU module): `GGML_SSM_INPLACE` is now **on by default** (bit-identical, R24;
`GGML_SSM_INPLACE=0` turns it off), so the baseline rows include it. New probe `GGML_CONV_RING=1` (default off,
bit-identical): the one-token Mamba2 conv reads the old conv state and the new input where they are and shifts the
state into the cache itself, so the 36 `CONCAT` and 36 conv-state `CPY` ops are not run. **A/B runs are now ABBA
blocks** inside one 200-token story: 16 warm-up tokens on side A, then 8-token blocks A B B A A B B A … (one
`cpu-prof` line per block, `ab=A|B blk=N`, `blk=0` = warm-up; `GGML_CPU_AB_WARM` / `GGML_CPU_AB_BLOCK` change the
16 / 8), so a drift during the run (page cache filling, clocks) falls on both sides alike. Fix: an empty
`GGML_Q3K_VEC` now means "default" (on); the r3/r4 kits reset it to '' and so ran it off (`q3k=0` in their lines;
the x4 kernel bypasses it, so their expert numbers are unaffected).

## What one run does (about 15 minutes, no settings to change, no extra memory)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 2 | official benchmark prompt, 32 tokens, saved settings (round-5 defaults: ILP + SSM in place, no skip) | — |
| `vanilla` | 2 | **stock llama.cpp settings, for comparison**: official 32 and story 200 with every Vollie knob that has an off switch off (list below), threads as saved | env for that run only |
| `ab` | 7 | ABBA A/B over the 200-token story: `SSM_INPLACE:1:0` and `Q3K_ILP:1:0` ×2 (**A = on = default, B = off**: B − A > 0 means it helps), `EXPERT_SKIP:0:115` (the budget threshold), `EXPERT_SKIP:0:118` (the edge: the largest threshold whose full-set PPL is still ≤ 9.797, 9.7952 = +1.98 %), `CONV_RING:0:1` (A = off) | `GGML_CPU_AB=knob:…` |
| `on` | 3 | always-on rows: story 200 with `GGML_EXPERT_SKIP=118`, story 200 with nothing (control), official 32 with `GGML_EXPERT_SKIP=118` | `GGML_EXPERT_SKIP` |

The parser's A/B Δ is always B − A, per side mean ± sd over the post-warm-up blocks, ± 1σ, and the per ABBA group
(B̄ − Ā) mean and sd; the two `Q3K_ILP:1:0` runs are also pooled. Every story run also logs `mhz_blocks`: the cpu4
and cpu7 clock every 8 tokens.

**Vanilla rows, what is off:** `GGML_TINY_OP=0`, `GGML_MMID=0`, `GGML_Q3K_X4=0`, `GGML_Q3K_VEC=0`, `GGML_Q3K_ILP=0`,
`GGML_SSM_INPLACE=0`, `GGML_CONV_RING=0`, `GGML_EXPERT_SKIP=0`, `GGML_EXPERT_IO=0` (expert prefetch and next-layer
prediction off); `GGML_Q3K_PF`, `GGML_EXPERT_ANON`, `GGML_EXPERT_TOPP/GAP` are off anyway. **What stays Vollie's in
those rows (no switch without a reload or more memory; README footnote, not a fake vanilla):** the KV cache is q8_0
with flash attention (stock: f16); the weights run from the mmap because Vollie's memory check turns the repacked
copies off on an 8 GB phone (stock `use_extra_bufts` = on, which ran this phone out of memory, R-log 2026-10-31);
`GGML_REPACK_RELEASE` acts at load only; the compute threads are pinned to the big cores and their count (3) is
the speed test's pick; the Granite rollback patch (`llama-arch.cpp`) is inert with 0 rollback slots; context size,
chat template and greedy sampling are the app's. The patched module's code is present but each path above is the
upstream one when its knob is off.

Every run waits for thermal headroom < 0.65 (≤ 90 s) like the speed test, and records the headroom
it started at, core clocks before/after, `/proc/meminfo` + the process's RssAnon/RssFile, major/minor
page faults, bytes read from storage (`/proc/self/io`, absent where Android denies it), token counts, time to
first token, duration, the module's `cpu-prof` lines, and the first 80 characters of the reply (same text
for every bit-exact knob).

The knobs are reset and the model reloaded normally afterwards.

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
  mhz_blocks           [[cpu4, cpu7] MHz] every 8 received tokens (story runs, r5)
```

Reading a `cpu-prof` line (A/B runs from r5: `cpu-prof ab=A blk=3 graphs=8 …`, one per block): `… graphs=32 tokens=32 … us/token=98176 compute=74768 wait=23016 nodes=1963 barriers=818 mmid_thr=22387/23027/22477 | MUL_MAT_ID 120:34214+16861 …` —
`mmid_thr` = µs each thread spent computing inside the expert mat-muls per token (spread = imbalance);
after `|`, per op type: count : µs computing + µs waiting at the sync after it (thread 0). `skip=60/1%` =
expert-skip threshold / share of routed experts skipped, both on that side; `kavg` = experts kept per layer and token on that side.
`x4`, `ilp`, `pf`, `mmid` = kernel and no-barrier knobs in effect. `anon=16/68%/11.2` = cache slots per expert
tensor on that side (0 = not read) / routed experts of decode tokens found cached / tensor copies per decode token
(Granite: 3 tensors per expert, ~0.34 MB each); `anon_mb` = anonymous memory the cache holds.
`majflt`, `minflt` per run divided by `tokens` = faults per token (the parser prints both).

## 실행 안내 (한국어)

**이번 회차(r5)에서 바뀐 것:** 상태 복사 생략(`SSM_INPLACE`, 출력은 완전히 동일)이 이제 기본으로 켜져 있습니다. A/B는 이제
200토큰 이야기 하나 안에서 8토큰씩 A·B·B·A 순서로 번갈아 잽니다(앞쪽이 늘 불리하던 순서 편향을 없애기 위해서입니다).
새 시험 `CONV_RING`(출력 동일)을 한 번 잽니다. 전문가 생략(expert skip)은 승인된 115와, 품질 기준(+2 %) 안에서 가장 큰 값
118(품질 기준 +2 % 안에서 가장 큰 값, 전체 PPL 9.7952)로 잽니다. 항상 켜는 줄도 118입니다. **"vanilla" 두 줄은 비교용으로, Vollie의 속도 기능을 끈 기본 llama.cpp 설정으로 잰 것입니다**(같은 모델,
추가 다운로드 없음). 메모리를 추가로 쓰는 단계는 없습니다.

**준비:** Vollie 2.2.10+33 설치(APK, `vollie-2.2.10+33-research-r5.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다.
전화기 설정은 아무것도 바꾸지 마세요(다른 앱도 그대로). 이번 회차는 추가 다운로드가 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭.
2. 약 15분 동안 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 50-150 KB). 같은 내용이 앱 파일 폴더에 `research-r5-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
