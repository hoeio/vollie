# Research kit (round 7) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.12+35, kit `r7`; round 6 was
2.2.11+34 / `r6`, results in `reports/03_final.md` R27). It measures the round-7 runs on the phone as it is, and
puts one JSON report on the clipboard ("Copy research results"). `tools/research_parse.py` turns that JSON into
`results.csv` rows and R-log text.

What changed in the round-7 build (CPU module, R28/R29): every `cpu-prof` line (each ABBA block too) carries
`majflt=` / `minflt=` = the process's page faults per token during that line's graphs (`getrusage`, per side, not
cumulative), so the skip's fault ratio is read per block instead of per run (R27: per-run faults could not tell
read-ahead from day state). New, **off unless set, NOT bit-exact** (declared numerics experiment):
`GGML_HEAD2=<M>` two-stage lm_head (`reports/memos/head2stage_fix.md`): a low-bit copy of the Q6_K head
(`GGML_HEAD2_Q` = `q2_k` 50.6 MB / `q3_k` 66.2 MB, anonymous memory; made only in head2 runs, on the prompt's head op, i.e. before the first generated token and before
ABBA block 0; kept across back-to-back head2 runs; freed by the first run with `GGML_HEAD2` off and no HEAD2 A/B) scores all
100,352 rows, the M best are recomputed exactly. `cpu-prof` field `head=` = µs per token of the head op on that side,
`head2=` = M on that side. Every earlier default (`CONV_RING`, skip-aware read-ahead, …) is unchanged.

## What one run does (about 15 minutes, no settings to change; head2 runs hold 50.6 MB more)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 2 | official benchmark prompt, 32 tokens, saved settings (defaults as in r6: ILP + SSM in place + conv ring, no skip) | — |
| `vanilla` | 2 | **stock llama.cpp settings, for comparison**: official 32 and story 200 with every Vollie knob that has an off switch off (list below), threads as saved | env for that run only |
| `ab` | 4 | ABBA A/B over the 200-token story, A = skip 118 in the last two: `EXPERT_SKIP:0:118` (eio 1, default), `EXPERT_SKIP:0:118 · eio 0` (`GGML_EXPERT_IO=0`; the per-block `majflt`/`minflt` of both sides at read-ahead on and off), `EXPERT_SKIP:118:<s>,HEAD2:0:<M> <q>` (118 → the combo `_head2Combo` in `research.dart`: skip `<s>` + head2 `<q>/<M>`, full-set PPL ≤ 9.797), `HEAD2:0:8192 q2_k · skip 118` (head2 alone on top of 118, the generalizable number; full-set PPL 9.6237 without skip) | `GGML_CPU_AB=knob:…`, `GGML_HEAD2_Q` |
| `on` | 5 | always-on rows: story 200 with the combo (`GGML_EXPERT_SKIP=<s>`, `GGML_HEAD2=<M>`, `GGML_HEAD2_Q=<q>`), story 200 with nothing (control), story 200 `eio 0`, story 200 combo · eio 0, official 32 with the combo | `GGML_EXPERT_SKIP`, `GGML_HEAD2`, `GGML_HEAD2_Q`, `GGML_EXPERT_IO` |

The parser's A/B Δ is always B − A, per side mean ± sd over the post-warm-up blocks, ± 1σ, and the per ABBA group
(B̄ − Ā) mean and sd; runs of one name are pooled, so eio 1 and eio 0 stay apart; the `MUL_MAT` line includes the head. Every story run also logs `mhz_blocks`: the cpu4
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
(Granite: 3 tensors per expert, ~0.34 MB each); `anon_mb` = anonymous memory the cache holds. `adv` (r6) = expert tensor slices given `MADV_WILLNEED` per token on
that side (0 when `GGML_EXPERT_IO=0`); r5 lines have no `adv`. `majflt`, `minflt` (r7) = page faults per token during
that line's graphs, per side; `head` = µs per token of the output head op; `head2` = GGML_HEAD2's M on that side (0 = stock).
The run's own `majflt`, `minflt` divided by `tokens` = faults per token over the whole run (the parser prints both).

## 실행 안내 (한국어)

**이번 회차(r7)에서 바뀐 것:** 이제 A/B의 8토큰 묶음마다 페이지 폴트 수가 따로 기록됩니다. 전문가 생략 118의 A/B를
전문가 미리 읽기를 켠 채(기본)와 끈 채("eio 0") 한 번씩 잽니다. 새로 **두 단계 출력층(head2)**을 시험합니다: 마지막 단어 고르기
층을 먼저 가볍게 훑어 보고 후보만 정확히 다시 계산합니다. 이것은 출력이 아주 조금 달라지는 실험이라(품질 기준 +2 % 안,
전체 PPL 9.797 이하), A/B 두 줄(생략 118 → 생략+head2, 생략 118 → 118+head2)과 "story 200 · skip … + head2 …" 줄들,
공식 32토큰 줄에서만 켜집니다. head2 줄은 메모리를 약 51 MB 더 쓰고, 처음 켤 때 몇 초 준비 시간이 걸립니다.
head2 사본은 head2를 켠 실행에서만 첫 토큰 전에 만들어지고, head2를 끈 다음 실행이 시작될 때 지워집니다(다른 줄에는 남지 않습니다). **"vanilla" 두 줄은
비교용으로, Vollie의 속도 기능을 끈 기본 llama.cpp 설정으로 잰 것입니다**(같은 모델, 추가 다운로드 없음).

**준비:** Vollie 2.2.12+35 설치(APK, `vollie-2.2.12+35-research-r7.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다.
전화기 설정은 아무것도 바꾸지 마세요(다른 앱도 그대로). 이번 회차는 추가 다운로드가 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭.
2. 약 15-20분 동안 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 50-150 KB). 같은 내용이 앱 파일 폴더에 `research-r7-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
