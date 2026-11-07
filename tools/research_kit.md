# Research kit (round 13) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.17+40, kit `r13`; rounds 11-12 were
2.2.16+39 / `r11`). It measures the round-13 runs on the phone as it is (about 10-11 minutes; the button label is
unchanged), and puts one JSON report on the clipboard ("Copy research results"). `tools/research_parse.py` turns that
JSON into `results.csv` rows and R-log text; `--gate 10` adds the block-gated score.

What changed in the round-13 build (pre-registration R39 in `reports/03_final.md`; Bonnie's memo
`reports/memos/round13_plan.md`; build and checks `lab/logs/r39/`):

- **Default unchanged from r11** (compiled in, not a saved setting): `GGML_EXPERT_SKIP=117 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0
  VOLLIE_SAMPLER_BYPASS=1` (`Research.served`) and the module's `GGML_Q3K_KERN=1`.
- **New knob `GGML_MMID_TOUCH` (default off, bit-identical):** in the one-token expert path (`GGML_MMID=3`), after a
  thread claims a 32-row chunk and before the kernel runs, one demand load of the first byte of each 4-KB page the chunk's
  weight rows span (gate/up 5-6 pages, down 2-3), so the page walks overlap. Identity: `lab/logs/r39/id.out` (2 × 10⁵
  random row-quads 0 differ; PPL chunks 1-3 at −ub 1 / −ub 32 / −b 512 equal to r9-r11 on every digit, stock and served,
  touch unset / '' / 1; greedy 32 on both kit prompts identical to the r34 text). `cpu-prof` lines carry `touch=` after
  `kern=`; the ABBA mechanism takes `knob:MMID_TOUCH:A:B`.
- **Block order `GGML_CPU_AB_BAAB=1`:** post-warm-up blocks go B A A B … (warm-up still A; `ab=` still names the side,
  A = the first value). The kit runs the ABBA three times: ABBA, BAAB, ABBA (R39: cancels the position-1 fault advantage).
- **`cpu-bench`:** the bench thread is pinned to cpu 4-6 for the bench only (`pinned=1`, `0` if the pin failed; r12's
  landed on an A55) and each line adds `hot=` (= `new`) `cold=` (the same kernel on a fresh slice of a 64-MB anon arena
  each trial, slices ≥ 8 MB apart, 32-row chunks of 4-row groups) `cold_touch=` (cold + the knob's page-head loads at each
  chunk start) `cold_thp=` (cold on a 2-MB-aligned arena with `MADV_HUGEPAGE`; diagnostic) with `exact_cold*` flags and
  `anon_huge_kb=` (AnonHugePages from `/proc/self/smaps_rollup`, −1 unreadable). The parser prints `cold/hot` and
  `(cold − cold_touch)/(cold − hot)`. The bench maps one 64-MB arena at a time (freed before the next).
- **Dropped:** the skip control, vanilla, the second baseline, both always-on stories.

## What one run does (about 10-11 minutes, no settings to change)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 1 | official benchmark prompt, 32 tokens, saved settings + the default | — |
| `bench` | 1 | the kernel timing: a one-token reply ("Hi") with `GGML_CPU_BENCH=1`; its two `cpu-bench q3k` lines are in `prof` | `GGML_CPU_BENCH=1` |
| `ab` | 3 | ABBA `MMID_TOUCH:0:1` over the 200-token story on the default, `GGML_CPU_PROF=1`, block orders ABBA / BAAB / ABBA (one name: the parser pools them) | `GGML_CPU_AB=knob:MMID_TOUCH:0:1` (+ `GGML_CPU_AB_BAAB=1` on the second) |
| `on` | 2 | official 32 on the default ×2 (reported with clock and faults; not a scored line) | — |

Memory: as r9 (the default holds the 86.7 MB head copy); the knob allocates nothing; the bench mallocs ≈ 7 MB of synthetic
slices per shape plus one 66-MB arena at a time, all freed before the reply. Load-time copy: head ≈ 0.23-0.34 s (r8 device).

The parser's A/B Δ is always B − A, per side mean ± sd over the post-warm-up blocks, ± 1σ, and the per ABBA group
(B̄ − Ā) mean and sd; runs of one name are pooled, so eio 1 and eio 0 stay apart; the `MUL_MAT` line includes the head. Every story run also logs `mhz_blocks`: the cpu4
and cpu7 clock every 8 received text chunks (r12: 3 entries per 200-token story, so several blocks share one entry).
`--gate T` (r13, R39 §B, class-K knobs): a block counts iff its `majflt` per token ≤ T and its `mhz_blocks` entry (mapped
by token position) is the run's modal clock; per run and pooled over runs of one name: counted blocks per side, side
majflt means, Δ ± σ of `MUL_MAT_ID` (compute + wait), `MUL_MAT`, head, `SSM_SCAN`; flags ≥ 8 per side, side means within
3, σ ≤ 1.0 → `scorable`, else `unresolved`; an untouched line beyond ± 0.3 is printed as a bias flag.

**Vanilla rows (r11 and earlier; not in r13), what is off:** `GGML_TINY_OP=0`, `GGML_MMID=0`, `GGML_Q3K_X4=0`, `GGML_Q3K_VEC=0`, `GGML_Q3K_ILP=0`,
`GGML_SSM_INPLACE=0`, `GGML_CONV_RING=0`, `GGML_EXPERT_SKIP=0`, `GGML_EXPERT_IO=0` (expert prefetch and next-layer
prediction off), `GGML_HEAD2=0`, `VOLLIE_SAMPLER_BYPASS=0`, `GGML_Q3K_KERN=0`; `GGML_Q3K_PF`, `GGML_EXPERT_ANON`, `GGML_EXPERT_TOPP/GAP` are off anyway. **What stays Vollie's in
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
  prof                 [str] cpu-prof / cpu-bench / repack-release lines the module printed during the run (r10: the
                       bench run's two "cpu-bench q3k" lines; r11: each ends cpu=<core> mhz=<start>/<end>; r13: + hot=
                       cold= cold_touch= cold_thp= exact_cold*= anon_huge_kb= and pinned= before mhz=)
  text                 first 80 chars of the reply
  mem_before_load      (release/after load) memInfo() before the reload
  error                exception text if the run failed
  mhz_blocks           [[cpu4, cpu7] MHz] every 8 received tokens (story runs, r5)
  sample_ms, decode_ms (r9) llamadart's sampling (+ token-to-piece) and llama_decode time over the reply
  warmup_ms, warmup    (r9) the one-token warm-up before the run (only when its module copies differ) and its requant lines
```

Reading a `cpu-prof` line (A/B runs from r5: `cpu-prof ab=A blk=3 graphs=8 …`, one per block): `… graphs=32 tokens=32 … us/token=98176 compute=74768 wait=23016 nodes=1963 barriers=818 mmid_thr=22387/23027/22477 | MUL_MAT_ID 120:34214+16861 …` —
`mmid_thr` = µs each thread spent computing inside the expert mat-muls per token (spread = imbalance);
after `|`, per op type: count : µs computing + µs waiting at the sync after it (thread 0). `skip=60/1%` =
expert-skip threshold / share of routed experts skipped, both on that side; `kavg` = experts kept per layer and token on that side.
`x4`, `ilp`, `pf`, `mmid` = kernel and no-barrier knobs in effect. `anon=16/68%/11.2` = cache slots per expert
tensor on that side (0 = not read) / routed experts of decode tokens found cached / tensor copies per decode token
(Granite: 3 tensors per expert, ~0.34 MB each); `anon_mb` = anonymous memory the cache holds. `adv` (r6) = expert tensor slices given `MADV_WILLNEED` per token on
that side (0 when `GGML_EXPERT_IO=0`); r5 lines have no `adv`. `majflt`, `minflt` (r7) = page faults per token during
that line's graphs, per side; `head` = µs per token of the output head op; `head2` = GGML_HEAD2's M on that side (0 = stock);
`chunk` (r9) = GGML_MM_CHUNK on that side; `expq4` (r9) = layers reading the expert Q4_0 copy on that side; `kern`
(r10) = GGML_Q3K_KERN on that side; `touch` (r13) = GGML_MMID_TOUCH on that side.
The run's own `majflt`, `minflt` divided by `tokens` = faults per token over the whole run (the parser prints both).

## 실행 안내 (한국어)

**이번 회차(r13)에서 바뀐 것:**
- **기본 설정은 지난 회차와 같습니다**(전문가 생략 117 + 두 단계 출력층 + 샘플러 빠른 경로 + 새 전문가 커널). 저장된 설정도 그대로.
- **새 실험 스위치 하나(`MMID_TOUCH`)를 A/B로 잽니다** — 전문가 가중치를 읽기 직전에 메모리 페이지 머리를 먼저 한 번씩
  건드려, 페이지를 찾는 시간을 겹치게 하는 방법입니다. 답은 한 글자도 달라지지 않습니다(컴퓨터에서 확인). 같은 A/B를
  세 번 돌리고, 두 번째는 순서를 뒤집어(BAAB) 순서 때문에 생기는 치우침을 없앱니다.
- **커널 시간 재기(`cpu-bench`)** — 이번에는 큰 코어(4-6번)에 고정해서 재고, 캐시에 없는(차가운) 메모리에서 읽을 때의
  시간도 함께 잽니다(몇 초, 모델이나 답에는 아무 변화 없음).
- 시간을 줄이려고 비교 기준 A/B, vanilla, 두 번째 기준 측정, 긴 이야기 두 개를 뺐습니다.

**가장 중요한 한 가지: 실행하는 동안(약 10-11분) 전화기를 만지지 마세요.** 다른 앱을 열거나 화면을 끄지 말고 그대로 두세요.

**준비:** Vollie 2.2.17+40 설치(APK, `vollie-2.2.17+40-research-r13.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다.
전화기 설정은 아무것도 바꾸지 마세요(다른 앱도 그대로). 이번 회차는 추가 다운로드가 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭(버튼 이름은 그대로지만 이번에는 약 10-11분 걸립니다).
2. 끝날 때까지 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 30-100 KB). 같은 내용이 앱 파일 폴더에 `research-r13-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
