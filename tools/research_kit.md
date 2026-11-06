# Research kit (round 10) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.15+38, kit `r10`; round 9 was
2.2.14+37 / `r9`). It measures the round-10 runs on the phone as it is (about 15 minutes), and puts one JSON report on
the clipboard ("Copy research results"). `tools/research_parse.py` turns that JSON into `results.csv` rows and R-log text.

What changed in the round-10 build (pre-registration `reports/05_compute_less.md` §9.9; Bonnie's memo
`reports/memos/expert_kernel.md`; Tom's audit `lab/logs/r33/q3k_kernel/README.md`; build and checks `lab/logs/r34/`):

- **Default (compiled into this build, not a saved setting):** `GGML_EXPERT_SKIP=117 GGML_HEAD2=2048 GGML_HEAD2_Q=q4_0`
  as r9, **plus `VOLLIE_SAMPLER_BYPASS=1`** (r9 (iii): exact, the sampler's fast path at temp 0). `Research.served` sets
  them at every model load and is the base of every kit run; the vanilla rows set `VOLLIE_SAMPLER_BYPASS=0` (stock, as r9).
- **(i) `GGML_Q3K_KERN=1`** (default 0 = the r9 `GGML_Q3K_ILP` path; **bit-identical**): the Q3_K expert dot of the
  `MMID=3` path as an in-register 4-row kernel (the 4 rows' qs/hmask word-transposed, `sdot` by lane, the plane value
  built in place with shift+and+`bit`/`sri`, the bias folded into the accumulator seed, scales pre-shifted so one
  accumulator holds 16 × the block sum; the float step is the stock one). Hand-scheduled asm: device build **377 NEON ops
  per 4-row block vs 562** (no `movi`, no spills; `lab/logs/r34/disasm/counts.txt`). Identity: 2 × 10⁵ random row-quads
  bitwise, PPL chunks 1-3 at −ub 1 / −ub 32 / −b 512 to every digit, greedy 32 on both kit prompts (`lab/logs/r34/id.out`).
  ABBA name `Q3K_KERN`; `cpu-prof` lines gain `kern=` (the knob on that side).
- **(ii) `GGML_MMID_FUSE`** (gate + up in one pass) was **not built** this round; not in the kit.
- **(iii) `GGML_CPU_BENCH=1`**: a **1-thread kernel timing, no model change**: on the first graph after it is set, the
  calling thread times, cache-hot on synthetic expert slices (random Q3_K rows of the served shapes), ns per row of: `old`
  (the r9 x4 ILP kernel), `new` ((i)), `int8` (the rows pre-expanded to int8, the "unpack-free" reference), `q4x4` (the
  repacked Q4_0 4x4 gemv of `repack.cpp`, the byte-rate reference) and `scalar` (a prototype that builds planes 0-1 of the
  next block on the integer units into a 512-byte L1 buffer while NEON runs this block). Best of 3 rounds × 7 trials,
  kernels interleaved. Two lines: `cpu-bench q3k threads=1 K=1536 rows=512 old=… new=… int8=… q4x4=… scalar=…
  exact_new=1 exact_int8=1 exact_scalar=1 cpu=<core>` and the same for `K=512 rows=1536` (`exact_*` = output bitwise equal
  to `old`; `cpu` = the core it ran on: 4-6 = A78, 7 = X1). The parser prints them under `== cpu-bench` with new/old,
  int8/old, scalar/old and old/q4x4.

## What one run does (about 15 minutes, no settings to change)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 2 | official benchmark prompt, 32 tokens, saved settings + the default | — |
| `bench` | 1 | the kernel timing (iii): a one-token reply ("Hi") with `GGML_CPU_BENCH=1`; its two `cpu-bench q3k` lines are in `prof` | `GGML_CPU_BENCH=1` |
| `vanilla` | 2 | **stock llama.cpp settings, for comparison**: official 32 and story 200 with every Vollie knob that has an off switch off (list below), threads as saved | env for that run only |
| `ab` | 2 | ABBA A/B over the 200-token story on the default, `GGML_CPU_PROF=1`: (a) `Q3K_KERN:0:1` (exact; scored on `MUL_MAT_ID` compute), (c) `EXPERT_SKIP:0:118` with head2 off and skip 0 on the prompt (the standing control, as r8/r9) | `GGML_CPU_AB=knob:…` |
| `on` | 4 | always-on: story 200 default (the day's control), story 200 default + `GGML_Q3K_KERN=1`; official 32 default, official 32 default + `GGML_Q3K_KERN=1` | as named |

Memory: as r9 (the default holds the 86.7 MB head copy); (i) and (iii) allocate nothing on the model path (the bench
mallocs ≈ 7 MB of synthetic slices for its two shapes and frees them). Load-time copy: head ≈ 0.23-0.34 s (r8 device).

The parser's A/B Δ is always B − A, per side mean ± sd over the post-warm-up blocks, ± 1σ, and the per ABBA group
(B̄ − Ā) mean and sd; runs of one name are pooled, so eio 1 and eio 0 stay apart; the `MUL_MAT` line includes the head. Every story run also logs `mhz_blocks`: the cpu4
and cpu7 clock every 8 tokens.

**Vanilla rows, what is off:** `GGML_TINY_OP=0`, `GGML_MMID=0`, `GGML_Q3K_X4=0`, `GGML_Q3K_VEC=0`, `GGML_Q3K_ILP=0`,
`GGML_SSM_INPLACE=0`, `GGML_CONV_RING=0`, `GGML_EXPERT_SKIP=0`, `GGML_EXPERT_IO=0` (expert prefetch and next-layer
prediction off), `GGML_HEAD2=0`, `VOLLIE_SAMPLER_BYPASS=0`; `GGML_Q3K_PF`, `GGML_Q3K_KERN`, `GGML_EXPERT_ANON`, `GGML_EXPERT_TOPP/GAP` are off anyway. **What stays Vollie's in
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
                       bench run's two "cpu-bench q3k" lines)
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
(r10) = GGML_Q3K_KERN on that side.
The run's own `majflt`, `minflt` divided by `tokens` = faults per token over the whole run (the parser prints both).

## 실행 안내 (한국어)

**이번 회차(r10)에서 바뀐 것:**
- **기본 설정:** 지난 회차와 같은 조합(전문가 생략 117 + 두 단계 출력층)에, 지난 회차에 결과가 완전히 같다고 확인된
  **샘플러 빠른 경로**가 기본으로 들어갔습니다(이 빌드에만 들어 있는 기본값, 저장된 설정은 그대로).
- **(i) 전문가 계산 커널을 새로 짠 버전(`Q3K_KERN`)** — 모델 파일의 같은 숫자를 같은 순서로 더하는, 결과가 완전히 같은
  계산을 더 적은 명령으로 하는 실험입니다. A/B 한 줄과 상시 켠 줄(이야기 200, 공식 32)로 잽니다.
- **(iii) 커널 시간 재기(`cpu-bench`)** — 한 스레드로 전문가 계산 커널 몇 가지의 속도만 재는 짧은 측정입니다(1초 안팎,
  모델이나 답에는 아무 변화 없음). 결과는 `cpu-bench q3k …` 두 줄로 남습니다.
- "EXPERT_SKIP:0:118" A/B 한 줄은 지난 회차와 똑같은 비교 기준입니다. **"vanilla" 두 줄은 비교용으로, Vollie의 속도
  기능을 끈 기본 llama.cpp 설정입니다**(같은 모델, 추가 다운로드 없음).

**준비:** Vollie 2.2.15+38 설치(APK, `vollie-2.2.15+38-research-r10.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다.
전화기 설정은 아무것도 바꾸지 마세요(다른 앱도 그대로). 이번 회차는 추가 다운로드가 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭.
2. 약 15분 동안 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 50-150 KB). 같은 내용이 앱 파일 폴더에 `research-r10-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
