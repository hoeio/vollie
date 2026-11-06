# Research kit (round 9) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.14+37, kit `r9`; round 8 was
2.2.13+36 / `r8`, results in `reports/03_final.md` R31). It measures the round-9 runs on the phone as it is (about 16
minutes), and puts one JSON report on the clipboard ("Copy research results"). `tools/research_parse.py` turns that
JSON into `results.csv` rows and R-log text.

What changed in the round-9 build (pre-registration `reports/05_compute_less.md` §7.11, Bonnie's memo
`reports/memos/last_3ms.md`):

- **New default (compiled into this build, not a saved setting):** `GGML_EXPERT_SKIP=117 GGML_HEAD2=2048
  GGML_HEAD2_Q=q4_0` (R31; full-set PPL 9.7886). `Research.served` in `research.dart` sets them at every model load
  and is the base of every kit run (a run's own env overrides it; `GGML_HEAD2=0` = the r8 off path, bit-identical).
  The head copy (86.7 MB) is made **at load** by a one-token warm-up ("Hi", 1 token), not in the first reply's TTFT;
  the kit repeats that warm-up before any run whose module copies differ from the last run's (`warmup_ms`, and the
  requant lines in `warmup`).
- **(i) `GGML_MM_CHUNK=<rows>`** (default 64 = stock; bit-identical): rows per chunk of one-token `MUL_MAT` and of the
  stock one-token `MUL_MAT_ID` path; the `GGML_MMID` no-barrier path (stock 32 rows) takes `MIN(32, n)`. ABBA name
  `MM_CHUNK`. Identity: `lab/logs/r32/id.out`.
- **(ii) `GGML_EXPERT_Q4_LAYERS=<n>`** (default 0 = off, bit-identical; NOT bit-exact when on, budgeted): the routed
  experts (gate/up/down, Q3_K in the file) of the n most-routed layers at skip 117 — **layers 39, 38, 9, 31** in that
  order (host, `lab/logs/r32/rank/`) — get an anonymous Q4_0 copy (`quantize_row_q4_0_ref`, repacked 4x4 like the head
  screen), made at load / in the warm-up (`cpu-prof expq4 requant -> q4_0x4x4 layers=… tensors=12/12 copy_mb=… ms=…`);
  those layers then run the byte-rate `sdot` gemv. ≈ 85 MB per layer. ABBA name `EXPERT_Q4_LAYERS` (both sides' copies
  exist before block 0). **Full-set gate (host, built runtime, `lab/logs/r32/ppl/`): FAILED at every
  size tried** — 117 + head2 + n = 4 (39, 38, 9, 31) **9.8048** (en 11.5415), n = 2 (39, 38) **9.8014** (en 11.5441),
  116 + head2 + n = 4 **9.7977** (en 11.5158), and a data run with `GGML_EXPERT_Q4_LIST=9,31,33,32` **9.8018** (en
  11.5243); budget ≤ 9.797. **Decision (Clyde): (ii) runs at n = 4 as an over-budget research row
  (research only, over budget: 9.8048) — a speed cost item, never served.** `GGML_EXPERT_Q4_LIST=<l,l,…>`
  (test override of the compiled ranking) was added for the data run.
- **(iii) printed overhead** (exact): `VOLLIE_SAMPLER_BYPASS=1` — at temp 0 llamadart drops top-k / top-p / temp
  (greedy takes the same argmax) **and libllama's repetition-penalty sampler visits only the ≤ 64 penalised tokens**
  instead of looking up all 100,352 candidates (host: sampling 0.43 → 0.10 ms/token; the top-k/top-p part alone saved
  nothing measurable; `lab/logs/r32/bypass.out`); `GGML_DENSE_WILLNEED=1` — `MADV_WILLNEED` over the dense weights
  (everything but `*_exps`, head included) at every prompt (`cpu-prof willneed tensors=… mb=… us=…`). Each run now
  records `sample_ms` / `decode_ms` (llamadart's clocks); the parser prints the outside-graph cost per printed token
  (decode wall − Σ windows − unprofiled tokens × steady, ÷ intervals; r8 combo story: 1.63).
- `cpu-prof` lines gain `chunk=` (MM_CHUNK on that side) and `expq4=` (layers read from the copy on that side).
  On an `EXPERT_Q4_LAYERS` B side `adv=` drops by the copied layers' share (their experts are not read ahead).

## What one run does (about 16 minutes, no settings to change)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 2 | official benchmark prompt, 32 tokens, saved settings + the new default | — |
| `vanilla` | 2 | **stock llama.cpp settings, for comparison**: official 32 and story 200 with every Vollie knob that has an off switch off (list below, now with `GGML_HEAD2=0`), threads as saved | env for that run only |
| `ab` | 3 | ABBA A/B over the 200-token story on the default, `GGML_CPU_PROF=1`: (a) `MM_CHUNK:64:16` (exact; scored on the `MUL_MAT` wait), (b) `EXPERT_Q4_LAYERS:0:4` (the expert copy; scored on `MUL_MAT_ID` compute), (c) `EXPERT_SKIP:0:118` with head2 off and skip 0 on the prompt (the standing control, as r8) | `GGML_CPU_AB=knob:…` |
| `on` | 5 | always-on: story 200 default (the day's control), story 200 default + `GGML_MM_CHUNK=16` + `VOLLIE_SAMPLER_BYPASS=1` + `GGML_DENSE_WILLNEED=1` ((i) + (iii)), story 200 default + `GGML_EXPERT_Q4_LAYERS=4`; official 32 default, official 32 default + (i) + (iii) | as named |

Memory: the default holds 86.7 MB (head copy) more than r7 did; the `EXPERT_Q4_LAYERS` runs hold 340 MB more.
Load-time copies: head ≈ 0.23-0.34 s (r8 device), experts ≤ 1.5 s *estimated* (host ×4.4).

The parser's A/B Δ is always B − A, per side mean ± sd over the post-warm-up blocks, ± 1σ, and the per ABBA group
(B̄ − Ā) mean and sd; runs of one name are pooled, so eio 1 and eio 0 stay apart; the `MUL_MAT` line includes the head. Every story run also logs `mhz_blocks`: the cpu4
and cpu7 clock every 8 tokens.

**Vanilla rows, what is off:** `GGML_TINY_OP=0`, `GGML_MMID=0`, `GGML_Q3K_X4=0`, `GGML_Q3K_VEC=0`, `GGML_Q3K_ILP=0`,
`GGML_SSM_INPLACE=0`, `GGML_CONV_RING=0`, `GGML_EXPERT_SKIP=0`, `GGML_EXPERT_IO=0` (expert prefetch and next-layer
prediction off), `GGML_HEAD2=0`; `GGML_Q3K_PF`, `GGML_EXPERT_ANON`, `GGML_EXPERT_TOPP/GAP` are off anyway. **What stays Vollie's in
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
`chunk` (r9) = GGML_MM_CHUNK on that side; `expq4` (r9) = layers reading the expert Q4_0 copy on that side.
The run's own `majflt`, `minflt` divided by `tokens` = faults per token over the whole run (the parser prints both).

## 실행 안내 (한국어)

**이번 회차(r9)에서 바뀐 것:**
- **기본 설정이 바뀌었습니다(이 빌드에만 들어 있는 기본값, 저장된 설정은 그대로):** 전문가 생략 117 + 두 단계 출력층
  (head2, Q4_0 사본, 2,048개 정확 재계산). 지난 회차에서 품질 기준 안에서 더 빠르다고 확인된 조합입니다. 출력층 사본(약
  87 MB)은 이제 **모델을 불러올 때** 짧은 준비 실행("Hi", 1토큰)으로 만들어서, 첫 답의 대기 시간에 들어가지 않습니다
  (불러오기가 약 1초 길어질 수 있습니다).
- **(i) 계산 조각을 더 잘게 나누기(64행 → 16행)** — 결과는 완전히 같고, 스레드들이 서로 기다리는 시간만 줄이는 실험.
- **(ii) 가장 많이 쓰이는 4개 층(39, 38, 9, 31)의 전문가를 불러올 때 Q4_0 사본으로 만들어 빠른 계산법으로 돌리기** — 출력이
  아주 조금 달라지는 실험(연구용, 예산 초과 9.8048: 품질 확인 결과 4개 층은 9.8048, 2개 층은 9.8014, 생략 116과 함께 4개 층은 9.7977로 모두 기준 9.797을 넘습니다. 이 줄들은 속도만 재는 연구용 측정이며 기본 설정이 되지 않습니다). 이 줄들은 메모리를 약 340 MB 더 씁니다. 사본은 해당 실행 직전의
  준비 실행에서 만들어집니다(예상 1.5초 이하).
- **(iii) 화면에 찍히는 속도의 바깥 비용 줄이기** — 다음 단어를 고르는 단계(샘플러)의 불필요한 정렬과 반복 벌점의
  전체 단어 검사를 건너뛰고, 답을 시작할 때 모델의 공통 가중치를 미리 읽어 두게 합니다. 결과 글은 완전히 같습니다.
- "EXPERT_SKIP:0:118" A/B 한 줄은 지난 회차와 똑같은 비교 기준입니다. **"vanilla" 두 줄은 비교용으로, Vollie의 속도
  기능을 끈 기본 llama.cpp 설정입니다**(같은 모델, 추가 다운로드 없음).

**준비:** Vollie 2.2.14+37 설치(APK, `vollie-2.2.14+37-research-r9.apk`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다.
전화기 설정은 아무것도 바꾸지 마세요(다른 앱도 그대로). 이번 회차는 추가 다운로드가 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭.
2. 약 16분 동안 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (길이 50-150 KB). 같은 내용이 앱 파일 폴더에 `research-r9-<시각>.json`으로도 저장됩니다.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
