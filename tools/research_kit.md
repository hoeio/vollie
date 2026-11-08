# Research kit (round 15) — what it runs, the result file, how to run it

The kit is the **"Run research kit (~15 min)"** button in Vollie's Settings (app 2.2.20+43, kit `r15`; round 14 was
2.2.19+42 / `r14`). APK: `lab/dist/vollie-2.2.20+43-research-r15.apk`, 17,142,970 B, sha256
`3a9a76afed833577262dfdcd17e40776695db1ca03f221cfcefc485ff2b3d4cf` (the kernel line is in: `Research.kernLine` = true). Round 15 is the **R46 pre-registration: a measurement round, about 8-10 minutes** (`reports/03_final.md`
R46; build `lab/logs/r46/`). It puts one JSON report on the clipboard ("Copy research results"); read it with
`python3 -I tools/research_parse.py <json> --pairs --gate 10 --sigma 0.4`.

What changed in the round-15 build (compiled defaults unchanged: skip 117, head2 q4_0/2048, sampler bypass, Q3K_KERN=1;
MMID_TOUCH, MMID_FUSE, MMID_CHUNK2, the router cast off):

- **`GGML_CPU_AB_TOKEN=1`** (off unless set): with `GGML_CPU_AB=knob:<NAME>:<A>:<B>` the knob flips per decode graph (= per
  token) in the order A B B A A B B A … (block 1, no warm-up), so pair k = tokens (2k−1, 2k) holds one A and one B. One short
  `cpu-prof` line per token (below). `knob:<NAME>:0:0` is the A/A null: the same code path, both labels side A's value.
- **`GGML_CPU_AB=knob:ROUTER_CAST:0:1`** (C1-F16, R43, a **declared numerics experiment** on side B): the 40 MoE routers
  (`ffn_gate_inp`, F32 in the file) get one F16 copy in anonymous memory (7.9 MB, made at the run's prompt graph, kept until
  exit); side B's router `MUL_MAT`s read it. `GGML_ROUTER_CAST=f16` casts every graph (the kit's official-prompt check).
  Every `cpu-prof` line now ends its header with `router=<µs/token of the router MUL_MATs> rcast=f32|f16`.
- The `cpu-bench` guard re-arms per kit run (the kit passes its own `GGML_CPU_BENCH` value; R45).
- Consecutive A/B story runs wait only while the thermal headroom is ≥ 0.85 (other runs still cool to < 0.65, ≤ 90 s).
- `GGML_HEAD2_LOWRANK` (C2) and `GGML_SHEXP_PRUNE` (C4) are compiled in, off, and never set by the kit.

## What one run does (about 8-10 minutes, no settings to change)

| phase | runs | what | knob |
|---|---|---|---|
| `baseline` | 1 | official benchmark prompt, 32 tokens, saved settings + the default (its text = the F32 reference) | — |
| `bench` | 1 | the kernel timing (r13b format, a core check): a one-token reply ("Hi") | `GGML_CPU_BENCH=<kit's value>` |
| `warmup` | 1 | story 200 at the default, **dropped, unscored** (its text = the F32 story reference) | — |
| `ab` | 1 | story 200 per token, **A/A null** | `GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:0:0`, `GGML_CPU_AB_TOKEN=1` |
| `ab` | 3 | story 200 per token, the fuse | `knob:MMID_FUSE+CHUNK2:0:1`, `GGML_CPU_AB_TOKEN=1` |
| `ab` | 2 | story 200 per token, the kernel (`Research.kernLine` = true: the owner said yes, R46) | `knob:Q3K_KERN:0:1`, `GGML_CPU_AB_TOKEN=1` |
| `ab` | 2 | story 200, block ABBA / BAAB (warm 16, block 8), the fuse (r14 continuity) | `knob:MMID_FUSE+CHUNK2:0:1` (+ `GGML_CPU_AB_BAAB=1`) |
| `on` | 2 | official benchmark prompt, 32 tokens, the default (reported with clock and faults, not scored) | — |
| `ab` | 2 | **last (R46 am. 1 (4))**: story 200, block ABBA / BAAB, C1-F16; `text_same` / `text_diverge_at` reported only | `knob:ROUTER_CAST:0:1` (+ BAAB) |
| `check` | 1 | official 32 with the cast on every token; `text_same` must be true (the C1 voiding check) | `GGML_ROUTER_CAST=f16` |

The C1 runs come last because the 7.9 MB router copy stays allocated until the process exits; no printed run follows it.
Report-level `router_text_same` = the `check` run's `text_same` (false voids the C1 runs, R46 item 2 + amendment 1 (1)); a
200-token story diverging from the F32 text is a reported fact, not a void.
Parser rules from R46 amendment 1: the A/A null is valid iff every scored line has |z| ≤ 3, Σz² over the seven lines ≤ 14.07
(χ²₇ at 0.95) and the minflt ratio is 1.00 ± 0.05; the interleaved A-token vs block-ABBA A mean is reported, not voiding.
Size: one per-token line ≈ 210-220 characters → ≈ 44 KB per interleaved run; the whole JSON ≈ 240-260 KB without the kernel
line, ≈ 330-350 KB with it (host replay 319 KB with it).

Memory: as r9 (the default holds the 86.7 MB head copy); the bench mallocs ≈ 7 MB of synthetic slices per shape plus one
48-MB arena, all freed before the reply. Load-time copy: head ≈ 0.23-0.34 s (r8 device).

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
                       cold= cold_touch= cold_thp= exact_cold*= anon_huge_kb= and pinned= before mhz=; r13b: hot= cold=
                       cold_touch= exact_hot/cold/cold_touch=, instrument= (K 512), cpu=a-b migr= cpu_ok=n/21 pinned=
                       mhz=min/max; no cold_thp / anon_huge_kb)
  text                 first 80 chars of the reply
  mem_before_load      (release/after load) memInfo() before the reload
  error                exception text if the run failed
  mhz_blocks           [[cpu4, cpu7] MHz] every 8 received tokens (story runs, r5)
  sample_ms, decode_ms (r9) llamadart's sampling (+ token-to-piece) and llama_decode time over the reply
  warmup_ms, warmup    (r9) the one-token warm-up before the run (only when its module copies differ) and its requant lines
  text_same            (r15, C1-F16 runs) the whole reply equals the F32 reference run's
  text_diverge_at, text_len   (r15, when text_same is false) first differing character; [this reply, reference] lengths
router_text_same       (r15, report level) every C1-F16 run's text_same
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
(r10) = GGML_Q3K_KERN on that side; `touch` (r13) = GGML_MMID_TOUCH on that side; `fuse`, `chunk2` (r14) = GGML_MMID_FUSE /
GGML_MMID_CHUNK2 on that side; `router`, `rcast` (r15) = µs per token of the router `MUL_MAT`s / their weight type on that side.
Per-token line (r15, `GGML_CPU_AB_TOKEN=1`): `cpu-prof ab=B tok=2 us/token= head= router= majflt= minflt= mhz=<cpu4>/<cpu7>
nodes= barriers= kern= fuse= chunk2= | …` (no compute= / wait= / rcast=: the op table and the run's env carry them) — faults are that token's counts (not per-token means), `mhz` is read
from sysfs right after the token (-1 where there is none), the op table holds only the scored ops `MUL_MAT_ID`, `MUL_MAT`, `GLU`,
`SSM_SCAN` (by cost).
The run's own `majflt`, `minflt` divided by `tokens` = faults per token over the whole run (the parser prints both).

## 실행 안내 (한국어)

**이번 회차(r15)는 약 8-10분 걸립니다.** 이번에는 새 기능을 켜는 회차가 아니라 **측정 방법을 바꾸는 회차**입니다. 지난 두 번은
8토큰 묶음 단위로 켠 쪽(B)과 끈 쪽(A)을 번갈아 쟀는데, 이번에는 **토큰 하나마다** 번갈아 재서 같은 시간에 훨씬 많은 비교를
얻습니다. 같은 쪽을 두 이름으로 재는 "비교 대조" 한 번, 지난 회차의 묶음 비교 두 번, 그리고 라우터 가중치를 F16으로 읽는
변경(답이 한 글자라도 바뀌면 그 결과는 무효 처리됩니다) 비교가 들어 있습니다. 기본 설정과 저장된 설정은 그대로입니다.
- 메모리가 부족한 날이어도 상관없습니다. 언제 돌려도 됩니다.

**가장 중요한 한 가지: 실행하는 동안(약 8-10분) 전화기를 만지지 마세요.** 다른 앱을 열거나 화면을 끄지 말고 그대로 두세요.

**준비:** Vollie 2.2.20+43 설치(APK, `vollie-2.2.20+43-research-r15.apk`, 17,142,970 바이트, sha256 `3a9a76af…b3d4cf`), 모델이 이미 받아져 있고 평소처럼 쓰는 상태면 됩니다.
전화기 설정은 아무것도 바꾸지 마세요(다른 앱도 그대로). 추가 다운로드는 없습니다.

1. Vollie → 설정(Settings) → **"Run research kit (~15 min)"** 탭(버튼 이름은 그대로지만 이번에는 약 8-10분 걸립니다).
2. 끝날 때까지 화면을 켠 채로 둡니다(앱이 켜둡니다). 진행 상황은 버튼 아래에 `Research · 분:초 · 단계`로 보입니다.
3. 끝나면 **"Copy research results"** 탭 → 클립보드에 JSON이 복사됩니다. 그대로 제게 붙여 보내 주세요
   (이번에는 길이가 약 240-350 KB로 평소보다 깁니다). 같은 내용이 앱 파일 폴더에 `research-r15-<시각>.json`으로도 저장됩니다
   — 붙여 넣기가 잘리거나 복사가 안 되면 그 파일을 보내 주세요.
4. 중간에 멈추려면 **"Stop research run"**. 그때까지의 결과도 복사할 수 있습니다.

**보내 주실 것:** 붙여 넣은 JSON 한 개. 가능하면 실행 직전의 전화기 상태 한 줄(예: 충전 중 여부, 방금
다른 무거운 앱을 썼는지)도 함께 적어 주세요.

**주의:** 결과 파일에는 성능 숫자와 기기 정보(칩, 메모리, 클럭)만 담기고 대화 내용은 들어가지 않습니다.
