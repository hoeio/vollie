#!/usr/bin/env python3
"""Campaign 2 research-kit report parser (kits r17 to r26; schemas in campaigns/02-25toks-5pct/reports/memo-10-kit-r17.md,
memo-12-kit-r18.md, memo-14-kit-r19.md, memo-17-kit-r20.md, memo-19-kit-r21.md, memo-20-kit-r22.md, memo-22-kit-r23.md, memo-23-kit-r24.md, memo-26-kit-r25.md and memo-27-kit-r26.md).
Standard library only.

usage: python3 -I tools/c2_research_parse.py <report.json | clipboard text "c2gz:..."> [--min-mhz 1800]
           [--prompt tools/c2_official_prompt.txt] [--sigma 0.4] [--r16 reports/device/r16_2026-11-08.json]
       python3 -I tools/c2_research_parse.py --selftest

Prints the identity checks (model, libraries against the build record, chat template, prompt, greedy texts), one line
per group (n, median / min / max printed tok/s, steady ms/token, clocks, faults, counters, hygiene flags), and for
every A/B run id (kind 'abba') B minus A over counted blocks: per-block mean ms/token of the graph (per-token lines)
and of the wall clock (chunk arrival times), blocks after the warm-up only (blk >= 1), clean blocks only (no major
fault, big-core clock >= --min-mhz where the clock is readable), sigma = sqrt(sA^2/nA + sB^2/nB). Campaign 1's rule:
a sub-1 ms effect needs sigma <= 0.4, else it is unresolved. All numbers are the phone's (*device*).

Kit r19 (campaign 1's runtime): per-token lines in campaign 1's r16 format (rows with tok_format 'c1'), page-cache
residency of the model before and after each run (res_before / res_after: experts / rest / total MB), clock caps, the
gate's headroom series. Hygiene flags per run: 'throttled' (a big core below --min-mhz at run end), 'fault-bound'
(> 100 major faults in the decode tokens of a profiled run, or > --fault-run major faults in a plain run, prefill
included), 'gate timed out' (the headroom stayed at or above the gate for the whole wait), 'gate unknown' (no headroom
reading at all).

Kit r20 (campaign 1's kit r16 protocol on OLMoE): rows carry pos (1-based place in the run list), proto ('r16', 'kw' or
'warm') and phase. The r16 part is printed position by position next to campaign 1's own r16 report (--r16): tok/s,
headroom at the gate, gate wait, idle before, prompt time, A78 clock at run end, model residency / RssFile, Cached,
major faults. Then every wait of a gated run: page-cache MB of the model lost between the previous run's end and this
run's start, with and without keep-warm active in that wait (GGML_KEEP_WARM on in the previous run), and the
keep-warm passes seen (c2-keepwarm lines: count, CPU ms, aborted). The warm block (proto 'warm', no gate) is a
different protocol and is printed apart; its warm-ups are their own groups and are not scored.

Kit r21 (N models, GGML_PREFILL_COLD in ABBA per block): identity per model (hash, libraries at each load, template,
one text across both arms, repack on, no skip/head2, a c2-io line per row), the advice probe, then per model and arm
the scored runs' medians (tok/s, decode and prefill read MB and major faults from the c2-io line, prefill ms, mincore
experts+rest before/after, MB marked), the ABBA pairs on minus off per block and per model, and a side-by-side table
with one column per model.

Kit r22 (N models, arms off / num in ABBA, factor hold per quad): identity per model and arm (one text per arm, md5;
off and num texts differ by design; the num env as in the model table and nowhere in the off arm; c2-io, c2-skip-run
and c2-head-run lines), the I/O source (getrusage inblock, /proc/self/io), RLIMIT_MEMLOCK; per model, arm and hold the
scored runs' medians; the num effect per model as the mean over ABBA quads of mean(num) - mean(off) with SE = sd/sqrt(n
quads), overall and split by hold; the hold effect (hold minus no hold) on the off arm, all runs and the runs whose gate
waited at least --wait-s seconds: tok/s and model MB resident before the run, difference of means with Welch SE; the
hold's cost (MB read back, CPU ms, ms after the gate).

Kit r23 (arms off / num in ABBA, factor threads 3 / 4 per quad, hold in every gated run): r22's identity checks plus one
text per arm and thread count, the load env per model (OLMoE VOLLIE_HEAD2_REPACK=1 at every load, nowhere else), num
runs whose c2-head-run line shows declines (c1-olmoe 0006's c2_declined / c2_why) or no active copy, gated runs without
hold; per model, arm and threads the scored runs' medians; num minus off over ABBA quads (all, per thread count);
threads 4 minus 3 per model and arm (per block, mean of the arm's runs at 4 minus at 3, mean over blocks ± SE; and all
runs, Welch); where the DartWorker threads ran (thread_samples); the hold per gated run and its cost.

Kit r24 (arms off / num in ABBA, factor gate rule r16 / plateau per quad, 3 threads, hold in every gated run, charger
guard): r23's checks plus the gate rule of every row against its spec, why each wait ended, charger pauses and the
runs with a charger at their start or end (listed and flagged, never dropped here); per model, arm and gate rule the
scored runs' medians; num minus off over ABBA quads (all, per gate rule; and without flagged runs); plateau minus r16
per model (tok/s, wait s, MB resident before, MB lost in the wait, headroom at the run's start, big-core clock and cap:
per block paired ± SE, and all runs, Welch); swap-ins and major faults per phase from /proc/vmstat; the idle baseline and
thermal zones; the hold table of r23.

Kit r25 (memo-26, R64: r24 plus the kv factor VOLLIE_KV_DONTNEED=1 in half of the scored runs): r24's checks plus the kv
checks (id suffix against env, kv outside scored runs, the c2-kvclear lines per kind and why, the VOID rule of R64 §4:
released = 0 in a kv run or > 0 in a run without kv, greedy text kv against plain per arm); per model kv minus plain,
paired within block x gate rule x arm and Welch over all runs, with medians, for pre-prompt major faults (run majflt
minus the c2-io window, memo-24 §A2.2), pswpin start -> first chunk, TTFT, decode graph ms/token (c2-io), tok/s, VmSwap
at run end and MB released; a mechanical R64 §4 readout; then r24's tables pooled over kv and on plain runs only. A kit
r25b report (r25's app, libllama with c1-olmoe 0009) carries tag= in its c2-kvclear lines; then the r26 mechanism lines
follow (R65 §4).

Kit r26 (memo-27): one arm per model (Granite num, OLMoE off), r16's gate only, the kv factor in two ABBA quads per
block. r24's checks plus: arms per model, gate rules, kv id suffix and placement, c2-kvclear lines without tag=, the void
check (released > 0 without kv) and the kv runs releasing 0, texts kv against plain; then per model kv minus plain
paired per ABBA quad and Welch (r25's metrics) with an R65 §4 effect readout, the mechanism lines (kv runs with why=ok
in every clear line, why and tag counts per kind), r24's idle baseline and zones, one stats line per model, arm and kv,
charger runs, swap per gated run and the hold table."""
import argparse
import base64
import gzip
import json
import os
import re
import statistics as st
import sys

SHA = '5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb'
BIG = (4, 5, 6, 7)  # Snapdragon 888: cpu4-6 A78, cpu7 X1
HERE = os.path.dirname(os.path.abspath(__file__))
R16 = os.path.join(HERE, '..', 'reports', 'device', 'r16_2026-11-08.json')
KW_RE = re.compile(r'c2-keepwarm pass=(\d+) .*?resident_mb=(\d+) touched=(\d+) wall_us=(\d+) cpu_us=(\d+) aborted=(\d)')
# memo-11 §3 (frozen, unanchored)
TOK_RE = re.compile(r'cpu-prof ab=([AB-]) tok=(\d+) us/token=(\d+) majflt=(\d+) minflt=(\d+) mhz=(\d+)/(\d+) '
                    r'run=([A-Za-z0-9._-]+) prompt=(\d+) blk=(\d+) threads=(\d+) ops=(\d+) barriers=(\d+) syncs=(\d+) '
                    r'skipped=(\d+) pairs=(\d+) head=(\d+) router=(\d+) mm=(\d+) mmid=(\d+) attn=(\d+) other=(\d+) '
                    r'wait=(\d+) \|')
TOK_FIELDS = ['ab', 'tok', 'us', 'majflt', 'minflt', 'mhz4', 'mhz7', 'prompt', 'blk', 'threads', 'ops', 'barriers',
              'syncs', 'skipped', 'pairs', 'head', 'router', 'mm', 'mmid', 'attn', 'other', 'wait']
CLASSES = ['mmid', 'mm', 'head', 'router', 'attn', 'other', 'wait']
COUNTS = ['ops', 'barriers', 'syncs', 'skipped', 'pairs']
SKIP_RE = re.compile(r'c2-skip-run: .*?expert_bytes_skipped=([\d.]+)%')
HEAD_RE = re.compile(r'c2-head-run: .*?k=(\d+) .*?exact_rows_per_col=([\d.]+)')
# campaign 1's per-token line (0001, GGML_CPU_AB_TOKEN=1; r16), kit r19
C1_TOK_RE = re.compile(r'cpu-prof ab=([AB]) tok=(\d+) us/token=(\d+) head=(\d+) router=(\d+) majflt=(\d+) minflt=(\d+) '
                       r'mhz=(-?\d+)/(-?\d+) nodes=(\d+) barriers=(\d+) kern=(\d+) fuse=(\d+) chunk2=(\d+) \|')
C1_FIELDS = ['ab', 'tok', 'us', 'head', 'router', 'majflt', 'minflt', 'mhz4', 'mhz7', 'nodes', 'barriers', 'kern', 'fuse',
             'chunk2']
C1_OP_RE = re.compile(r' (MUL_MAT_ID|MUL_MAT|GLU|SSM_SCAN) (\d+):(\d+)\+(\d+)')
GRAPH_RE = re.compile(r'c2-graph-run: .*?ops_run=([\d.]+) ops_tiny=([\d.]+) barriers=([\d.]+) syncs=([\d.]+)')


def load(path):
    with open(path, encoding='utf-8') as f:
        s = f.read().strip()
    if s.startswith('c2gz:'):
        s = gzip.decompress(base64.b64decode(s[5:])).decode('utf-8')
    return json.loads(s)


def mhz(s):
    m = re.match(r'mhz=([\d,]+)', s or '')
    return [int(x) for x in m.group(1).split(',')] if m else []


def big_min(s):
    v = mhz(s)
    return min(v[i] for i in BIG) if len(v) > max(BIG) else None


def tokens(run, fields=TOK_FIELDS):
    """Per-token records of a run: the compact tuples (r18, r19) or parsed full lines (r17 'prof', r16-style lines).
    Campaign 1's lines (r19) also get the op table: mmid / mm (computing µs) and wait (summed waits) per token."""
    if run.get('tok_format') == 'c1':
        fields = C1_FIELDS
    out = [dict(zip(fields, t)) for t in run.get('tok') or []]
    c1 = [(m, line) for line in run.get('prof') or [] for m in [C1_TOK_RE.search(line)] if m]
    if c1 and not out:
        out = [dict(zip(C1_FIELDS, [m.group(1)] + [int(x) for x in m.groups()[1:]])) for m, _ in c1]
    ops = {}
    for m, line in c1:
        o = {}
        for name, _, comp, wait in C1_OP_RE.findall(line):
            o[{'MUL_MAT_ID': 'mmid', 'MUL_MAT': 'mm', 'GLU': 'glu', 'SSM_SCAN': 'ssm'}[name]] = int(comp)
            o['wait'] = o.get('wait', 0) + int(wait)
        ops[int(m.group(2))] = o
    for t in out:
        t.update(ops.get(t['tok'], {}))
    if out:
        return out
    for line in run.get('prof') or []:
        m = TOK_RE.search(line)
        if m:
            g = m.groups()
            out.append(dict(zip(TOK_FIELDS, [g[0]] + [int(x) for i, x in enumerate(g[1:], 2) if i != 8])))
        elif 'tok=' in line:  # older numeric key=value lines (campaign 1 style)
            f = {k: float(v) for k, v in re.findall(r'(\w[\w/]*)=(-?[\d.]+)', line.split('|')[0])}
            if 'tok' in f:
                out.append({'ab': '-', 'tok': int(f['tok']), 'us': f.get('us/token', 0), 'majflt': f.get('majflt', 0),
                            'blk': int(f.get('blk', 0))})
    return out


def steady_ms(run, fields=TOK_FIELDS):
    tl = [t for t in tokens(run, fields) if t['tok'] >= 2 and t.get('us')]
    if tl:
        return st.median(t['us'] / 1000 for t in tl), 'prof'
    c = run.get('chunk_ms') or []
    d = [b - a for a, b in zip(c, c[1:])]
    return (st.median(d), 'chunk') if d else (None, None)


def decode_majflt(run, fields=TOK_FIELDS):
    tl = tokens(run, fields)
    return sum(int(t.get('majflt', 0)) for t in tl) if tl else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def expected_render(prompt):
    # OLMoE-0125-Instruct's GGUF template: bos_token + '<|user|>\n' + content + '\n' + '<|assistant|>\n'
    return '<|endoftext|><|user|>\n' + prompt + '\n<|assistant|>\n'


def env_of(run):
    return run.get('env_set') or run.get('env') or {}


def checks(r, prompt_file):
    out = []
    m = r.get('model') or {}
    ok = m.get('bytes') == m.get('expected_bytes') and m.get('verified_marker') == SHA and m.get('sha256_end') == SHA
    out.append(f"model {m.get('file')} {m.get('bytes')} B, marker {str(m.get('verified_marker'))[:8]}, "
               f"sha256 at end {str(m.get('sha256_end'))[:8]} -> {'OK' if ok else 'CHECK'}")
    built = (r.get('build') or {}).get('libs') or {}
    for k, v in sorted((r.get('libs') or {}).items()):
        tag = '' if k not in built else (' = build' if built[k] == v else ' != BUILD ' + built[k][:12])
        out.append(f'lib {k} {v}{tag}')
    b = r.get('build') or {}
    if b:
        out.append(f"build llama.cpp {str(b.get('llama_cpp'))[:9]}; series " +
                   ' '.join(f"{p['file'][:4]}:{p['sha256'][:8]}" for p in b.get('series', [])) + '; app ' +
                   ' '.join(f"{p['file'][:4]}:{p['sha256'][:8]}" for p in b.get('app_patches', [])))
    t = r.get('template') or {}
    p = r.get('prompt', '')
    same = t.get('rendered') == expected_render(p)
    out.append(f"template arch={t.get('architecture')} custom={t.get('custom_template')} rendered_tokens="
               f"{t.get('rendered_tokens')} rendered==OLMoE's own: {'yes' if same else 'NO'}")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, encoding='utf-8') as f:
            out.append('prompt == ' + os.path.basename(prompt_file) + ': ' + ('yes' if f.read() == p else 'NO'))
    pt = sorted({x.get('prompt_tokens') for x in r.get('runs', [])
                 if x.get('prompt_tokens') is not None and x.get('kind', 'official32') not in ('abba', 'story200')})
    out.append(f'prompt_tokens seen in official runs: {pt}')
    if r.get('unmatched_lines'):
        out.append(f"unmatched runtime lines: {len(r['unmatched_lines'])}")
    return out


def identity(r):
    """Greedy texts of the 32-token runs: equal within a numerics class (repack, skip, head2); stock == stack."""
    out, cls = [], {}
    for x in r.get('runs', []):
        if x.get('kind') in ('abba', 'story200') or x.get('text') is None:
            continue
        e = env_of(x)
        key = (x.get('repack_loaded', x.get('repack')), e.get('VOLLIE_SKIP_MASS') or e.get('GGML_EXPERT_SKIP') or None,
               e.get('VOLLIE_HEAD2') or e.get('GGML_HEAD2') or None)
        cls.setdefault(key, []).append(x)
    for key, rs in cls.items():
        texts = {x['text'] for x in rs}
        ids = sorted({x.get('id') for x in rs})
        out.append(f"text class repack={key[0]} skip_mass={key[1]} head2={key[2]}: {len(rs)} runs {ids}: "
                   + ('identical' if len(texts) == 1 else f'DIFFER ({len(texts)} texts)'))
    by_id = {}
    for x in r.get('runs', []):
        if x.get('kind') not in ('abba', 'story200') and x.get('text') is not None:
            by_id.setdefault(x.get('id'), set()).add(x['text'])
    if 'stock' in by_id and 'stack' in by_id:
        out.append('stock == stack: ' + ('yes' if by_id['stock'] == by_id['stack'] and len(by_id['stock']) == 1 else 'NO'))
    c1 = [k for k in ('c1_off', 'c1_ship', 'c1_ship_io', 'c1_off_prof', 'c1_ship_prof', 'c1_ship_io_prof') if k in by_id]
    if c1:
        texts = set().union(*(by_id[k] for k in c1))
        out.append(f"r19 rows {'/'.join(c1)} give one text: " + ('yes' if len(texts) == 1 else f'NO ({len(texts)} texts)'))
    if r.get('kit') == 'r20':
        texts = set().union(*by_id.values()) if by_id else set()
        out.append(f"r20: every 32-token run ({'/'.join(sorted(by_id))}) gives one text: "
                   + ('yes' if len(texts) == 1 else f'NO ({len(texts)} texts)'))
    if 'skip' in by_id and 'package' in by_id:
        out.append('skip == package (head2 changes no greedy token): ' + ('yes' if by_id['skip'] == by_id['package'] else 'no'))
    return out


def counters(rs):
    sk = [float(m.group(1)) for x in rs for l in x.get('c2') or [] for m in [SKIP_RE.search(l)] if m]
    hd = [float(m.group(2)) for x in rs for l in x.get('c2') or [] for m in [HEAD_RE.search(l)] if m]
    gr = [tuple(map(float, m.groups())) for x in rs for l in x.get('c2') or [] for m in [GRAPH_RE.search(l)] if m]
    s = []
    if sk:
        s.append(f'expert bytes skipped {fmt(st.median(sk))} %')
    if hd:
        s.append(f'head2 exact rows/col {fmt(st.median(hd), 0)}')
    if gr:
        s.append('graph ops/tiny/barriers/syncs ' + '/'.join(fmt(st.median(g[i] for g in gr), 0) for i in range(4)))
    return '; '.join(s)


def residency(rs):
    """Median page-cache residency of the model (MB) before and after the group's runs (r19), and the clock caps."""
    s = []
    for k in ('res_before', 'res_after'):
        v = [x[k] for x in rs if isinstance(x.get(k), dict)]
        if v:
            s.append(f"{k[4:]} experts {st.median(d['experts_mb'] for d in v):.0f} rest {st.median(d['rest_mb'] for d in v):.0f} MB")
    caps = sorted({x.get('cap_after') for x in rs if x.get('cap_after')})
    if caps:
        s.append('caps after ' + ','.join(c.replace('cap=', '') for c in caps))
    return ('resident ' + '; '.join(s)) if s else ''


def groups(r, min_mhz, fault_run=1000):
    fields = r.get('tok_fields') or TOK_FIELDS
    runs = r.get('runs', [])
    order, by = [], {}
    for x in runs:
        g = x.get('group', '?')
        if x.get('proto') == 'warm' and x.get('phase') == 'warmup':
            g += ' warm-up (unscored)'
        if g not in by:
            order.append(g)
            by[g] = []
        by[g].append(x)
    ref = next((x.get('text') for x in runs if x.get('kind', 'official32') != 'abba' and not x.get('env') and x.get('text')), None)
    out = []
    for g in order:
        rs = by[g]
        tps = [x['decode_tps'] for x in rs if x.get('decode_tps') is not None]
        sms = [steady_ms(x, fields) for x in rs]
        sm = [s for s, _ in sms if s is not None]
        src = {k for _, k in sms if k}
        clk = [c for c in (big_min(x.get('mhz_after')) for x in rs) if c is not None]
        flags = []
        for x in rs:
            why = []
            if x.get('error'):
                why.append('error')
            c = big_min(x.get('mhz_after'))
            if c is not None and c < min_mhz:
                why.append(f'throttled: clock {c}<{min_mhz}')
            dm = decode_majflt(x, fields)
            if dm is not None and dm > 100:
                why.append(f'fault-bound: decode majflt {dm}')
            elif dm:
                why.append(f'decode majflt {dm}')
            elif dm is None and (x.get('majflt') or 0) > fault_run:
                why.append(f"fault-bound: run majflt {x['majflt']}")
            if x.get('gate_unknown'):
                why.append('gate unknown')
            elif x.get('gate_timeout'):
                why.append(f"gate timed out (heat {fmt(x.get('heat'))})")
            if x.get('cached_prompt_tokens'):
                why.append(f"cached {x['cached_prompt_tokens']}")
            if x.get('kind', 'official32') != 'abba' and not x.get('env') and ref is not None and x.get('text') != ref:
                why.append('text differs from the first stock run')
            if why:
                flags.append(f"{x.get('id')}#{x.get('rep')} (" + ', '.join(why) + ')')
        cnt = '; '.join(c for c in (counters(rs), residency(rs)) if c)
        out.append(
            f"{g}: n {len(rs)} printed tok/s median {fmt(st.median(tps) if tps else None)} min {fmt(min(tps) if tps else None)} "
            f"max {fmt(max(tps) if tps else None)} | steady ms/token {fmt(st.median(sm) if sm else None)}"
            f"{' (' + '+'.join(sorted(src)) + ')' if src else ''} | big-core MHz after: min {min(clk) if clk else '-'} "
            f"median {st.median(clk) if clk else '-'} | majflt median {st.median([x.get('majflt', 0) for x in rs])} "
            f"minflt median {st.median([x.get('minflt', 0) for x in rs])}{' | ' + cnt if cnt else ''} | "
            f"flags: {'; '.join(flags) or 'none'}")
    return out


def blocks(run, min_mhz, fields=TOK_FIELDS):
    """Blocks (blk >= 1) of one A/B generation: {blk: dict(side, clean, graph ms, wall ms, class ms, counts)}."""
    toks = tokens(run, fields)
    c = run.get('chunk_ms') or []
    wall = {k: c[k] - c[k - 1] for k in range(1, len(c))}  # chunk k arrives after decode graph tok=k (+ sampling)
    by = {}
    for t in toks:
        if t.get('blk', 0) >= 1 and t.get('ab') in ('A', 'B'):
            by.setdefault(t['blk'], []).append(t)
    out = {}
    for b, ts in by.items():
        sides = {t['ab'] for t in ts}
        clk_ok = all((t.get('mhz4', 0) == 0 or t['mhz4'] >= min_mhz) and (t.get('mhz7', 0) == 0 or t['mhz7'] >= min_mhz)
                     for t in ts)
        w = [wall[t['tok']] for t in ts if t['tok'] in wall]
        out[b] = {
            'side': sides.pop() if len(sides) == 1 else '?',
            'clean': clk_ok and all(t.get('majflt', 0) == 0 for t in ts),
            'graph': st.mean(t['us'] for t in ts) / 1000,
            'wall': st.mean(w) if len(w) == len(ts) else None,
            **{k: st.mean(t.get(k, 0) for t in ts) / 1000 for k in CLASSES},
            **{'n_' + k: st.mean(t.get(k, 0) for t in ts) for k in COUNTS},
        }
    return out


def diff(a, b):
    if len(a) < 2 or len(b) < 2:
        return None, None
    return st.mean(b) - st.mean(a), (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5


def verdict(d, s, sigma_max):
    if d is None:
        return 'too few blocks'
    if s <= sigma_max:
        return 'resolved'
    if abs(d) >= 1.0 and abs(d) >= 2 * s:
        return 'resolved (|d| >= 1 ms, sigma above the sub-1 ms limit)'
    return f'UNRESOLVED (sigma {s:.2f} > {sigma_max})'


def ab(r, min_mhz, sigma_max):
    fields = r.get('tok_fields') or TOK_FIELDS
    by = {}
    for x in r.get('runs', []):
        if x.get('kind') == 'abba':
            by.setdefault(x.get('id'), []).append(x)
    out = []
    for rid, rs in by.items():
        bl = [b for x in rs for b in blocks(x, min_mhz, fields).values()]
        side = {s: [b for b in bl if b['side'] == s] for s in 'AB'}
        clean = {s: [b for b in side[s] if b['clean']] for s in 'AB'}
        knob = next((env_of(x).get('VOLLIE_AB') for x in rs if env_of(x).get('VOLLIE_AB')), '?')
        d, s = diff([b['graph'] for b in clean['A']], [b['graph'] for b in clean['B']])
        da, sa = diff([b['graph'] for b in side['A']], [b['graph'] for b in side['B']])  # faults included (EXPERT_IO)
        wa = [b['wall'] for b in clean['A'] if b['wall'] is not None]
        wb = [b['wall'] for b in clean['B'] if b['wall'] is not None]
        dw, sw = diff(wa, wb)
        cls = []
        for k in CLASSES:
            dk, sk = diff([b[k] for b in clean['A']], [b[k] for b in clean['B']])
            if dk is not None:
                cls.append(f'{k} {dk:+.2f}±{sk:.2f}')
        cnt = ' '.join(f"{k} {fmt(st.mean(b['n_' + k] for b in clean['A']), 0)}->{fmt(st.mean(b['n_' + k] for b in clean['B']), 0)}"
                       for k in COUNTS if clean['A'] and clean['B'])
        out.append(
            f"{rid} [{knob}] gens {len(rs)} | blocks A {len(side['A'])} (clean {len(clean['A'])}) B {len(side['B'])} "
            f"(clean {len(clean['B'])}) | graph B-A {fmt(d)} ± {fmt(s)} ms/token: {verdict(d, s, sigma_max)} | "
            f"all blocks {fmt(da)} ± {fmt(sa)} | "
            f"wall B-A {fmt(dw)} ± {fmt(sw)}: {verdict(dw, sw, sigma_max)} | classes {', '.join(cls) or '-'} | counts {cnt or '-'}")
    return out


def memv(s, k):
    m = re.search(k + r'=(\d+)MB', s or '')
    return int(m.group(1)) if m else None


def a78(s):
    v = mhz(s)
    return v[4] if len(v) > 4 else None


def res_total(d):
    return d.get('total_mb') if isinstance(d, dict) else None


def kw_passes(x):
    p = [tuple(int(v) for v in m.groups()) for l in x.get('c2') or [] for m in [KW_RE.search(l)] if m]
    return p


def r16_compare(r, r16):
    """Kit r20's r16-protocol rows next to campaign 1's r16 run at the same position."""
    rows = [x for x in r.get('runs', []) if x.get('proto') == 'r16']
    if not rows:
        return []
    old = (r16 or {}).get('runs', [])
    sec = lambda v: '-' if v is None else f'{v / 1000:.0f}'
    out = ['r16 protocol, position by position (left: campaign 1 r16, Granite; right: this report, OLMoE). '
           'tok/s | headroom at gate | gate wait s | idle before s | prompt s | A78 MHz end | RssFile / model res MB | '
           'Cached MB | majflt']
    for x in rows:
        k = x.get('pos')
        o = old[k - 1] if k and k - 1 < len(old) else {}
        rb = x.get('res_before') if isinstance(x.get('res_before'), dict) else {}
        out.append(
            f"pos {k:>2} {x.get('phase', '?'):8} {x.get('kind', '?'):10} | "
            f"r16 {fmt(o.get('decode_tps'))} | {fmt(o.get('heat'), 3)} | {sec(o.get('cool_ms'))} | {sec(o.get('idle_ms'))} | "
            f"{fmt((o.get('ttft_ms') or 0) / 1000 if o.get('ttft_ms') else None, 1)} | {a78(o.get('mhz_after')) or '-'} | "
            f"file {memv(o.get('mem_before'), 'file') or '-'} | {memv(o.get('mem_before'), 'cached') or '-'} | "
            f"{o.get('majflt', '-')}  ||  "
            f"r20 {fmt(x.get('decode_tps'))} | {fmt(x.get('heat'), 3)}{'T' if x.get('gate_timeout') else ''} | "
            f"{sec(x.get('cool_ms'))} | {sec(x.get('idle_ms'))} | "
            f"{fmt((x.get('ttft_ms') or 0) / 1000 if x.get('ttft_ms') else None, 1)} | {a78(x.get('mhz_after')) or '-'}"
            f"{' ' + x['cap_after'].replace('cap=', 'cap ') if x.get('cap_after') else ''} | "
            f"res {rb.get('experts_mb', '-')}+{rb.get('rest_mb', '-')} -> {res_total(x.get('res_after')) or '-'} | "
            f"{memv(x.get('mem_before'), 'cached') or '-'} | {x.get('majflt', '-')}")
    plain = lambda rs: [x['decode_tps'] for x in rs if x.get('phase') == 'on' and x.get('kind', 'official32') == 'official32'
                        and x.get('decode_tps') is not None]
    a, b = plain(old), plain(rows)
    out.append(f"R49's ten plain official runs (phase on): r16 median {fmt(st.median(a) if a else None)} (n {len(a)}); "
               f"r20 median {fmt(st.median(b) if b else None)} (n {len(b)}). T = gate timed out.")
    return out


def waits(r):
    """Every gated run's wait: model MB lost from the page cache between the previous run's end and this run's start."""
    runs = r.get('runs', [])
    out, drop = [], {True: [], False: []}
    for prev, x in zip(runs, runs[1:]):
        if not x.get('gate') or res_total(prev.get('res_after')) is None or res_total(x.get('res_before')) is None:
            continue
        on = bool(env_of(prev).get('GGML_KEEP_WARM'))
        d = res_total(prev['res_after']) - res_total(x['res_before'])
        idle = (x.get('idle_ms') or 0) / 1000
        p = kw_passes(x)
        drop[on].append(d)
        if on or p:
            out.append(f"wait before pos {x.get('pos')} ({x.get('proto')}): idle {idle:.0f} s, keep-warm "
                       f"{'on' if on else 'off'}, lost {d} MB; passes {len(p)}, CPU {sum(q[4] for q in p) / 1000:.0f} ms, "
                       f"wall max {max((q[3] for q in p), default=0) / 1000:.0f} ms, aborted {sum(q[5] for q in p)}, "
                       f"resident at last pass {p[-1][1] if p else '-'} MB")
    for on in (False, True):
        if drop[on]:
            out.append(f"waits with keep-warm {'on' if on else 'off'}: n {len(drop[on])}, model MB lost per wait median "
                       f"{st.median(drop[on]):.0f} min {min(drop[on])} max {max(drop[on])}")
    return out


def warm(r):
    """The warm block (no gate): per run tok/s, headroom and A78 clock before -> after, residency."""
    rows = [x for x in r.get('runs', []) if x.get('proto') == 'warm']
    out = ['warm block (different protocol: no gate, back to back): pos id tok/s | headroom before -> after | '
           'A78 MHz before -> after | cap after | model res before -> after MB | idle before ms'] if rows else []
    for x in rows:
        out.append(f"pos {x.get('pos')} {x.get('id')}{' (warm-up)' if x.get('phase') == 'warmup' else ''} "
                   f"{fmt(x.get('decode_tps'))} | {fmt(x.get('heat'), 3)} -> {fmt(x.get('heat_after'), 3)} | "
                   f"{a78(x.get('mhz_before')) or '-'} -> {a78(x.get('mhz_after')) or '-'} | "
                   f"{(x.get('cap_after') or '-').replace('cap=', '')} | {res_total(x.get('res_before')) or '-'} -> "
                   f"{res_total(x.get('res_after')) or '-'} | {x.get('idle_ms', '-')}")
    return out


# kit r21 (memo-19): patches/llama.cpp-c1-olmoe/0002's lines
IO_RE = re.compile(r'c2-io run=(\S+) prompt_graphs=(\d+) decode_graphs=(\d+) us=(-?\d+)/(-?\d+)/(-?\d+) '
                   r'read_bytes=(-?\d+)/(-?\d+)/(-?\d+) rchar=(-?\d+)/(-?\d+)/(-?\d+) majflt=(-?\d+)/(-?\d+)/(-?\d+) '
                   r'minflt=(-?\d+)/(-?\d+)/(-?\d+)')
PCOLD_RE = re.compile(r'c2-pcold tau=(\d+) advice=(\w+) tokens=(\d+) layers=(\d+) routed_slices=(\d+) marked_slices=(\d+) '
                      r'mb=([\d.]+) fails=(\d+) us=(\d+)')
ADV_RE = re.compile(r'c2-pcold-advice advice=(\w+) errno_cold=(\d+) errno_pageout=(\d+) page=(\d+)')


def io_of(x):
    """The row's c2-io line as prefill / decode parts: read MB, rchar MB, major faults, ms; None when absent."""
    for l in x.get('c2') or []:
        m = IO_RE.search(l)
        if m and m.group(1) == x.get('run_tag', m.group(1)):
            v = [int(g) for g in m.groups()[1:]]
            us, rb, rc, mj = v[2:5], v[5:8], v[8:11], v[11:14]
            part = lambda a, i: None if a[i] < 0 or a[i + 1] < 0 else a[i + 1] - a[i]
            mb = lambda a, i: None if part(a, i) is None else part(a, i) / 1e6
            return {'prompt_graphs': v[0], 'decode_graphs': v[1],
                    'pre_read_mb': mb(rb, 0), 'dec_read_mb': mb(rb, 1), 'pre_rchar_mb': mb(rc, 0), 'dec_rchar_mb': mb(rc, 1),
                    'pre_majflt': part(mj, 0), 'dec_majflt': part(mj, 1),
                    'pre_ms': None if part(us, 0) is None else part(us, 0) / 1000,
                    'dec_ms': None if part(us, 1) is None else part(us, 1) / 1000}
    return None


def pcold_of(x):
    m = [PCOLD_RE.search(l) for l in x.get('c2') or []]
    m = [q for q in m if q and int(q.group(3)) > 1]
    return {'advice': m[-1].group(2), 'mb': sum(float(q.group(7)) for q in m), 'slices': sum(int(q.group(6)) for q in m),
            'fails': sum(int(q.group(8)) for q in m), 'us': sum(int(q.group(9)) for q in m)} if m else None


def med(v):
    v = [a for a in v if a is not None]
    return st.median(v) if v else None


def spread(v):
    v = [a for a in v if a is not None]
    return f"{fmt(st.median(v))} [{fmt(min(v))}..{fmt(max(v))}] n {len(v)}" if v else '-'


def r21_models(r):
    order = list((r.get('models') or {}).keys())
    for x in r.get('runs', []):
        if x.get('model') not in order:
            order.append(x.get('model'))
    return order


def r21_checks(r):
    out = []
    built = (r.get('build') or {}).get('libs') or {}
    for k, m in (r.get('models') or {}).items():
        ok = m.get('bytes') == m.get('expected_bytes') and m.get('verified_marker') == m.get('expected_sha256') == m.get('sha256_end')
        out.append(f"model {k}: {m.get('file')} {m.get('bytes')} B arch {m.get('arch')} template {m.get('template_source')} "
                   f"repack rule {m.get('repack_rule')}; marker {str(m.get('verified_marker'))[:8]}, sha256 at end "
                   f"{str(m.get('sha256_end'))[:8]} -> {'OK' if ok else 'CHECK'}")
    libs = {}
    for x in r.get('runs', []):
        for k, v in (x.get('libs') or {}).items():
            libs.setdefault(k, set()).add(v)
    for k, v in sorted(libs.items()):
        out.append(f"lib {k}: {len(v)} hash(es) at the loads" + ('' if k not in built else
                   (' = build' if v == {built[k]} else ' != BUILD ' + built[k][:12])))
    b = r.get('build') or {}
    if b:
        out.append(f"build llama.cpp {str(b.get('llama_cpp'))[:9]}; series " +
                   ' '.join(f"{p['file'][:4]}:{p['sha256'][:8]}" for p in b.get('series', [])) + '; app ' +
                   ' '.join(f"{p['file'][:4]}:{p['sha256'][:8]}" for p in b.get('app_patches', [])))
    for k, t in (r.get('templates') or {}).items():
        out.append(f"template {k}: arch {t.get('architecture')} custom {t.get('custom_template')} rendered_tokens "
                   f"{t.get('rendered_tokens')}")
    for k in r21_models(r):
        rs = [x for x in r.get('runs', []) if x.get('model') == k]
        texts = {x.get('text') for x in rs if x.get('text') is not None}
        bad_env = [x.get('run_tag') for x in rs if {'GGML_EXPERT_SKIP', 'GGML_HEAD2'} & set(env_of(x))]
        nr = [x.get('run_tag') for x in rs if x.get('repack_loaded') is not True]
        noio = [x.get('run_tag') for x in rs if io_of(x) is None]
        err = [x.get('run_tag') for x in rs if x.get('error')]
        pt = sorted({x.get('prompt_tokens') for x in rs if x.get('prompt_tokens') is not None})
        out.append(f"{k}: {len(rs)} runs, one text across both arms: {'yes' if len(texts) == 1 else f'NO ({len(texts)} texts)'}; "
                   f"repack on in all: {'yes' if not nr else 'NO ' + ','.join(nr)}; skip/head2 set: {','.join(bad_env) or 'none'}; "
                   f"prompt_tokens {pt}; rows without a c2-io line: {','.join(noio) or 'none'}; errors: {','.join(err) or 'none'}")
    adv = {m.groups() for x in r.get('runs', []) for l in x.get('c2') or [] for m in [ADV_RE.search(l)] if m}
    out.append('advice probe: ' + ('; '.join(f"advice={a} errno_cold={b} errno_pageout={c} page={d}" for a, b, c, d in sorted(adv))
                                   if adv else 'no c2-pcold-advice line'))
    if r.get('unmatched_lines'):
        out.append(f"unmatched runtime lines: {len(r['unmatched_lines'])}")
    return out


def r21_stats(rs):
    """One model and arm, scored runs: the medians the memo-19 tables use."""
    io = [io_of(x) or {} for x in rs]
    pc = [pcold_of(x) or {} for x in rs]
    rb = lambda k, f: med(x[k][f] for x in rs if isinstance(x.get(k), dict))
    return {
        'n': len(rs), 'tps': med(x.get('decode_tps') for x in rs),
        'tps_min': min((x['decode_tps'] for x in rs if x.get('decode_tps') is not None), default=None),
        'tps_max': max((x['decode_tps'] for x in rs if x.get('decode_tps') is not None), default=None),
        'dec_read_mb': med(i.get('dec_read_mb') for i in io), 'dec_majflt': med(i.get('dec_majflt') for i in io),
        'pre_read_mb': med(i.get('pre_read_mb') for i in io), 'pre_ms': med(i.get('pre_ms') for i in io),
        'ttft_ms': med(x.get('ttft_ms') for x in rs),
        'res_before_exp': rb('res_before', 'experts_mb'), 'res_before_rest': rb('res_before', 'rest_mb'),
        'res_after_exp': rb('res_after', 'experts_mb'), 'res_after_rest': rb('res_after', 'rest_mb'),
        'marked_mb': med(p.get('mb') for p in pc), 'advice': ','.join(sorted({p['advice'] for p in pc if p.get('advice')})) or '-',
        'cached_mb': med(memv(x.get('mem_before'), 'cached') for x in rs),
        'gate_timeouts': sum(1 for x in rs if x.get('gate_timeout')),
    }


def r21_pairs(rs):
    """ABBA pairs of one block's scored runs in list order (off on on off ...): (off, on) per pair."""
    out = []
    for i in range(0, len(rs) - 3, 4):
        q = rs[i:i + 4]
        if [x.get('arm') for x in q] == ['off', 'on', 'on', 'off']:
            out += [(q[0], q[1]), (q[3], q[2])]
    return out


def r21_tables(r):
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r21']
    if not runs:
        return []
    models = r21_models(r)
    scored = lambda m, a: [x for x in runs if x.get('model') == m and x.get('arm') == a and x.get('phase') == 'on']
    S = {(m, a): r21_stats(scored(m, a)) for m in models for a in ('off', 'on')}
    out = ['kit r21, scored runs (phase on), per model and arm (*device*):']
    for m in models:
        for a in ('off', 'on'):
            s = S[(m, a)]
            out.append(f"  {m} {a}: n {s['n']} tok/s median {fmt(s['tps'])} [{fmt(s['tps_min'])}..{fmt(s['tps_max'])}] | "
                       f"decode read MB {fmt(s['dec_read_mb'], 0)} majflt {fmt(s['dec_majflt'], 0)} | prefill read MB "
                       f"{fmt(s['pre_read_mb'], 0)} ms {fmt(s['pre_ms'], 0)} ttft ms {fmt(s['ttft_ms'], 0)} | res before "
                       f"{fmt(s['res_before_exp'], 0)}+{fmt(s['res_before_rest'], 0)} after {fmt(s['res_after_exp'], 0)}+"
                       f"{fmt(s['res_after_rest'], 0)} MB | marked MB {fmt(s['marked_mb'], 0)} advice {s['advice']} | "
                       f"Cached MB {fmt(s['cached_mb'], 0)} | gate timeouts {s['gate_timeouts']}")
    out.append('ABBA pairs per block, on minus off: tok/s and decode read MB (median [min..max] n):')
    allp = {m: [] for m in models}
    for g in sorted({x.get('group') for x in runs}, key=lambda g: int(str(g).lstrip('b') or 0)):
        rs = [x for x in runs if x.get('group') == g and x.get('phase') == 'on']
        if not rs:
            continue
        p = r21_pairs(rs)
        allp[rs[0].get('model')] += p
        dt = [None if b.get('decode_tps') is None or a.get('decode_tps') is None else b['decode_tps'] - a['decode_tps'] for a, b in p]
        dr = [None if (io_of(b) or {}).get('dec_read_mb') is None or (io_of(a) or {}).get('dec_read_mb') is None
              else io_of(b)['dec_read_mb'] - io_of(a)['dec_read_mb'] for a, b in p]
        out.append(f"  {g} {rs[0].get('model')}: tok/s {spread(dt)} | decode read MB {spread(dr)}")
    out.append('per model, every pair: on minus off')
    for m in models:
        p = allp[m]
        dt = [None if b.get('decode_tps') is None or a.get('decode_tps') is None else b['decode_tps'] - a['decode_tps'] for a, b in p]
        rel = [None if not (io_of(a) or {}).get('dec_read_mb') or (io_of(b) or {}).get('dec_read_mb') is None
               else 100 * (io_of(b)['dec_read_mb'] / io_of(a)['dec_read_mb'] - 1) for a, b in p]
        out.append(f"  {m}: tok/s {spread(dt)} | decode read_bytes change % {spread(rel)}")
    # side by side: one column per model, in the table's order (N models)
    rows = [('tok/s off', 'tps', 'off', 2), ('tok/s on', 'tps', 'on', 2), ('decode read MB off', 'dec_read_mb', 'off', 0),
            ('decode read MB on', 'dec_read_mb', 'on', 0), ('decode majflt off', 'dec_majflt', 'off', 0),
            ('decode majflt on', 'dec_majflt', 'on', 0), ('prefill ms off', 'pre_ms', 'off', 0), ('prefill ms on', 'pre_ms', 'on', 0),
            ('prefill read MB off', 'pre_read_mb', 'off', 0), ('prefill read MB on', 'pre_read_mb', 'on', 0),
            ('res before experts off', 'res_before_exp', 'off', 0), ('res before experts on', 'res_before_exp', 'on', 0),
            ('res after experts off', 'res_after_exp', 'off', 0), ('res after experts on', 'res_after_exp', 'on', 0),
            ('marked MB on', 'marked_mb', 'on', 0)]
    out.append('side by side (medians, *device*): | metric | ' + ' | '.join(models) + ' |')
    for label, k, a, nd in rows:
        out.append(f"| {label} | " + ' | '.join(fmt(S[(m, a)][k], nd) for m in models) + ' |')
    return out


# kit r22 (memo-20): patches/llama.cpp-c1-olmoe/0005's per-prompt lines and the app's hold
BLK_RE = re.compile(r'c2-io-blk run=(\S+) inblock=(-?\d+)/(-?\d+)/(-?\d+)')
SKIPRUN_RE = re.compile(r'c2-skip-run: run=(\S+) .*?decode_tokens=(\d+) decode_pairs_skipped=([\d.]+)% '
                        r'decode_expert_bytes_skipped=([\d.]+)% decode_MB_per_token_selected=([\d.]+) '
                        r'decode_MB_per_token_skipped=([\d.]+) decode_bytes_unread=([\d.]+)%')
HEADRUN_RE = re.compile(r'c2-head-run: run=(\S+) c2_k=(\d+) c2_calls=(\d+) c2_cols=(\d+) c2_exact_rows_per_col=([\d.]+) '
                        r'c2_copies=(\d+) c2_copy=(\d) c1_m=(\d+) c1_calls=(\d+) c1_copies=(\d+) c1_copy=(\d)')


def tagged(x, rx):
    for l in x.get('c2') or []:
        m = rx.search(l)
        if m and m.group(1) == x.get('run_tag', m.group(1)):
            return m
    return None


def blk_of(x):
    """getrusage inblock over the prompt as MB: prefill, decode; None when absent or unreadable (-1)."""
    m = tagged(x, BLK_RE)
    if not m:
        return None
    a, b, c = (int(g) for g in m.groups()[1:])
    mb = lambda u, v: None if u < 0 or v < 0 else (v - u) * 512 / 1e6
    return {'pre_mb': mb(a, b), 'dec_mb': mb(b, c)}


def skiprun_of(x):
    m = tagged(x, SKIPRUN_RE)
    return None if not m else {'dec_tokens': int(m.group(2)), 'dec_pairs_pct': float(m.group(3)),
                               'dec_bytes_pct': float(m.group(4)), 'dec_mb_sel': float(m.group(5)),
                               'dec_mb_skip': float(m.group(6)), 'dec_unread_pct': float(m.group(7))}


DECL_RE = re.compile(r' c2_declined=(\d+) c2_why=(\S+)')  # c1-olmoe 0006 (kit r23): appended to c2-head-run


def headrun_of(x):
    m = tagged(x, HEADRUN_RE)
    if not m:
        return None
    v = m.groups()
    d = DECL_RE.search(m.string)
    return {'c2_k': int(v[1]), 'c2_calls': int(v[2]), 'c2_copies': int(v[5]), 'c2_copy': v[6] == '1', 'c1_m': int(v[7]),
            'c1_calls': int(v[8]), 'c1_copies': int(v[9]), 'c1_copy': v[10] == '1',
            'c2_declined': int(d.group(1)) if d else None, 'c2_why': d.group(2) if d else None}


def md5(t):
    import hashlib
    return hashlib.md5(t.encode()).hexdigest()


def mean_se(v):
    v = [a for a in v if a is not None]
    if not v:
        return None, None, 0
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else None), len(v)


def fmt_se(v, nd=2):
    m, se, n = mean_se(v)
    return '-' if m is None else f"{m:+.{nd}f} ± {'-' if se is None else f'{se:.{nd}f}'} (n {n})"


def welch(a, b, nd=2):
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if not a or not b:
        return '-'
    d = st.mean(a) - st.mean(b)
    se = (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5 if len(a) > 1 and len(b) > 1 else None
    return f"{d:+.{nd}f} ± {'-' if se is None else f'{se:.{nd}f}'} (n {len(a)} / {len(b)})"


def res_mb(x, k='res_before'):
    d = x.get(k)
    return d.get('total_mb') if isinstance(d, dict) else None


def r22_checks(r):
    out = [l for l in r21_checks(r) if not re.match(r'\S+: \d+ runs, one text', l) and not l.startswith('advice probe')]
    models = r.get('models') or {}
    for k in r21_models(r):
        rs = [x for x in r.get('runs', []) if x.get('model') == k]
        num = (models.get(k) or {}).get('num') or {}
        parts = []
        texts = {}
        for a in ('off', 'num'):
            t = {md5(x['text'])[:8] for x in rs if x.get('arm') == a and x.get('text') is not None}
            texts[a] = t
            parts.append(f"{a}: {'one text ' + next(iter(t)) if len(t) == 1 else f'NO ({len(t)} texts)'}")
        parts.append('off != num' if texts['off'] and texts['num'] and texts['off'] != texts['num'] else 'off == num (CHECK)')
        bad_off = [x.get('run_tag') for x in rs if x.get('arm') == 'off' and set(num) & set(env_of(x))]
        bad_num = [x.get('run_tag') for x in rs if x.get('arm') == 'num' and any(env_of(x).get(e) != v for e, v in num.items())]
        pc = [x.get('run_tag') for x in rs if {'GGML_PREFILL_COLD', 'GGML_KEEP_WARM'} & set(env_of(x))]
        nr = [x.get('run_tag') for x in rs if x.get('repack_loaded') is not True]
        noio = [x.get('run_tag') for x in rs if io_of(x) is None]
        nskip = [x.get('run_tag') for x in rs if x.get('arm') == 'num' and skiprun_of(x) is None]
        nhead = [x.get('run_tag') for x in rs if x.get('arm') == 'num' and not ((headrun_of(x) or {}).get('c2_copy')
                                                                                  or (headrun_of(x) or {}).get('c1_copy'))]
        err = [x.get('run_tag') for x in rs if x.get('error')]
        out.append(f"{k}: {len(rs)} runs; texts {'; '.join(parts)}; num env {num}: missing in {','.join(bad_num) or 'none'}, "
                   f"in off arm {','.join(bad_off) or 'none'}; PREFILL_COLD/KEEP_WARM set: {','.join(pc) or 'none'}; repack on in "
                   f"all: {'yes' if not nr else 'NO ' + ','.join(nr)}; no c2-io: {','.join(noio) or 'none'}; num without "
                   f"c2-skip-run: {','.join(nskip) or 'none'}; num without an active head copy: {','.join(nhead) or 'none'}; "
                   f"errors: {','.join(err) or 'none'}")
    runs = r.get('runs', [])
    ib = [x.get('inblock_read_bytes') for x in runs if x.get('inblock_read_bytes') is not None]
    io = [x.get('io_read_bytes') for x in runs if x.get('io_read_bytes') is not None]
    out.append(f"I/O source: getrusage inblock in {len(ib)} rows (non-zero in {sum(1 for v in ib if v)}), /proc/self/io in "
               f"{len(io)} rows; memlock rlimit {r.get('memlock_rlimit')}")
    return out


def r22_stats(rs):
    s = r21_stats(rs)
    sk = [skiprun_of(x) or {} for x in rs]
    bl = [blk_of(x) or {} for x in rs]
    hs = [x.get('hold_stats') or {} for x in rs]
    s.update({
        'dec_inblock_mb': med(b.get('dec_mb') for b in bl), 'pre_inblock_mb': med(b.get('pre_mb') for b in bl),
        'skip_bytes_pct': med(k.get('dec_bytes_pct') for k in sk), 'skip_mb_tok': med(k.get('dec_mb_skip') for k in sk),
        'res_before': med(res_mb(x) for x in rs), 'wait_lost_mb': med(x.get('wait_lost_mb') for x in rs),
        'cool_s': med(None if x.get('cool_ms') is None else x['cool_ms'] / 1000 for x in rs),
        'hold_mb': med(h.get('advised_mb') for h in hs), 'hold_cpu_ms': med(None if h.get('cpu_us') is None else h['cpu_us'] / 1000 for h in hs),
        'hold_final_ms': med(h.get('final_ms') for h in hs), 'hold_inblock_mb': med(None if h.get('inblock') is None else h['inblock'] * 512 / 1e6 for h in hs),
    })
    return s


def r22_quads(rs):
    """One block's scored runs in list order: the ABBA quads (off num num off, one hold value each)."""
    out = []
    for i in range(0, len(rs) - 3, 4):
        q = rs[i:i + 4]
        if [x.get('arm') for x in q] == ['off', 'num', 'num', 'off'] and len({bool(x.get('hold')) for x in q}) == 1:
            out.append(q)
    return out


def quad_effect(q):
    v = [x.get('decode_tps') for x in q]
    return None if None in v else (v[1] + v[2]) / 2 - (v[0] + v[3]) / 2


def r22_tables(r, wait_s=30):
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r22']
    if not runs:
        return []
    models = r21_models(r)
    sc = lambda m, a=None, h=None: [x for x in runs if x.get('model') == m and x.get('phase') == 'on'
                                    and (a is None or x.get('arm') == a) and (h is None or bool(x.get('hold')) == h)]
    out = ['kit r22, scored runs (phase on), per model, arm and hold (*device*):']
    for m in models:
        for a in ('off', 'num'):
            for h in (False, True):
                s = r22_stats(sc(m, a, h))
                out.append(f"  {m} {a} hold={'on ' if h else 'off'}: n {s['n']} tok/s {fmt(s['tps'])} [{fmt(s['tps_min'])}.."
                           f"{fmt(s['tps_max'])}] | res before MB {fmt(s['res_before'], 0)} lost in wait {fmt(s['wait_lost_mb'], 0)} "
                           f"gate s {fmt(s['cool_s'], 0)} | decode majflt {fmt(s['dec_majflt'], 0)} inblock MB {fmt(s['dec_inblock_mb'], 0)} | "
                           f"prefill ms {fmt(s['pre_ms'], 0)} ttft ms {fmt(s['ttft_ms'], 0)} | skip bytes % {fmt(s['skip_bytes_pct'])} "
                           f"MB/tok {fmt(s['skip_mb_tok'], 1)} | hold read-back MB {fmt(s['hold_mb'], 0)} cpu ms "
                           f"{fmt(s['hold_cpu_ms'], 0)} after-gate ms {fmt(s['hold_final_ms'], 0)} | Cached MB {fmt(s['cached_mb'], 0)}")
    out.append('num minus off, mean over ABBA quads of mean(num) - mean(off), tok/s ± SE:')
    for m in models:
        qs = []
        for g in sorted({x.get('group') for x in runs if x.get('model') == m}, key=lambda g: int(str(g).lstrip('b') or 0)):
            qs += r22_quads([x for x in runs if x.get('group') == g and x.get('phase') == 'on'])
        e = lambda h=None: [quad_effect(q) for q in qs if h is None or bool(q[0].get('hold')) == h]
        out.append(f"  {m}: all {fmt_se(e())} | hold on {fmt_se(e(True))} | hold off {fmt_se(e(False))}")
    out.append(f"hold minus no hold, off arm (tok/s; MB resident before the run), Welch SE; 'waited' = gate wait >= {wait_s} s:")
    for m in models:
        on, off = sc(m, 'off', True), sc(m, 'off', False)
        w = lambda v: [x for x in v if (x.get('cool_ms') or 0) >= wait_s * 1000]
        tps = lambda v: [x.get('decode_tps') for x in v]
        rb = lambda v: [res_mb(x) for x in v]
        out.append(f"  {m}: all tok/s {welch(tps(on), tps(off))}, res MB {welch(rb(on), rb(off), 0)} | waited tok/s "
                   f"{welch(tps(w(on)), tps(w(off)))}, res MB {welch(rb(w(on)), rb(w(off)), 0)}")
        onn, offn = sc(m, 'num', True), sc(m, 'num', False)
        out.append(f"  {m} (num arm, for reference): all tok/s {welch(tps(onn), tps(offn))}, res MB {welch(rb(onn), rb(offn), 0)}")
    hs = [x.get('hold_stats') for x in runs if x.get('hold') and isinstance(x.get('hold_stats'), dict)]
    if hs:
        g = lambda k, f=1: spread([None if h.get(k) is None else h[k] * f for h in hs])
        out.append(f"hold cost per gated run (median [min..max] n): read back MB {g('advised_mb')} | CPU ms {g('cpu_us', 1e-3)} | "
                   f"passes {g('passes')} | ms after the gate {g('final_ms')} | blocks read MB {g('inblock', 512 / 1e6)} | "
                   f"fadvise fails {g('advise_fails')}")
    return out


# kit r23 (memo-22): factor threads per quad, hold in every gated run, OLMoE's load env, 0006's head decline line
def r23_checks(r):
    out = r22_checks(r)
    runs = r.get('runs', [])
    models = r.get('models') or {}
    for k in r21_models(r):
        rs = [x for x in runs if x.get('model') == k]
        load = (models.get(k) or {}).get('load') or {}
        loads = [x for x in rs if x.get('reload_ms') is not None]
        bad_load = [x.get('run_tag') for x in loads if any((x.get('load_env_set') or {}).get(e) != v for e, v in load.items())]
        other = {e for m, d in models.items() if m != k for e in ((d or {}).get('load') or {})} - set(load)
        leak_load = [x.get('run_tag') for x in loads if other & set(x.get('load_env_set') or {})]
        in_run = [x.get('run_tag') for x in rs if set(load) & set(env_of(x))]
        texts = {}
        for x in rs:
            if x.get('text') is not None:
                texts.setdefault((x.get('arm'), x.get('threads')), set()).add(md5(x['text'])[:8])
        tt = '; '.join(f"{a} t{t}: {next(iter(v)) if len(v) == 1 else f'NO ({len(v)})'}" for (a, t), v in sorted(texts.items(), key=str))
        same = all(len({next(iter(v)) for (a2, _), v in texts.items() if a2 == a and len(v) == 1}) <= 1 for a in ('off', 'num'))
        hr = {x.get('run_tag'): headrun_of(x) for x in rs}
        decl = [f"{t}({h['c2_declined']} {h['c2_why']})" for t, h in hr.items() if h and h.get('c2_declined')]
        nohead = [x.get('run_tag') for x in rs if x.get('arm') == 'num' and not ((hr[x.get('run_tag')] or {}).get('c2_copy')
                                                                                 or (hr[x.get('run_tag')] or {}).get('c1_copy'))]
        nohold = [x.get('run_tag') for x in rs if x.get('gate') and not (x.get('hold') and isinstance(x.get('hold_stats'), dict))]
        thr = [x.get('run_tag') for x in rs if x.get('threads_loaded') not in (None, [x.get('threads'), x.get('threads')])]
        out.append(f"{k} (r23): load env {load or '{}'} at {len(loads)} load(s): missing in {','.join(bad_load) or 'none'}, other "
                   f"model's load env in {','.join(leak_load) or 'none'}, in a run env {','.join(in_run) or 'none'}; texts per arm "
                   f"and threads: {tt}; same text at 3 and 4 threads: {'yes' if same else 'NO'}; head declines: "
                   f"{','.join(decl) or 'none'}; num without an active head copy: {','.join(nohead) or 'none'}; gated without "
                   f"hold: {','.join(nohold) or 'none'}; threads applied != spec: {','.join(thr) or 'none'}")
    out.append(f"kernel release {r.get('kernel_release')!r}; owner note {r.get('owner_note')!r}")
    return out


def r23_blocks(runs, m):
    return sorted({x.get('group') for x in runs if x.get('model') == m}, key=lambda g: int(str(g).lstrip('b') or 0))


def r23_quads(rs):
    """One block's scored runs in list order: the ABBA quads (off num num off, one thread count each)."""
    return [q for q in r22_quads(rs) if len({x.get('threads') for x in q}) == 1]


def hold_lines(runs):
    out = []
    out.append('hold per gated run: pos tag | gate s | want MB, lost in wait MB | passes wait+final | CPU ms (mincore ms) | '
               'after-gate ms | back % | advised MB sum / distinct, of earlier advice resident at the last pass | stop | res before MB')
    for x in [x for x in runs if x.get('gate')]:
        h = x.get('hold_stats') or {}
        out.append(f"  {x.get('pos')} {x.get('run_tag')} | {fmt((x.get('cool_ms') or 0) / 1000, 0)} | {x.get('want_mb')}, "
                   f"{x.get('wait_lost_mb')} | {h.get('wait_passes', 0)}+{h.get('final_passes', 0)} | "
                   f"{fmt(None if h.get('cpu_us') is None else h['cpu_us'] / 1000, 0)} ({h.get('mincore_ms')}) | {h.get('final_ms')} | "
                   f"{fmt(None if h.get('back_permille') is None else h['back_permille'] / 10, 1)} | {h.get('advised_mb')} / "
                   f"{h.get('advised_distinct_mb')}, {h.get('adv_back_mb')} of {h.get('adv_before_mb')} | {h.get('stop')} | {res_mb(x)}")
    hs = [x.get('hold_stats') for x in runs if x.get('gate') and isinstance(x.get('hold_stats'), dict) and 'passes' in x['hold_stats']]
    if hs:
        g = lambda k, f=1: spread([None if h.get(k) is None else h[k] * f for h in hs])
        ok = sum(1 for h in hs if (h.get('back_permille') or 0) >= 950)
        out.append(f"hold cost per gated run (median [min..max] n): CPU ms {g('cpu_us', 1e-3)} | mincore ms {g('mincore_ms')} | "
                   f"passes {g('passes')} | ms after the gate {g('final_ms')} | back % {g('back_permille', 0.1)} (>= 95 % in {ok} "
                   f"of {len(hs)}) | advised MB distinct {g('advised_distinct_mb')} | blocks read MB {g('inblock', 512 / 1e6)} | "
                   f"fadvise fails {g('advise_fails')}")
    return out


def r23_tables(r):
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r23']
    if not runs:
        return []
    models = r21_models(r)
    sc = lambda m, a=None, t=None: [x for x in runs if x.get('model') == m and x.get('phase') == 'on'
                                    and (a is None or x.get('arm') == a) and (t is None or x.get('threads') == t)]
    ts = sorted({x.get('threads') for x in runs if x.get('phase') == 'on'})
    out = ['kit r23, scored runs (phase on), per model, arm and threads (*device*):']
    for m in models:
        for a in ('off', 'num'):
            for t in ts:
                v = sc(m, a, t)
                s = r22_stats(v)
                bk = med(None if (x.get('hold_stats') or {}).get('back_permille') is None else x['hold_stats']['back_permille'] / 10
                         for x in v)
                out.append(f"  {m} {a} t{t}: n {s['n']} tok/s {fmt(s['tps'])} [{fmt(s['tps_min'])}..{fmt(s['tps_max'])}] | res "
                           f"before MB {fmt(s['res_before'], 0)} lost in wait {fmt(s['wait_lost_mb'], 0)} back % {fmt(bk, 1)} gate s "
                           f"{fmt(s['cool_s'], 0)} | decode majflt {fmt(s['dec_majflt'], 0)} inblock MB {fmt(s['dec_inblock_mb'], 0)} | "
                           f"prefill ms {fmt(s['pre_ms'], 0)} ttft ms {fmt(s['ttft_ms'], 0)} | skip bytes % {fmt(s['skip_bytes_pct'])} | "
                           f"hold cpu ms {fmt(s['hold_cpu_ms'], 0)} after-gate ms {fmt(s['hold_final_ms'], 0)} | Cached MB "
                           f"{fmt(s['cached_mb'], 0)}")
    out.append('num minus off, mean over ABBA quads of mean(num) - mean(off), tok/s ± SE:')
    for m in models:
        qs = [q for g in r23_blocks(runs, m) for q in r23_quads([x for x in runs if x.get('group') == g and x.get('phase') == 'on'])]
        e = lambda t=None: [quad_effect(q) for q in qs if t is None or q[0].get('threads') == t]
        out.append(f"  {m}: all {fmt_se(e())} | " + ' | '.join(f"t{t} {fmt_se(e(t))}" for t in ts))
    if len(ts) == 2:
        lo, hi = ts
        out.append(f'threads {hi} minus {lo}, tok/s: per block mean(arm at {hi}) - mean(arm at {lo}), mean over blocks ± SE; '
                   'all runs, Welch:')
        for m in models:
            parts = []
            for a in ('off', 'num', None):
                d = []
                for g in r23_blocks(runs, m):
                    b = [x for x in sc(m, a) if x.get('group') == g]
                    u, v = [x.get('decode_tps') for x in b if x.get('threads') == hi], [x.get('decode_tps') for x in b if x.get('threads') == lo]
                    if u and v and None not in u + v:
                        d.append(st.mean(u) - st.mean(v))
                tps = lambda t: [x.get('decode_tps') for x in sc(m, a, t)]
                parts.append(f"{a or 'both arms'} blocks {fmt_se(d)}, runs {welch(tps(hi), tps(lo))}")
            out.append(f"  {m}: " + ' | '.join(parts))
    out.append('where the DartWorker threads ran (thread_samples, scored runs): running per sample median; share of running '
               'samples on cpu 0-3 / 4-6 / 7:')
    for m in models:
        for t in ts:
            smp = [s for x in sc(m, None, t) for s in (x.get('thread_samples') or [])]
            cpus = [c for s in smp for c in s[2:]]
            sh = lambda f: '-' if not cpus else f"{100 * sum(1 for c in cpus if f(c)) / len(cpus):.0f}%"
            out.append(f"  {m} t{t}: samples {len(smp)}, running {fmt(med(s[1] for s in smp), 1)}, cpu 0-3 {sh(lambda c: c < 4)} "
                       f"4-6 {sh(lambda c: 4 <= c <= 6)} 7 {sh(lambda c: c == 7)}; pinned {sorted({str(x.get('pinned_cpus')) for x in sc(m, None, t)})}")
    return out + hold_lines(runs)


# kit r24 (memo-23): factor gate rule per quad, charger guard, vmstat counters, idle baseline, thermal zones
def cap_of(s):
    v = [int(x) for x in re.findall(r'\d+', (re.match(r'cap=([\d/]+)', s or '') or [None, ''])[1])]
    return min(v) if v else None


def vm_delta(x, a, b, k):
    u, v = x.get(a), x.get(b)
    return v[k] - u[k] if isinstance(u, dict) and isinstance(v, dict) and k in u and k in v else None


PLATEAU = {'from_ms': 30000, 'window': 2, 'eps': 0.01, 'fall': 0.05}  # kit r24's Research.plateau* (memo-23 §2)


def plateau_end(series, below=0.65, wait_ms=90000, p=PLATEAU):
    """Kit r24's plateau gate replayed on a heat series ([[ms, headroom or None], ...] at the gate polls): (ms, why, h)."""
    hs, k = [], p['window']
    for ms, h in series:
        if h is not None:
            hs.append(h)
            if h < below:
                return ms, 'gate', h
            if ms >= p['from_ms'] and len(hs) >= 2 * k and min(hs[-k:]) >= min(hs[-2 * k:-k]) - p['eps'] \
                    and max(hs) - min(hs[-k:]) >= p['fall']:
                return ms, 'plateau', h
        if ms >= wait_ms:
            return ms, 'timeout', h
    return (series[-1][0], 'timeout', series[-1][1]) if series else (None, 'none', None)


def plateau_replay(r):
    """Every gated run with a heat series (kits r22 to r24): the r16 wait as run, and what the plateau gate would have done."""
    out = ['plateau gate replayed on the heat series (pos tag | wait s as run, end | plateau: wait s, end, headroom | '
           'lowest headroom later in the series):']
    rows = []
    for x in r.get('runs', []):
        sr = x.get('heat_series')
        if not (x.get('gate') and sr):
            continue
        e, why, h = plateau_end(sr)
        later = [v for ms, v in sr if v is not None and ms > e]
        rows.append((e, why, h, min(later) if later else None, sr[-1][1]))
        out.append(f"  {x.get('pos')} {x.get('run_tag')} | {(x.get('cool_ms') or 0) / 1000:.0f}, "
                   f"{x.get('wait_end') or ('timeout' if x.get('gate_timeout') else 'gate')} | {e / 1000:.0f}, {why}, {h} | "
                   f"{min(later) if later else '-'}")
    if rows:
        out.append(f"summary: n {len(rows)}; ends {{{', '.join(f'{w}: {sum(1 for x in rows if x[1] == w)}' for w in ('gate', 'plateau', 'timeout'))}}}; "
                   f"wait s {spread([x[0] / 1000 for x in rows])}; headroom at the end {spread([x[2] for x in rows])}; "
                   f"headroom at the r16 end {spread([x[4] for x in rows])}; further fall after the end "
                   f"{spread([max(0.0, x[2] - x[3]) for x in rows if x[3] is not None])}, > 0.03 in "
                   f"{sum(1 for x in rows if x[3] is not None and x[2] - x[3] > 0.03)}")
    return out


def r24_checks(r):
    out = [l for l in r23_checks(r) if not l.startswith('kernel release')]
    specs = {sp.get('id'): sp for sp in r.get('specs') or []}
    runs = r.get('runs', [])
    bad = [x.get('run_tag') for x in runs if x.get('gate') and x.get('gate_rule') != (specs.get(x.get('id')) or {}).get('gate_rule')]
    ends = {}
    for x in runs:
        if x.get('gate'):
            ends.setdefault(x.get('gate_rule'), {}).setdefault(x.get('wait_end'), 0)
            ends[x.get('gate_rule')][x.get('wait_end')] += 1
    plug = [x.get('run_tag') for x in runs if x.get('plugged_run') or x.get('plugged')]
    pauses = [f"{x.get('run_tag')}({sum(p[1] for p in x['plug_pauses']) // 1000} s)" for x in runs if x.get('plug_pauses')]
    vm = sum(1 for x in runs if isinstance(x.get('vmstat_after'), dict))
    zs = r.get('thermal_zones') or []
    zok = [z for z in zs if z.get('temp') is not None]
    base = (r.get('idle_baseline') or {}).get('samples') or []
    out.append(f"r24: gate rule != spec in {','.join(bad) or 'none'}; wait ends per rule {ends}; charger pauses "
               f"{','.join(pauses) or 'none'}; runs with a charger at start or end (FLAG) {','.join(plug) or 'none'}; vmstat "
               f"readable in {vm} of {len(runs)} rows; thermal zones readable {len(zok)} of {len(zs)}"
               f"{' (' + str(zs[0].get('error')) + ')' if zs and not zok else ''}; idle baseline {len(base)} samples")
    out.append(f"kernel release {r.get('kernel_release')!r}; owner note {r.get('owner_note')!r}")
    return out


def quad_sel_effect(q, sel):
    """mean(num) - mean(off) over the quad's runs that pass sel (r25 plain runs: one off and one num per quad)."""
    u = [x.get('decode_tps') for x in q if sel(x) and x.get('arm') == 'num']
    v = [x.get('decode_tps') for x in q if sel(x) and x.get('arm') == 'off']
    return None if not u or not v or None in u + v else st.mean(u) - st.mean(v)


def r24_tables(r, proto='r24', sel=None, label='', extras=True):
    """r24's tables; kit r25 calls it on proto 'r25' pooled over kv (sel None) and on plain runs (sel, extras off)."""
    runs = [x for x in r.get('runs', []) if x.get('proto') == proto]
    if not runs:
        return []
    models = r21_models(r)
    sc = lambda m, a=None, g=None: [x for x in runs if x.get('model') == m and x.get('phase') == 'on'
                                    and (a is None or x.get('arm') == a) and (g is None or x.get('gate_rule') == g)
                                    and (sel is None or sel(x))]
    eff = quad_effect if sel is None else lambda q: quad_sel_effect(q, sel)
    out = extras_head(r) if extras else []
    out.append(f'kit {proto}{label}, scored runs (phase on), per model, arm and gate rule (*device*):')
    for m in models:
        for a in ('off', 'num'):
            for g in ('r16', 'plateau'):
                out.append(stats_line(f"  {m} {a} {g:7}", sc(m, a, g)))
    out.append('num minus off, mean over ABBA quads of mean(num) - mean(off), tok/s ± SE (flag-free = quads without a flagged run)'
               + ('' if sel is None else ' [only the selected runs of each quad: one num, one off]') + ':')
    for m in models:
        qs = [q for g in r23_blocks(runs, m) for q in r22_quads([x for x in runs if x.get('group') == g and x.get('phase') == 'on'])
              if len({x.get('gate_rule') for x in q}) == 1]
        e = lambda g=None, clean=False: [eff(q) for q in qs if (g is None or q[0].get('gate_rule') == g)
                                         and not (clean and any(flagged(x) for x in q if sel is None or sel(x)))]
        out.append(f"  {m}: all {fmt_se(e())} | r16 {fmt_se(e('r16'))} | plateau {fmt_se(e('plateau'))} | flag-free {fmt_se(e(clean=True))}")
    metrics = [('tok/s', lambda x: x.get('decode_tps'), 2), ('wait s', lambda x: None if x.get('wait_ms') is None else x['wait_ms'] / 1000, 1),
               ('res before MB', res_mb, 0), ('lost in wait MB', lambda x: x.get('wait_lost_mb'), 0),
               ('heat at start', lambda x: x.get('heat'), 3), ('big-core MHz', lambda x: big_min(x.get('mhz_before')), 0),
               ('cap MHz', lambda x: cap_of(x.get('cap_before')), 0)]
    out.append('plateau minus r16 per model, both arms: per block mean(plateau runs) - mean(r16 runs), mean over blocks ± SE; '
               'all runs, Welch:')
    for m in models:
        parts = []
        for name, f, d in metrics:
            dd = []
            for g in r23_blocks(runs, m):
                b = [x for x in sc(m) if x.get('group') == g]
                u = [f(x) for x in b if x.get('gate_rule') == 'plateau']
                v = [f(x) for x in b if x.get('gate_rule') == 'r16']
                if u and v and None not in u + v:
                    dd.append(st.mean(u) - st.mean(v))
            parts.append(f"{name} {fmt_se(dd)}, runs {welch([f(x) for x in sc(m, None, 'plateau')], [f(x) for x in sc(m, None, 'r16')], d)}")
        out.append(f"  {m}: " + ' | '.join(parts))
    return out + extras_tail(runs) if extras else out


def flagged(x):
    return bool(x.get('plugged_run') or x.get('plugged'))


def extras_head(r):
    """r24: the idle baseline and the readable thermal zones at the start."""
    out = []
    base = (r.get('idle_baseline') or {}).get('samples') or []
    if base:
        out.append('idle baseline before the first run (ms, headroom, battery C, plugged, status): ' +
                   ' '.join(str(b) for b in base))
    zs = [z for z in r.get('thermal_zones') or [] if z.get('temp') is not None]
    if zs:
        out.append('thermal zones at the start: ' + ', '.join(f"{z.get('type')}={z.get('temp')}" for z in zs))
    return out


def stats_line(head, v):
    """r24's one-line summary of a set of scored runs."""
    st_ = r22_stats(v)
    return (f"{head}: n {st_['n']} (flagged {sum(map(flagged, v))}) tok/s {fmt(st_['tps'])} "
            f"[{fmt(st_['tps_min'])}..{fmt(st_['tps_max'])}] | wait s {fmt(med((x.get('wait_ms') or 0) / 1000 for x in v), 0)} "
            f"ends {sorted(x.get('wait_end') or '-' for x in v)} | heat at start {fmt(med(x.get('heat') for x in v), 3)} | "
            f"res before MB {fmt(st_['res_before'], 0)} lost in wait {fmt(st_['wait_lost_mb'], 0)} | decode majflt "
            f"{fmt(st_['dec_majflt'], 0)} inblock MB {fmt(st_['dec_inblock_mb'], 0)} | pswpin prefill / decode "
            f"{fmt(med(vm_delta(x, 'vmstat_before', 'vmstat_first', 'pswpin') for x in v), 0)} / "
            f"{fmt(med(vm_delta(x, 'vmstat_first', 'vmstat_after', 'pswpin') for x in v), 0)} | VmSwap before MB "
            f"{fmt(med(memv(x.get('mem_before'), 'swap') for x in v), 0)} | hold cpu ms {fmt(st_['hold_cpu_ms'], 0)}")


def extras_tail(runs):
    """r24: charger runs, swap and faults per gated run, then r23's hold table."""
    out = []
    plug = [x for x in runs if flagged(x)]
    out.append(f"runs with a charger at their start or end: {len(plug)}" +
               ''.join(f"\n  {x.get('pos')} {x.get('run_tag')} plugged before {x.get('plugged')} after {x.get('plugged_after')} "
                       f"tok/s {fmt(x.get('decode_tps'))}" for x in plug))
    out.append('swap and faults per gated run (system pswpin pages, prefill / decode; the run\'s majflt prefill / decode from c2-io):')
    for x in [x for x in runs if x.get('gate')]:
        i = io_of(x) or {}
        out.append(f"  {x.get('pos')} {x.get('run_tag')} | pswpin {vm_delta(x, 'vmstat_before', 'vmstat_first', 'pswpin')} / "
                   f"{vm_delta(x, 'vmstat_first', 'vmstat_after', 'pswpin')} | pgmajfault {vm_delta(x, 'vmstat_before', 'vmstat_first', 'pgmajfault')}"
                   f" / {vm_delta(x, 'vmstat_first', 'vmstat_after', 'pgmajfault')} | majflt {i.get('pre_majflt')} / {i.get('dec_majflt')} | "
                   f"VmSwap MB {memv(x.get('mem_before'), 'swap')} -> {memv(x.get('mem_first'), 'swap')} -> {memv(x.get('mem_after'), 'swap')}")
    return out + hold_lines(runs)


# kit r25 (memo-26, R64): r24 plus the kv factor (VOLLIE_KV_DONTNEED=1 in half of the scored runs, c1-olmoe 0008)
KVCLEAR_RE = re.compile(r'c2-kvclear run=(\S+) kind=(\w+) on=(\d) bytes=(\d+) released=(\d+) why=(\w+) us=(\d+)')


def kv_on(x):
    return 'VOLLIE_KV_DONTNEED' in (x.get('env') or env_of(x))


def kvclear_of(x):
    """The row's c2-kvclear lines (one per buffer cleared): [(kind, on, bytes, released, why, us)]."""
    return [(m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6), int(m.group(7)))
            for l in x.get('c2') or [] for m in [KVCLEAR_RE.search(l)] if m and m.group(1) == x.get('run_tag', m.group(1))]


def released_mb(x):
    k = kvclear_of(x)
    return sum(c[3] for c in k) / 1e6 if k else None


def pre_prompt_majflt(x):
    """memo-24 §A2.2: the run's majflt minus the c2-io window (prompt start to end)."""
    i = io_of(x) or {}
    if x.get('majflt') is None or i.get('pre_majflt') is None or i.get('dec_majflt') is None:
        return None
    return x['majflt'] - i['pre_majflt'] - i['dec_majflt']


def dec_ms_tok(x):
    """Decode graph ms per token from the c2-io window (independent of sampling and chunk delivery)."""
    i = io_of(x) or {}
    return i['dec_ms'] / i['decode_graphs'] if i.get('dec_ms') is not None and i.get('decode_graphs') else None


KV_METRICS = [('pre-prompt majflt', pre_prompt_majflt, 0),
              ('pswpin start->first chunk', lambda x: vm_delta(x, 'vmstat_before', 'vmstat_first', 'pswpin'), 0),
              ('TTFT ms', lambda x: x.get('ttft_ms'), 1), ('decode graph ms/token', dec_ms_tok, 2),
              ('tok/s', lambda x: x.get('decode_tps'), 2), ('VmSwap at end MB', lambda x: memv(x.get('mem_after'), 'swap'), 0),
              ('released MB', released_mb, 1)]


def r25_checks(r):
    out = r24_checks(r)
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r25']
    bad_id = [x.get('run_tag') for x in runs if kv_on(x) != str(x.get('id', '')).endswith('_kv')]
    outside = [x.get('run_tag') for x in runs if kv_on(x) and x.get('phase') != 'on']
    nokv = [x.get('run_tag') for x in runs if kv_on(x) and not (released_mb(x) or 0) > 0]
    leak = [x.get('run_tag') for x in runs if not kv_on(x) and (released_mb(x) or 0) > 0]
    noline = [x.get('run_tag') for x in runs if not kvclear_of(x)]
    why = {}
    for x in runs:
        for c in kvclear_of(x):
            k = (('kv' if kv_on(x) else 'plain'), c[0], c[4])
            why[k] = why.get(k, 0) + 1
    out.append(f"r25 kv factor: id suffix != env in {','.join(bad_id) or 'none'}; kv outside scored runs "
               f"{','.join(outside) or 'none'}; rows without a c2-kvclear line {','.join(noline) or 'none'}; c2-kvclear lines "
               f"(run, kind, why): {dict(sorted(why.items()))}")
    out.append(f"r25 VOID check (R64 §4): released = 0 in a kv run: {','.join(nokv) or 'none'}; released > 0 in a run "
               f"without kv: {','.join(leak) or 'none'} -> {'VOID' if nokv or leak else 'ok'}")
    for m in r21_models(r):
        parts = []
        for a in ('off', 'num'):
            t = {kv: {md5(x['text'])[:8] for x in runs if x.get('model') == m and x.get('arm') == a and kv_on(x) == kv
                      and x.get('text') is not None} for kv in (True, False)}
            same = len(t[True] | t[False]) == 1
            parts.append(f"{a} kv {sorted(t[True])} plain {sorted(t[False])}: {'same' if same else 'DIFFER (kill rule)'}")
        out.append(f"{m} (r25) texts kv against plain: " + '; '.join(parts))
    return out


def r25_kv(r):
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r25' and x.get('phase') == 'on']
    out = ['kv factor (R64), per model: kv minus plain; paired = within block x gate rule x arm (one kv, one plain run), mean '
           '± SE over pairs; Welch = all kv runs against all plain runs; medians kv / plain (*device*):']
    for m in r21_models(r):
        rs = [x for x in runs if x.get('model') == m]
        cells = {}
        for x in rs:
            cells.setdefault((x.get('group'), x.get('gate_rule'), x.get('arm')), {True: [], False: []})[kv_on(x)].append(x)
        pairs = [(c[True][0], c[False][0]) for c in cells.values() if len(c[True]) == 1 and len(c[False]) == 1]
        odd = [f"{'/'.join(map(str, k))}({len(c[True])} kv, {len(c[False])} plain)" for k, c in cells.items()
               if not (len(c[True]) == 1 and len(c[False]) == 1)]
        kv, pl = [x for x in rs if kv_on(x)], [x for x in rs if not kv_on(x)]
        out.append(f"  {m}: {len(kv)} kv and {len(pl)} plain scored runs, {len(pairs)} pairs; unpaired cells "
                   f"{','.join(odd) or 'none'}")
        for name, f, d in KV_METRICS:
            p = [None if f(a) is None or f(b) is None else f(a) - f(b) for a, b in pairs]
            out.append(f"    {name}: paired {fmt_se(p, d)} | Welch {welch([f(x) for x in kv], [f(x) for x in pl], d)} | "
                       f"median {fmt(med(f(x) for x in kv), d)} / {fmt(med(f(x) for x in pl), d)}")
        fo, fp = med(pre_prompt_majflt(x) for x in kv), med(pre_prompt_majflt(x) for x in pl)
        a, b = [x.get('ttft_ms') for x in kv if x.get('ttft_ms') is not None], [x.get('ttft_ms') for x in pl if x.get('ttft_ms') is not None]
        d, se = diff(b, a)  # mean(kv) - mean(plain), Welch SE
        faults = fo is not None and fp is not None and fo < fp / 2
        ttft = 'n/a' if d is None else 'lower > 2 SE' if d < -2 * se else 'HIGHER > 2 SE (kill)' if d > 2 * se else 'within 2 SE'
        out.append(f"    R64 §4 readout (texts and void in the checks above): faults median kv < half of plain: "
                   f"{'yes' if faults else 'no'}; TTFT {'-' if d is None else f'{d:+.1f} ± {se:.1f} ms'}: {ttft} -> "
                   f"{'pass' if faults and d is not None and d < -2 * se else 'kill' if d is not None and d > 2 * se else 'unresolved'}")
        for g in ('r16', 'plateau'):
            q = [(a, b) for a, b in pairs if a.get('gate_rule') == g]
            out.append(f"    gate {g}: pairs {len(q)} | TTFT ms {fmt_se([a['ttft_ms'] - b['ttft_ms'] for a, b in q if a.get('ttft_ms') is not None and b.get('ttft_ms') is not None], 1)} "
                       f"| pre-prompt majflt {fmt_se([None if pre_prompt_majflt(a) is None or pre_prompt_majflt(b) is None else pre_prompt_majflt(a) - pre_prompt_majflt(b) for a, b in q], 0)}")
    return out


def r25_tables(r):
    if not any(x.get('proto') == 'r25' for x in r.get('runs', [])):
        return []
    return (r25_kv(r) + ["R63's gate and num comparisons, kv on and off pooled:"] + r24_tables(r, 'r25', label=' (kv pooled)')
            + ["R63's gate and num comparisons, plain runs only:"]
            + r24_tables(r, 'r25', sel=lambda x: not kv_on(x), label=' (plain only)', extras=False)
            + (kv_mech([x for x in r.get('runs', []) if x.get('proto') == 'r25'], r21_models(r))
               if any(c[2] is not None for x in r.get('runs', []) for c in kvtags_of(x)) else []))


# kit r26 (memo-27): one arm per model (Granite num, OLMoE off), r16's gate only, the kv factor in two ABBA quads per
# block (on off off on / off on on off; even blocks the reverse); c1-olmoe 0009 adds tag=0x.. to the c2-kvclear line.
TAG_RE = re.compile(r' tag=0x([0-9a-fA-F]+)')


def kvtags_of(x):
    """The row's c2-kvclear lines as [(kind, why, tag)]; tag None on lines without tag= (0008, kits before r25b)."""
    return [(m.group(2), m.group(6), None if t is None else '0x' + t.group(1).lower())
            for l in x.get('c2') or [] for m in [KVCLEAR_RE.search(l)] if m and m.group(1) == x.get('run_tag', m.group(1))
            for t in [TAG_RE.search(l)]]


def kv_mech(runs, models):
    """Mechanism per model (R65 §4): kv runs whose every clear line says why=ok; why and tag counts per kind."""
    out = ['kv mechanism per model: kv runs with why=ok in every c2-kvclear line; lines per (kind, why) and (kind, tag), '
           'kv runs / plain runs (*device*):']
    for m in models:
        rs = [x for x in runs if x.get('model') == m and x.get('phase') == 'on']
        kv, pl = [x for x in rs if kv_on(x)], [x for x in rs if not kv_on(x)]
        ok = [x for x in kv if kvtags_of(x) and all(c[1] == 'ok' for c in kvtags_of(x))]
        cnt = lambda v, i: dict(sorted({k: sum(1 for x in v for c in kvtags_of(x) if (c[0], c[i]) == k)
                                        for k in {(c[0], c[i]) for x in v for c in kvtags_of(x)}}.items(), key=str))
        out.append(f"  {m}: why=ok in every line {len(ok)} of {len(kv)} kv runs (not: {','.join(x.get('run_tag') for x in kv if x not in ok) or 'none'})"
                   f" | why kv {cnt(kv, 1)} plain {cnt(pl, 1)} | tag kv {cnt(kv, 2)} plain {cnt(pl, 2)} | released MB per kv run "
                   f"median {fmt(med(released_mb(x) for x in kv), 1)} min {fmt(min((released_mb(x) or 0 for x in kv), default=None), 1)}")
    return out


def r26_checks(r):
    out = r24_checks(r)
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r26']
    arm = (r.get('models') or {})
    bad_arm = [x.get('run_tag') for x in runs if (arm.get(x.get('model')) or {}).get('arm') not in (None, x.get('arm'))]
    rules = sorted({x.get('gate_rule') for x in runs if x.get('gate')}, key=str)
    bad_id = [x.get('run_tag') for x in runs if kv_on(x) != str(x.get('id', '')).endswith('_kv')]
    outside = [x.get('run_tag') for x in runs if kv_on(x) and x.get('phase') != 'on']
    nokv = [x.get('run_tag') for x in runs if kv_on(x) and not (released_mb(x) or 0) > 0]
    leak = [x.get('run_tag') for x in runs if not kv_on(x) and (released_mb(x) or 0) > 0]
    noline = [x.get('run_tag') for x in runs if not kvclear_of(x)]
    notag = [x.get('run_tag') for x in runs if any(c[2] is None for c in kvtags_of(x))]
    out.append(f"r26: arms per model { {m: sorted({str(x.get('arm')) for x in runs if x.get('model') == m}) for m in r21_models(r)} }, arm != report "
               f"in {','.join(bad_arm) or 'none'}; gate rules of gated runs {rules}; kv id suffix != env in {','.join(bad_id) or 'none'}; "
               f"kv outside scored runs {','.join(outside) or 'none'}; rows without a c2-kvclear line {','.join(noline) or 'none'}; "
               f"lines without tag= in {','.join(notag) or 'none'}")
    out.append(f"r26 void check: released > 0 in a run without kv: {','.join(leak) or 'none'} -> {'VOID' if leak else 'ok'}; "
               f"kv runs releasing 0 (mechanism miss): {','.join(nokv) or 'none'}")
    for m in r21_models(r):
        parts = []
        for a in sorted({x.get('arm') for x in runs if x.get('model') == m}, key=str):
            t = {kv: {md5(x['text'])[:8] for x in runs if x.get('model') == m and x.get('arm') == a and kv_on(x) == kv
                      and x.get('text') is not None} for kv in (True, False)}
            same = len(t[True] | t[False]) == 1
            parts.append(f"{a} kv {sorted(t[True])} plain {sorted(t[False])}: {'same' if same else 'DIFFER (kill rule)'}")
        out.append(f"{m} (r26) texts kv against plain: " + '; '.join(parts))
    return out


def kv_quads(rs):
    """One block's scored runs in list order: the ABBA quads of the kv factor (on off off on or off on on off)."""
    qs = [rs[i:i + 4] for i in range(0, len(rs) - 3, 4)]
    return [q for q in qs if [kv_on(x) for x in q] in ([True, False, False, True], [False, True, True, False])]


def r26_kv(r):
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r26' and x.get('phase') == 'on']
    out = ['kv factor (r26), per model: kv minus plain; paired = per ABBA quad mean(kv) - mean(plain), mean ± SE over '
           'quads; Welch = all kv runs against all plain runs; medians kv / plain (*device*):']
    for m in r21_models(r):
        rs = [x for x in runs if x.get('model') == m]
        qs = [q for g in r23_blocks(rs, m) for q in kv_quads([x for x in rs if x.get('group') == g])]
        kv, pl = [x for x in rs if kv_on(x)], [x for x in rs if not kv_on(x)]
        out.append(f"  {m}: {len(kv)} kv and {len(pl)} plain scored runs, {len(qs)} ABBA quads, "
                   f"{len(rs) - 4 * len(qs)} scored runs outside a quad")
        eff = lambda q, f: None if None in [f(x) for x in q] else st.mean(f(x) for x in q if kv_on(x)) - st.mean(f(x) for x in q if not kv_on(x))
        for name, f, d in KV_METRICS:
            out.append(f"    {name}: paired {fmt_se([eff(q, f) for q in qs], d)} | Welch {welch([f(x) for x in kv], [f(x) for x in pl], d)} | "
                       f"median {fmt(med(f(x) for x in kv), d)} / {fmt(med(f(x) for x in pl), d)}")
        fq = [v for v in (eff(q, pre_prompt_majflt) for q in qs) if v is not None]
        tq = [v for v in (eff(q, lambda x: x.get('ttft_ms')) for q in qs) if v is not None]
        fm = st.mean(fq) if fq else None
        tm, tse = (st.mean(tq), st.stdev(tq) / len(tq) ** 0.5) if len(tq) > 1 else (None, None)
        ttft = 'n/a' if tm is None else 'lower > 2 SE' if tm < -2 * tse else 'HIGHER > 2 SE (kill)' if tm > 2 * tse else 'within 2 SE'
        out.append(f"    effect readout (R65 §4 wording, paired over quads): pre-prompt majflt mean {fmt(fm, 0)}: "
                   f"{'lower' if fm is not None and fm < 0 else 'not lower'}; TTFT {'-' if tm is None else f'{tm:+.1f} ± {tse:.1f} ms'}: "
                   f"{ttft} -> {'pass' if fm is not None and fm < 0 and tm is not None and tm < -2 * tse else 'kill' if tm is not None and tm > 2 * tse else 'unresolved'}")
    return out


def r26_tables(r):
    runs = [x for x in r.get('runs', []) if x.get('proto') == 'r26']
    if not runs:
        return []
    models = r21_models(r)
    out = r26_kv(r) + kv_mech(runs, models) + extras_head(r)
    out.append('kit r26, scored runs (phase on), per model, arm and kv (*device*):')
    for m in models:
        for a in sorted({x.get('arm') for x in runs if x.get('model') == m}, key=str):
            for kv in (True, False):
                out.append(stats_line(f"  {m} {a} {'kv' if kv else 'plain'}",
                                      [x for x in runs if x.get('model') == m and x.get('arm') == a and x.get('phase') == 'on' and kv_on(x) == kv]))
    return out + extras_tail(runs)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('report', nargs='?')
    a.add_argument('--min-mhz', type=int, default=1800)
    a.add_argument('--sigma', type=float, default=0.4)
    a.add_argument('--fault-run', type=int, default=1000, help='plain-run major faults above which a run is fault-bound')
    a.add_argument('--prompt', default=os.path.join(HERE, 'c2_official_prompt.txt'))
    a.add_argument('--r16', default=R16, help="campaign 1's kit r16 report, for kit r20's position-by-position table")
    a.add_argument('--wait-s', type=float, default=30, help="kit r22: a gate wait of at least this many seconds is 'waited'")
    a.add_argument('--selftest', action='store_true')
    a.add_argument('--plateau', action='store_true', help="replay kit r24's plateau gate on the report's heat series (r22 on)")
    o = a.parse_args(argv)
    if o.selftest:
        return selftest()
    if not o.report:
        a.error('report.json required')
    r = load(o.report)
    print(f"kit {r.get('kit')} app {r.get('app')} started {r.get('started')} stopped_early {r.get('stopped_early')}")
    if o.plateau:
        print('\n'.join(plateau_replay(r)))
        return 0
    r16 = load(o.r16) if r.get('kit') == 'r20' and o.r16 and os.path.exists(o.r16) else None
    parts = (r21_checks(r), r21_tables(r)) if r.get('kit') == 'r21' else (r22_checks(r), r22_tables(r, o.wait_s)) \
        if r.get('kit') == 'r22' else (r23_checks(r), r23_tables(r)) if r.get('kit') == 'r23' \
        else (r24_checks(r), r24_tables(r)) if r.get('kit') == 'r24' \
        else (r25_checks(r), r25_tables(r)) if r.get('kit') == 'r25' \
        else (r26_checks(r), r26_tables(r)) if r.get('kit') == 'r26' else (
        checks(r, o.prompt), identity(r), groups(r, o.min_mhz, o.fault_run), ab(r, o.min_mhz, o.sigma),
        r16_compare(r, r16), waits(r), warm(r))
    for part in parts:
        if part:
            print('\n'.join(part))
    return 0


def selftest():
    import random
    import tempfile
    p = 'Read these notes.'
    mk = lambda **k: {'group': 'stock', 'kind': 'official32', 'env': {}, 'rep': 0, 'id': 'stock', 'repack_loaded': True,
                      'mhz_after': 'mhz=1804,1804,1804,1804,2112,2112,2112,2035', 'majflt': 0, 'minflt': 10,
                      'text': 'The ocean', 'cached_prompt_tokens': 0, 'prompt_tokens': 151, **k}
    stack = {'VOLLIE_TINY_OP': '65536'}
    r = {'kit': 'r18', 'app': 't', 'prompt': p,
         'model': {'file': 'olmoe-Mi.gguf', 'bytes': 3072399552, 'expected_bytes': 3072399552,
                   'verified_marker': SHA, 'sha256_end': SHA},
         'libs': {'libllama.so': 'aa' * 32}, 'build': {'libs': {'libllama.so': 'aa' * 32}},
         'template': {'architecture': 'olmoe', 'custom_template': False, 'rendered': expected_render(p),
                      'rendered_tokens': 151},
         'runs': [mk(decode_tps=20.0), mk(decode_tps=22.0, rep=1), mk(decode_tps=24.0, rep=2, gate_timeout=True),
                  mk(group='stack', id='stack', env=stack, env_set=stack, decode_tps=25.0),
                  mk(group='profiled', id='prof', kind='profiled32', env=stack, decode_tps=21.0,
                     chunk_ms=[100.0, 145.0, 190.0, 240.0],
                     prof=['x cpu-prof ab=- tok=2 us/token=44000 majflt=3 minflt=1 mhz=0/0 run=prof.0 prompt=2 blk=0 '
                           'threads=3 ops=1 barriers=1 syncs=1 skipped=0 pairs=0 head=0 router=0 mm=0 mmid=0 attn=0 '
                           'other=0 wait=0 | ADD 1:1+1']),
                  mk(group='skip', id='skip', env={'VOLLIE_SKIP_MASS': '900'}, text='Skipped', decode_tps=26.0,
                     c2=['c2-skip-run: run=skip.0 prompt=5 tokens=183 pairs=1464 pairs_skipped=13.5% '
                         'expert_bytes_skipped=13.6% MB_per_token_selected=1.0 MB_per_token_skipped=0.1 '
                         'batch_bytes_unread=2.0%']),
                  mk(group='norepack', id='nr', repack_loaded=False, env=stack, decode_tps=19.0, text='Other',
                     mhz_after='mhz=1804,1804,1804,1804,1555,1555,1555,1555')]}
    # A/B: 2 generations, warm-up 16, blocks of 8 in A B B A order; B is 1.0 ms faster per token, noise sd 1 ms/token;
    # one A block has a major fault, one B block a low clock: both dropped.
    rnd = random.Random(7)
    for gen in range(2):
        tup, chunks, t_ms = [], [0.0], 0.0
        for tok in range(1, 16 + 8 * 24 + 1):
            blk = 0 if tok <= 16 else (tok - 17) // 8 + 1
            side = 'A' if blk == 0 or blk % 4 in (1, 0) else 'B'
            us = int((40.0 - (1.0 if side == 'B' else 0.0) + rnd.gauss(0, 1.0)) * 1000)
            maj = 5 if (gen, blk) == (0, 1) else 0
            m4 = 1555 if (gen, blk) == (1, 2) else 2112
            tup.append([side, tok, us, maj, 9, m4, 2035, 2, blk, 3, 469, 242 if side == 'B' else 469, 355, 0, 0,
                        1000, 200, 1500, 20000, 100, 1000, 5000])
            t_ms += us / 1000 + 0.5
            chunks.append(t_ms)
        r['runs'].append({'group': 'ab', 'kind': 'abba', 'id': 'ab_tiny', 'rep': gen, 'env': {'VOLLIE_AB': 'knob:TINY_OP:0:65536'},
                          'tok': tup, 'chunk_ms': chunks, 'text': 'story', 'decode_tps': 25.0})
    g = groups(r, 1800)
    assert g[0].startswith('stock: n 3 printed tok/s median 22.00 min 20.00 max 24.00'), g[0]
    assert 'stock#2 (gate timed out (heat -))' in g[0] and 'stock#0' not in g[0], g[0]
    assert 'steady ms/token 44.00 (prof)' in g[2] and 'decode majflt 3' in g[2], g[2]
    assert 'expert bytes skipped 13.60 %' in g[3], g[3]
    assert 'clock 1555<1800' in g[4], g[4]
    assert steady_ms({'chunk_ms': [100.0, 145.0, 190.0, 240.0]}) == (45.0, 'chunk')
    idt = identity(r)
    assert 'stock == stack: yes' in idt, idt
    assert any('repack=False' in l and 'identical' in l for l in idt), idt
    c = checks(r, None)
    assert c[0].endswith('OK') and c[1].endswith('= build') and "rendered==OLMoE's own: yes" in c[3], c
    a = ab(r, 1800, 0.4)
    assert len(a) == 1 and 'blocks A 24 (clean 23) B 24 (clean 23)' in a[0], a
    d = float(re.search(r'graph B-A (-?[\d.]+) ± ([\d.]+)', a[0]).group(1))
    s = float(re.search(r'graph B-A (-?[\d.]+) ± ([\d.]+)', a[0]).group(2))
    assert abs(d + 1.0) < 3 * s and s < 0.4 and 'graph B-A' in a[0] and ': resolved' in a[0], a[0]
    assert 'barriers 469->242' in a[0], a[0]
    dw = float(re.search(r'wall B-A (-?[\d.]+)', a[0]).group(1))
    assert abs(dw - d) < 0.01, (dw, d)  # the synthetic wall adds a constant 0.5 ms per token
    assert verdict(-0.5, 0.6, 0.4).startswith('UNRESOLVED') and verdict(-2.0, 0.6, 0.4).startswith('resolved (')
    r['runs'][0]['text'] = 'changed'
    assert any('DIFFER' in l for l in identity(r))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('c2gz:' + base64.b64encode(gzip.compress(json.dumps(r).encode())).decode())
    assert load(f.name)['kit'] == 'r18'
    os.unlink(f.name)
    # r19: campaign 1's per-token lines, residency, the three hygiene flags, the r19 text check.
    c1l = lambda tok, maj, us: (f'cpu-prof ab=A tok={tok} us/token={us} head=900 router=120 majflt={maj} minflt=5 mhz=2112/2035 '
                                f'nodes=469 barriers=241 kern=1 fuse=1 chunk2=1 | MUL_MAT_ID 32:30000+500 MUL_MAT 120:9000+300')
    res = {'experts_mb': 1500, 'rest_mb': 40, 'total_mb': 1540, 'ms': 30}
    r19 = {'kit': 'r19', 'runs': [
        mk(group='c1_off', id='c1_off', env={'GGML_MMID': '0'}, decode_tps=5.0, majflt=25000, res_before=res, res_after=res,
           cap_after='cap=1440/1420', mhz_after='mhz=1804,1804,1804,1804,1440,1440,1440,1420', gate_timeout=True, heat=0.97),
        mk(group='c1_ship', id='c1_ship', env={'GGML_EXPERT_IO': '1'}, decode_tps=6.0, gate_unknown=True, gate_timeout=True),
        mk(group='c1_ship_prof', id='c1_ship_prof', kind='profiled32', env={'GGML_CPU_PROF': '1'}, tok_format='c1',
           tok=[['A', 1, 50000, 900, 120, 300, 5, 2112, 2035, 469, 241, 1, 1, 1], ['B', 2, 45000, 900, 120, 2, 5, 2112, 2035, 469, 241, 1, 1, 1]],
           prof=[c1l(1, 300, 50000), c1l(2, 2, 45000)]),
        mk(group='c1_ship_io_prof', id='c1_ship_io_prof', kind='profiled32', env={'GGML_CPU_PROF': '1'},
           prof=[c1l(2, 0, 44000), c1l(3, 1, 46000)])]}
    t = tokens(r19['runs'][2])
    assert t[1]['us'] == 45000 and t[1]['mmid'] == 30000 and t[1]['wait'] == 800 and t[0]['majflt'] == 300, t
    assert steady_ms(r19['runs'][2]) == (45.0, 'prof') and steady_ms(r19['runs'][3]) == (45.0, 'prof')
    g = groups(r19, 1800)
    assert 'throttled: clock 1420<1800' in g[0] and 'fault-bound: run majflt 25000' in g[0] and 'gate timed out (heat 0.97)' in g[0], g[0]
    assert 'resident before experts 1500 rest 40 MB; after experts 1500 rest 40 MB; caps after 1440/1420' in g[0], g[0]
    assert 'gate unknown' in g[1] and 'gate timed out' not in g[1], g[1]
    assert 'fault-bound: decode majflt 302' in g[2] and 'decode majflt 1)' in g[3], (g[2], g[3])
    assert 'r19 rows c1_off/c1_ship/c1_ship_prof/c1_ship_io_prof give one text: yes' in identity(r19), identity(r19)
    r19['runs'][1]['text'] = 'other'
    assert any('give one text: NO (2 texts)' in l for l in identity(r19))
    # r20: positions against r16, waits with and without keep-warm, the warm block, stories out of the text check.
    r16 = load(R16)
    assert len(r16['runs']) == 21 and r16['runs'][0]['phase'] == 'baseline'
    kwl = lambda n, cpu, ab=0: (f'c2-keepwarm pass={n} period_s=10 ranges=48 mb=2769 resident_mb=1800 touched=460800 '
                                f'wall_us=300000 cpu_us={cpu} aborted={ab}')
    ship = {'GGML_TINY_OP': '262144', 'GGML_EXPERT_IO': '1', 'VOLLIE_SAMPLER_BYPASS': '1'}
    rz = lambda t: {'experts_mb': t - 40, 'rest_mb': 40, 'total_mb': t, 'ms': 300}
    m20 = lambda pos, **k: mk(**{'pos': pos, 'proto': 'r16', 'phase': 'on', 'group': 'g1', 'id': 'ship', 'env': ship,
                                 'gate': 0.65, 'cool_ms': 90000, 'idle_ms': 95000, 'heat': 0.8, 'gate_timeout': True,
                                 'ttft_ms': 8800.0, 'res_before': rz(1000), 'res_after': rz(2000),
                                 'mem_before': 'avail=1MB cached=2000MB file=900MB', **k})
    r20 = {'kit': 'r20', 'runs': [
        m20(1, phase='baseline', group='baseline', decode_tps=4.0),
        m20(2, phase='warmup', group='warmup', kind='story200', id='ship_story', text='A lighthouse', prompt_tokens=17,
            decode_tps=5.0),
        m20(3, decode_tps=7.0, res_before=rz(1500)),
        m20(4, proto='kw', group='kw', id='ship_kw', env={**ship, 'GGML_KEEP_WARM': '10'}, decode_tps=8.0, res_before=rz(1200)),
        m20(5, proto='kw', group='kw', id='ship_kw', env={**ship, 'GGML_KEEP_WARM': '10'}, decode_tps=9.0, res_before=rz(1950),
            c2=[kwl(1, 250000), kwl(2, 260000, 1)]),
        m20(6, proto='warm', group='warm_ship', phase='warmup', id='ship_wu', gate=None, decode_tps=10.0, heat=0.8,
            heat_after=0.82, mhz_before='mhz=1,1,1,1,1555,1555,1555,1555'),
        m20(7, proto='warm', group='warm_ship', id='ship_w', gate=None, decode_tps=12.0, heat=0.82, heat_after=0.84),
        m20(8, proto='warm', group='warm_ship', id='ship_w', gate=None, decode_tps=14.0, heat=0.84, heat_after=0.85)]}
    t = r16_compare(r20, r16)
    assert t[1].startswith('pos  1 baseline official32 | r16 12.27 | 0.870 | 90 | - | 9.2 | 1555 | file 313 | 2490 | ') \
        and '||  r20 4.00 | 0.800T | 90 | 95 | 8.8 | 2112 | res 960+40 -> 2000 | 2000 | 0' in t[1], t[1]
    assert 'r16 12.43' in t[2] and 'r16 13.97' in t[3] and 'file 658' in t[3], t
    assert "r16 median 20.51 (n 10); r20 median 7.00 (n 1)" in t[-1], t[-1]
    w = waits(r20)
    assert any('wait before pos 5 (kw): idle 95 s, keep-warm on, lost 50 MB; passes 2, CPU 510 ms' in l and 'aborted 1' in l
               for l in w), w
    assert 'waits with keep-warm off: n 3, model MB lost per wait median 800 min 500 max 1000' in w[-2], w
    assert w[-1].startswith('waits with keep-warm on: n 1, model MB lost per wait median 50'), w
    wb = warm(r20)
    assert len(wb) == 4 and 'pos 6 ship_wu (warm-up) 10.00 | 0.800 -> 0.820 | 1555 -> 2112' in wb[1], wb
    g = groups(r20, 1800)
    assert any(l.startswith('warm_ship warm-up (unscored): n 1') for l in g) and any(l.startswith('warm_ship: n 2 printed tok/s median 13.00') for l in g), g
    idt = identity(r20)
    assert any('r20: every 32-token run' in l and 'one text: yes' in l for l in idt), idt
    # r21: N models (three here), c2-io / c2-pcold lines, ABBA pairs, the side-by-side table.
    def row21(m, g, arm, i, tps, dec_mb, phase='on'):
        tag = f'{m}_{arm}.{i}'
        rb0, rb1 = 1_000_000, 1_000_000 + 500_000_000
        line = (f'c2-io run={tag} prompt_graphs=1 decode_graphs=31 us=0/6000000/9000000 read_bytes={rb0}/{rb1}/'
                f'{rb1 + int(dec_mb * 1e6)} rchar=0/0/0 majflt=0/900/{900 + int(dec_mb * 4)} minflt=0/0/0')
        c2 = [line] + ([f'c2-pcold tau=8 advice=cold tokens=156 layers=15 routed_slices=2754 marked_slices=729 mb=645.0 '
                        f'fails=0 us=500'] if arm == 'on' else [])
        return {'proto': 'r21', 'model': m, 'group': g, 'arm': arm, 'phase': phase, 'run_tag': tag, 'decode_tps': tps,
                'env_set': {'GGML_C2_IO': '1', **({'GGML_PREFILL_COLD': '8'} if arm == 'on' else {})}, 'repack_loaded': True,
                'text': 'T' + m, 'prompt_tokens': 155, 'c2': c2, 'res_before': rz(1500), 'res_after': rz(2000),
                'mem_before': 'avail=1MB cached=3000MB', 'libs': {'libllama.so': 'aa' * 32}}
    shaM = lambda k: {'file': k + '.gguf', 'bytes': 1, 'expected_bytes': 1, 'expected_sha256': 'h' + k,
                      'verified_marker': 'h' + k, 'sha256_end': 'h' + k, 'arch': k, 'template_source': 'model', 'repack_rule': True}
    r21 = {'kit': 'r21', 'models': {k: shaM(k) for k in ('granite', 'olmoe', 'lfm')},
           'build': {'libs': {'libllama.so': 'aa' * 32}}, 'runs': []}
    io_seq = {'granite': [(10, 900), (13, 500), (13, 520), (10, 880)] * 2, 'olmoe': [(11, 700), (12, 600), (12, 610), (11, 690)] * 2,
              'lfm': [(9, 100), (9, 100), (9, 100), (9, 100)] * 2}
    for b, m in enumerate(['granite', 'olmoe', 'lfm'], 1):
        r21['runs'].append(row21(m, f'b{b}', 'off', 90, 5, 999, 'warmup'))
        for i, (arm, (tps, mb)) in enumerate(zip(['off', 'on', 'on', 'off'] * 2, io_seq[m])):
            r21['runs'].append(row21(m, f'b{b}', arm, i, tps, mb))
    assert io_of(r21['runs'][1])['dec_read_mb'] == 900 and io_of(r21['runs'][1])['pre_read_mb'] == 500
    assert io_of(r21['runs'][1])['pre_ms'] == 6000 and io_of(r21['runs'][1])['dec_majflt'] == 3600
    assert pcold_of(r21['runs'][2])['mb'] == 645.0 and pcold_of(r21['runs'][1]) is None
    c = r21_checks(r21)
    assert all(l.endswith('OK') for l in c[:3]) and 'lib libllama.so: 1 hash(es) at the loads = build' in c[3], c
    assert any(l.startswith('granite: 9 runs, one text across both arms: yes; repack on in all: yes; skip/head2 set: none')
               for l in c), c
    t = r21_tables(r21)
    assert any(l.startswith('  granite off: n 4 tok/s median 10.00') and 'decode read MB 890' in l for l in t), t
    assert any(l.startswith('  granite on: n 4 tok/s median 13.00') and 'marked MB 645 advice cold' in l for l in t), t
    assert any(l.startswith('  b1 granite: tok/s 3.00 [3.00..3.00] n 4 | decode read MB -380.00 [-400.00..-360.00] n 4') for l in t), t
    assert any(l.startswith('  olmoe: tok/s 1.00') for l in t), t
    assert any(l.startswith('side by side') and l.endswith('| granite | olmoe | lfm |') for l in t), t
    assert '| tok/s on | 13.00 | 12.00 | 9.00 |' in t, t
    r21['runs'][3]['text'] = 'other'
    assert any('one text across both arms: NO (2 texts)' in l for l in r21_checks(r21))
    # r22: arms off / num in ABBA, hold per quad (first quad with hold in odd blocks), 0005's lines, the app's hold.
    nums = {'granite': {'GGML_EXPERT_SKIP': '130', 'GGML_HEAD2': '2048', 'GGML_HEAD2_Q': 'q4_0', 'VOLLIE_SKIP_STATS': '1'},
            'olmoe': {'VOLLIE_SKIP_MASS': '900', 'VOLLIE_HEAD2': '1024', 'VOLLIE_SKIP_STATS': '1'}}
    def row22(m, g, arm, hold, i, tps, res, cool, phase='on'):
        tag = f'{m}_{arm}{"_h" if hold else ""}.{i}'
        c2 = [f'c2-io run={tag} prompt_graphs=1 decode_graphs=31 us=0/5000000/6500000 read_bytes=-1/-1/-1 rchar=0/0/0 '
              f'majflt=0/100/150 minflt=0/0/0', f'c2-io-blk run={tag} inblock=0/1000000/1200000']
        if arm == 'num':
            c2 += [f'c2-skip-run: run={tag} sum=0 max=0 mass=900 c1_skip=0 c1_topp=0 c1_gap=0 prefill_tokens=156 '
                   f'prefill_pairs_skipped=13.0000% prefill_expert_bytes_skipped=13.0000% prefill_MB_per_token_selected=346.000 '
                   f'prefill_MB_per_token_skipped=45.000 prefill_bytes_unread=1.0000% decode_tokens=31 '
                   f'decode_pairs_skipped=13.4000% decode_expert_bytes_skipped=13.4000% decode_MB_per_token_selected=346.000 '
                   f'decode_MB_per_token_skipped=46.400 decode_bytes_unread=13.4000%',
                   f'c2-head-run: run={tag} c2_k=1024 c2_calls=32 c2_cols=32 c2_exact_rows_per_col=6936.0 c2_copies=1 c2_copy=1 '
                   f'c1_m=0 c1_calls=0 c1_copies=0 c1_copy=0']
        return {'proto': 'r22', 'model': m, 'group': g, 'arm': arm, 'hold': hold, 'phase': phase, 'run_tag': tag,
                'decode_tps': tps, 'env_set': {**ship, 'GGML_C2_IO': '1', **(nums[m] if arm == 'num' else {})},
                'repack_loaded': True, 'text': f'T{m}{arm}', 'c2': c2, 'res_before': rz(res), 'res_after': rz(2600),
                'cool_ms': cool, 'wait_lost_mb': 2600 - res, 'mem_before': 'avail=1MB cached=3000MB',
                'inblock_read_bytes': 1000, 'libs': {'libllama.so': 'aa' * 32},
                **({'hold_stats': {'advised_mb': 2600 - res, 'cpu_us': 20000, 'passes': 9, 'final_ms': 800, 'inblock': 100000,
                                   'advise_fails': 0}} if hold else {})}
    r22 = {'kit': 'r22', 'models': {k: {**shaM(k), 'num': nums[k]} for k in ('granite', 'olmoe')},
           'build': {'libs': {'libllama.so': 'aa' * 32}}, 'memlock_rlimit': {'soft': 65536, 'hard': 65536}, 'runs': []}
    for b, m in enumerate(['granite', 'olmoe', 'olmoe', 'granite'], 1):
        r22['runs'].append(row22(m, f'b{b}', 'off', False, 90, 5, 1000, 0, 'warmup'))
        for qi, hold in enumerate([b % 2 == 1, b % 2 == 0]):
            for j, arm in enumerate(['off', 'num', 'num', 'off']):
                # num +2 tok/s; hold +1 tok/s and 2,500 MB resident instead of 1,500 after a 60 s wait
                tps = 17 + (2 if arm == 'num' else 0) + (1 if hold else 0) + (0.2 if j == 3 else 0) + 0.1 * qi
                r22['runs'].append(row22(m, f'b{b}', arm, hold, 4 * qi + j, tps, 2500 if hold else 1500, 60000))
    assert blk_of(r22['runs'][1]) == {'pre_mb': 512.0, 'dec_mb': 102.4}
    assert skiprun_of(r22['runs'][2])['dec_bytes_pct'] == 13.4 and headrun_of(r22['runs'][2])['c2_copy']
    c = r22_checks(r22)
    assert any(l.startswith('granite: 18 runs; texts off: one text') and 'off != num' in l and 'missing in none, in off arm none'
               in l and 'num without an active head copy: none' in l for l in c), c
    assert any(l.startswith('I/O source: getrusage inblock in 36 rows (non-zero in 36)') for l in c), c
    t = r22_tables(r22)
    assert any(l.startswith('  granite num hold=on : n 4 tok/s 20.05') and 'skip bytes % 13.40' in l for l in t), t
    assert any(l.startswith('  olmoe: all +1.90 ± 0.00 (n 4) | hold on +1.90') for l in t), t
    assert any(l.startswith('  granite: all tok/s +1.00 ± 0.09 (n 4 / 4), res MB +1000 ± 0 (n 4 / 4) | waited tok/s +1.00') for l in t), t
    assert any(l.startswith('hold cost per gated run') and 'read back MB 100.00 [100.00..100.00] n 16' in l for l in t), t
    r22['runs'][3]['text'] = 'other'
    assert any('num: NO (2 texts)' in l for l in r22_checks(r22) if l.startswith('granite'))
    # r23: threads 3 / 4 per quad (4 first in odd blocks), hold in every gated run, OLMoE's load env, 0006's decline fields.
    loads = {'granite': {}, 'olmoe': {'VOLLIE_HEAD2_REPACK': '1'}}
    def row23(m, g, arm, i, tps, thr, phase='on', first=False):
        x = row22(m, g, arm, phase != 'warmup', i, tps, 2500, 60000 if phase != 'warmup' else 0, phase)
        x.update({'proto': 'r23', 'threads': thr, 'threads_loaded': [thr, thr], 'run_tag': f'{m}_{arm}_t{thr}.{i}', 'pos': i,
                  'gate': 0.6 if phase != 'warmup' else None, 'pinned_cpus': [4, 5, 6, 7], 'want_mb': 2600,
                  'thread_samples': [[900, thr, *([4, 5, 6, 7][:thr])]]})
        x['c2'] = [l.replace('c1_copy=0', 'c1_copy=0 c2_declined=0 c2_why=-').replace(f"run={m}_{arm}{'_h' if phase != 'warmup' else ''}.{i} ",
                   f"run={x['run_tag']} ") for l in x['c2']]
        if first:
            x.update({'reload_ms': 9000, 'load_env_set': {'GGML_C2_IO': '1', **loads[m]}})
        if phase != 'warmup':
            x['hold_stats'].update({'wait_passes': 1, 'final_passes': 2, 'back_permille': 970, 'mincore_ms': 400,
                                    'advised_distinct_mb': 90, 'adv_back_mb': 80, 'adv_before_mb': 90, 'stop': 'back'})
        return x
    r23 = {'kit': 'r23', 'kernel_release': '5.4.86-test', 'owner_note': 'close background apps',
           'models': {k: {**shaM(k), 'num': nums[k], 'load': loads[k]} for k in ('granite', 'olmoe')},
           'build': {'libs': {'libllama.so': 'aa' * 32}}, 'memlock_rlimit': {'soft': 65536, 'hard': 65536}, 'runs': []}
    for b, m in enumerate(['granite', 'olmoe', 'olmoe', 'granite'], 1):
        r23['runs'] += [row23(m, f'b{b}', 'off', 90 + k, 5, 3, 'warmup', k == 0) for k in range(2)]
        for qi, thr in enumerate([4, 3] if b % 2 else [3, 4]):
            for j, arm in enumerate(['off', 'num', 'num', 'off']):
                # num +2 tok/s, 4 threads +0.5 tok/s
                r23['runs'].append(row23(m, f'b{b}', arm, 4 * qi + j, 17 + (2 if arm == 'num' else 0) + (0.5 if thr == 4 else 0), thr))
        r23['runs'] += [row23(m, f'b{b}', a, 8 + k, 18, 3, 'prof') for k, a in enumerate(['off', 'num'] if b % 2 else ['num', 'off'])]
    assert headrun_of(r23['runs'][3])['c2_declined'] == 0 and headrun_of(r22['runs'][2])['c2_declined'] is None
    c = r23_checks(r23)
    assert any(l.startswith('olmoe (r23): load env') and "missing in none, other model's load env in none, in a run env none" in l
               and 'same text at 3 and 4 threads: yes; head declines: none; num without an active head copy: none; gated '
               'without hold: none; threads applied != spec: none' in l for l in c), c
    assert any(l.startswith("kernel release '5.4.86-test'") for l in c), c
    t = r23_tables(r23)
    assert any(l.startswith('  granite num t4: n 4 tok/s 19.50') and 'back % 97.0' in l for l in t), t
    assert any(l.startswith('  olmoe: all +2.00 ± 0.00 (n 4) | t3 +2.00 ± 0.00 (n 2) | t4 +2.00') for l in t), t
    assert any(l.startswith('  granite: off blocks +0.50 ± 0.00 (n 2), runs +0.50') for l in t), t
    assert any(l.startswith('  olmoe t4: samples 8, running 4.0, cpu 0-3 0% 4-6 75% 7 25%') for l in t), t
    assert any(l.startswith('hold cost per gated run') and 'back % 97.00 [97.00..97.00] n 40 (>= 95 % in 40 of 40)' in l for l in t), t
    r23['runs'][4]['c2'][-1] = r23['runs'][4]['c2'][-1].replace('c2_declined=0 c2_why=-', 'c2_declined=32 c2_why=type:CPU_KLEIDIAI')
    r23['runs'][0]['load_env_set'] = {}
    c = r23_checks(r23)
    assert any(l.startswith('granite (r23)') and 'head declines: granite_num_t4.2(32 type:CPU_KLEIDIAI)' in l for l in c), c
    assert any(l.startswith('granite (r23)') and 'missing in none' in l for l in c), c  # granite has no load env to miss
    r23['runs'][24]['load_env_set'] = {}
    assert any(l.startswith('olmoe (r23)') and 'missing in olmoe_off_t3.90' in l for l in r23_checks(r23))
    # r24: gate rule per quad (plateau first in odd blocks), 3 threads, charger flag, vmstat counters.
    def row24(m, g, arm, i, rule, phase='on', first=False):
        pl = rule == 'plateau'
        tps = 17 + (2 if arm == 'num' else 0) + (1 if pl else 0)  # num +2, plateau +1 tok/s
        x = row23(m, g, arm, i, tps, 3, phase, first)
        tag = f"{m}_{arm}_{'pl' if pl else 'r16'}.{i}"
        x['c2'] = [l.replace(f"run={x['run_tag']} ", f'run={tag} ') for l in x['c2']]
        x.update({'proto': 'r24', 'id': f"{m}_{arm}_{'pl' if pl else 'r16' if phase != 'warmup' else 'wu'}", 'run_tag': tag,
                  'gate_rule': rule, 'wait_end': None if phase == 'warmup' else 'plateau' if pl else 'timeout',
                  'wait_ms': 50000 if pl else 90000, 'heat': 0.73 if pl else 0.72, 'plugged': False, 'plugged_after': False,
                  'plugged_run': False, 'res_before': rz(2400 if pl else 2300), 'wait_lost_mb': 200 if pl else 300,
                  'mhz_before': 'mhz=1,1,1,1,2112,2112,2112,2035', 'cap_before': 'cap=2035/2035',
                  'vmstat_before': {'pswpin': 0, 'pgmajfault': 0}, 'vmstat_first': {'pswpin': 1000, 'pgmajfault': 1200},
                  'vmstat_after': {'pswpin': 1100, 'pgmajfault': 1400}})
        if phase == 'warmup':
            x['gate'] = None
        return x
    r24 = dict(r23, kit='r24', runs=[], idle_baseline={'fields': ['ms', 'headroom'], 'samples': [[0, 0.8], [5000, 0.79]]},
               thermal_zones=[{'zone': 'thermal_zone0', 'type': None, 'temp': None, 'error': 'type: denied'}])
    specs = {}
    for b, m in enumerate(['granite', 'olmoe', 'olmoe', 'granite'], 1):
        r24['runs'] += [row24(m, f'b{b}', 'off', 90 + k, 'none', 'warmup', k == 0) for k in range(2)]
        for qi, rule in enumerate(['plateau', 'r16'] if b % 2 else ['r16', 'plateau']):
            for j, arm in enumerate(['off', 'num', 'num', 'off']):
                r24['runs'].append(row24(m, f'b{b}', arm, 4 * qi + j, rule))
        r24['runs'] += [row24(m, f'b{b}', a, 8 + k, 'r16', 'prof') for k, a in enumerate(['off', 'num'] if b % 2 else ['num', 'off'])]
    for x in r24['runs']:
        specs[x['id']] = {'id': x['id'], 'gate_rule': x['gate_rule']}
    r24['specs'] = list(specs.values())
    r24['runs'][3]['plugged_after'] = r24['runs'][3]['plugged_run'] = True
    c = r24_checks(r24)
    assert any(l.startswith('r24: gate rule != spec in none') and 'FLAG) granite_num_pl.1' in l and 'readable 0 of 1 (type: denied)' in l
               for l in c), c
    t = r24_tables(r24)
    assert any(l.startswith('  granite num plateau: n 4 (flagged 1) tok/s 20.00') and 'pswpin prefill / decode 1000 / 100' in l for l in t), t
    assert any(l.startswith('  olmoe: all +2.00 ± 0.00 (n 4) | r16 +2.00 ± 0.00 (n 2) | plateau +2.00') and 'flag-free +2.00 ± 0.00 (n 4)' in l
               for l in t), t
    assert any(l.startswith('  granite: all') and 'flag-free +2.00 ± 0.00 (n 3)' in l for l in t), t
    assert any(l.startswith('  granite: tok/s +1.00 ± 0.00 (n 2)') and 'wait s -40.00 ± 0.00 (n 2)' in l and 'res before MB +100.00' in l
               for l in t), t
    assert any(l.startswith('runs with a charger at their start or end: 1') for l in t), t
    assert any(l.startswith('hold cost per gated run') for l in t), t
    # r25: r24 plus kv alternated per scored run (odd blocks: on off on off, off on off on; even blocks the reverse).
    # kv: TTFT -150 ms, pre-prompt majors 40,000 -> 500, pswpin to first chunk 50,000 -> 1,000, VmSwap at end 300 -> 100 MB.
    kvb = {'granite': [('kv', 17821696), ('rs', 58056704)], 'olmoe': [('kv', 285208576)]}
    def row25(m, g, arm, i, rule, kv=False, phase='on', first=False):
        x = row24(m, g, arm, i, rule, phase, first)
        x.update({'proto': 'r25', 'id': x['id'] + ('_kv' if kv else ''), 'ttft_ms': 5000.0 - (150 if kv else 0) + 10 * (i % 4),
                  'majflt': (500 if kv else 40000) + 150, 'mem_after': f"anon=400MB swap={100 if kv else 300}MB",
                  'vmstat_first': {'pswpin': 1000 if kv else 50000, 'pgmajfault': 1200},
                  'vmstat_after': {'pswpin': 1100 if kv else 50100, 'pgmajfault': 1400}})
        if kv:
            x['env'] = {**x['env_set'], 'VOLLIE_KV_DONTNEED': '1'}
            x['env_set'] = x['env']
        x['c2'] = x['c2'] + [f"c2-kvclear run={x['run_tag']} kind={k} on={int(kv)} bytes={n + 4096} released={n if kv else 0} "
                             f"why={'ok' if kv else 'off'} us={900 if kv else 30000}" for k, n in kvb[m]]
        return x
    r25 = dict(r24, kit='r25', runs=[], kv_rule='scored runs alternate VOLLIE_KV_DONTNEED=1')
    for b, m in enumerate(['granite', 'olmoe', 'olmoe', 'granite'], 1):
        r25['runs'] += [row25(m, f'b{b}', 'off', 90 + k, 'none', False, 'warmup', k == 0) for k in range(2)]
        for qi, rule in enumerate(['plateau', 'r16'] if b % 2 else ['r16', 'plateau']):
            for j, arm in enumerate(['off', 'num', 'num', 'off']):
                r25['runs'].append(row25(m, f'b{b}', arm, 4 * qi + j, rule, (j + qi + (0 if b % 2 else 1)) % 2 == 0))
        r25['runs'] += [row25(m, f'b{b}', a, 8 + k, 'r16', False, 'prof') for k, a in enumerate(['off', 'num'] if b % 2 else ['num', 'off'])]
    r25['specs'] = list({x['id']: {'id': x['id'], 'gate_rule': x['gate_rule']} for x in r25['runs']}.values())
    assert [kv_on(x) for x in r25['runs'][2:10]] == [True, False, True, False, False, True, False, True]
    assert pre_prompt_majflt(r25['runs'][2]) == 500 and released_mb(r25['runs'][2]) == (17821696 + 58056704) / 1e6
    assert dec_ms_tok(r25['runs'][2]) == 1500 / 31
    c = r25_checks(r25)
    assert any(l.startswith('r24: gate rule != spec in none') for l in c), c
    assert any(l.startswith('r25 kv factor: id suffix != env in none; kv outside scored runs none; rows without a c2-kvclear '
                            'line none') for l in c), c
    assert any(l.endswith('-> ok') and l.startswith('r25 VOID check') for l in c), c
    assert any(l.startswith('olmoe (r25) texts kv against plain:') and l.count(': same') == 2 for l in c), c
    t = r25_tables(r25)
    i = t.index(next(l for l in t if l.startswith('  olmoe: 8 kv and 8 plain scored runs, 8 pairs; unpaired cells none')))
    assert t[i + 1].startswith('    pre-prompt majflt: paired -39500 ± 0 (n 8)') and 'median 500 / 40000' in t[i + 1], t[i + 1]
    assert t[i + 2].startswith('    pswpin start->first chunk: paired -49000 ± 0 (n 8)'), t[i + 2]
    assert t[i + 3].startswith('    TTFT ms: paired -150.0 ± ') and 'Welch -150.0 ± ' in t[i + 3], t[i + 3]
    assert t[i + 5].startswith('    tok/s: paired +0.00 ± 0.00 (n 8)'), t[i + 5]
    assert t[i + 6].startswith('    VmSwap at end MB: paired -200 ± 0 (n 8)'), t[i + 6]
    assert t[i + 7].startswith('    released MB: paired +285.2 ± 0.0 (n 8)'), t[i + 7]
    assert t[i + 8].endswith('-> pass') and 'faults median kv < half of plain: yes' in t[i + 8], t[i + 8]
    assert t[i + 9].startswith('    gate r16: pairs 4 | TTFT ms -140.0 ± 11.5 (n 4) | pre-prompt majflt -39500'), t[i + 9]
    pooled = t[t.index("R63's gate and num comparisons, kv on and off pooled:"):t.index("R63's gate and num comparisons, plain runs only:")]
    plain = t[t.index("R63's gate and num comparisons, plain runs only:"):]
    assert any(l.startswith('  olmoe: all +2.00 ± 0.00 (n 4) | r16 +2.00') for l in pooled), pooled
    assert any(l.startswith('  olmoe: all +2.00 ± 0.00 (n 4) | r16 +2.00') for l in plain), plain
    assert any(l.startswith('  granite: tok/s +1.00 ± 0.00 (n 2)') for l in plain), plain
    assert any(l.startswith('  granite off plateau: n 2 (flagged 0)') for l in plain), plain
    assert any(l.startswith('hold cost per gated run') for l in pooled) and not any(l.startswith('hold cost') for l in plain)
    r25['runs'][3]['c2'][-1] = r25['runs'][3]['c2'][-1].replace('released=0', 'released=4096')  # a plain run releases
    r25['runs'][2]['c2'] = [l.replace('released=', 'released=0 x=') for l in r25['runs'][2]['c2']]  # a kv run releases 0
    c = r25_checks(r25)
    v = next(l for l in c if l.startswith('r25 VOID check'))
    assert v.endswith('-> VOID') and 'kv run: granite_off_pl.0;' in v and 'without kv: granite_num_pl.1 ' in v, v
    r25['runs'][2]['text'] = 'other'
    assert any(l.startswith('granite (r25) texts') and 'off kv' in l and 'DIFFER (kill rule)' in l for l in r25_checks(r25))
    # r26: one arm per model (granite num, olmoe off), r16 only, kv in ABBA quads (odd blocks on off off on, off on on off;
    # even blocks the reverse); kv lines carry tag=0xb4, plain lines tag=0x00 (0009). kv effects as in r25's synthetic rows.
    arm26 = {'granite': 'num', 'olmoe': 'off'}
    def row26(m, g, i, kv=False, phase='on', first=False):
        x = row25(m, g, arm26[m], i, 'r16' if phase != 'warmup' else 'none', kv, phase, first)
        x['proto'] = 'r26'
        x['c2'] = [l + (f" tag={'0xb4' if kv else '0x00'}" if l.startswith('c2-kvclear') else '') for l in x['c2']]
        return x
    r26 = dict(r24, kit='r26', runs=[], models={m: {'arm': a} for m, a in arm26.items()})
    for b, m in enumerate(['granite', 'olmoe', 'olmoe', 'granite'], 1):
        r26['runs'] += [row26(m, f'b{b}', 90 + k, False, 'warmup', k == 0) for k in range(2)]
        pat = [True, False, False, True, False, True, True, False]
        r26['runs'] += [row26(m, f'b{b}', j, kv if b % 2 else not kv) for j, kv in enumerate(pat)]
        r26['runs'] += [row26(m, f'b{b}', 8 + k, False, 'prof') for k in range(2)]
    r26['specs'] = list({x['id']: {'id': x['id'], 'gate_rule': x['gate_rule']} for x in r26['runs']}.values())
    assert kvtags_of(r26['runs'][2]) == [('kv', 'ok', '0xb4'), ('rs', 'ok', '0xb4')]
    assert kvtags_of(r25['runs'][4])[0][2] is None  # r25 lines without tag=
    c = r26_checks(r26)
    assert any(l.startswith("r26: arms per model {'granite': ['num'], 'olmoe': ['off']}, arm != report in none; gate rules of "
                            "gated runs ['r16']; kv id suffix != env in none; kv outside scored runs none; rows without a "
                            "c2-kvclear line none; lines without tag= in none") for l in c), c
    assert any(l.startswith('r26 void check: released > 0 in a run without kv: none -> ok; kv runs releasing 0 (mechanism '
                            'miss): none') for l in c), c
    assert any(l.startswith('olmoe (r26) texts kv against plain: off kv') and l.endswith(': same') for l in c), c
    t = r26_tables(r26)
    i = t.index(next(l for l in t if l.startswith('  granite: 8 kv and 8 plain scored runs, 4 ABBA quads, 0 scored runs outside')))
    assert t[i + 1].startswith('    pre-prompt majflt: paired -39500 ± 0 (n 4)') and 'median 500 / 40000' in t[i + 1], t[i + 1]
    assert t[i + 3].startswith('    TTFT ms: paired -150.0 ± 0.0 (n 4)'), t[i + 3]  # the +10 * (i % 4) drift cancels in ABBA
    assert t[i + 7].startswith('    released MB: paired +75.9 ± 0.0 (n 4)'), t[i + 7]
    assert t[i + 8].endswith('-> pass') and 'pre-prompt majflt mean -39500: lower' in t[i + 8], t[i + 8]
    mech = next(l for l in t if l.startswith('  olmoe: why=ok'))
    assert mech.startswith('  olmoe: why=ok in every line 8 of 8 kv runs (not: none)') and "tag kv {('kv', '0xb4'): 8} plain "\
        "{('kv', '0x00'): 8}" in mech and 'released MB per kv run median 285.2' in mech, mech
    assert any(l.startswith('  granite num kv: n 8 (flagged 0) tok/s 19.00') for l in t), t
    assert any(l.startswith('  olmoe off plain: n 8 (flagged 0) tok/s 17.00') for l in t), t
    assert any(l.startswith('hold cost per gated run') for l in t), t
    # a kv run whose clear falls back (why=map, tag kept): mechanism and release lines name it; a plain run releasing: void
    r26['runs'][2]['c2'] = [l.replace('released=17821696 why=ok', 'released=0 why=map') for l in r26['runs'][2]['c2']]
    r26['runs'][3]['c2'][-1] = r26['runs'][3]['c2'][-1].replace('released=0', 'released=4096')
    t = r26_tables(r26)
    mech = next(l for l in t if l.startswith('  granite: why=ok'))
    assert '7 of 8 kv runs (not: granite_num_r16.0)' in mech and "('kv', 'map'): 1" in mech, mech
    v = next(l for l in r26_checks(r26) if l.startswith('r26 void check'))
    assert 'without kv: granite_num_r16.1 -> VOID' in v, v
    r26['runs'][4]['c2'] = [l.split(' tag=')[0] if l.startswith('c2-kvclear') else l for l in r26['runs'][4]['c2']]
    assert any('lines without tag= in granite_num_r16.2' in l for l in r26_checks(r26))
    # r25 report with tag= lines (kit r25b): the mechanism lines follow r25's tables; without tag= they do not
    assert not any(l.startswith('kv mechanism per model') for l in r25_tables(r25))
    r25b = dict(r25, runs=[dict(x, c2=[l + ' tag=0xb4' if l.startswith('c2-kvclear') else l for l in x['c2']]) for x in r25['runs']])
    assert any(l.startswith('kv mechanism per model') for l in r25_tables(r25b))
    # the plateau gate, as the app's unit test (R62's pos 4 and 6 series, every 5 s)
    p4 = [0.873, 0.873, 0.873, 0.873, 0.802, 0.788, 0.755, 0.741, 0.721, 0.735, 0.725, 0.735, 0.748, 0.764, 0.77]
    p6 = [0.725, 0.767, 0.789, 0.81, 0.833, 0.833, 0.808, 0.778, 0.753, 0.741, 0.717, 0.737, 0.753, 0.776, 0.79]
    assert plateau_end([[5000 * i, h] for i, h in enumerate(p4)]) == (50000, 'plateau', 0.725)
    assert plateau_end([[5000 * i, h] for i, h in enumerate(p6)]) == (60000, 'plateau', 0.753)
    assert plateau_end([[0, 0.7], [5000, 0.64]]) == (5000, 'gate', 0.64)
    print('selftest OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
