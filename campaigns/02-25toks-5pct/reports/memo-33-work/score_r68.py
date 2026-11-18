#!/usr/bin/env python3
"""R68 scoring (kit r28). Standalone: python3 -I score_r68.py <report.json> [more.json] | --selftest | --write-synthetic DIR

Scores what prereg/R68-rehearsal-touch-nice-references-device.md §3 predicts and §4 scores, per session and per model,
with the readings fixed in prereg/R68-amendment-2.md. The item IDs in the output (A1 ... E) are those of
memo-33-work/ebony-r68-audit.md, adopted there; the lead's three choices are A2, B4 (with the mirror control) and C1
(strict: both policies, before and after the run).
Inputs: kit r28 reports (c2 APK: session A = deep-sleeping list on, B = off; c1 / stock APK: session R) and T-A cap-probe
reports, in any order. Numbers printed from a phone report are *device*; the bands beside them are the pre-registration's
*estimated* predictions. --selftest numbers are synthetic and mean nothing.
Whatever the amendment leaves open is printed as "AMBIGUOUS: <item>" and scored with the most literal reading.
The field layout follows tools/c2_research_parse.py (r28_runs, r28_checks, r28_tables); nothing is imported from the repo.
"""
import base64, gzip, hashlib, json, os, re, statistics as st, sys

# --- constants ------------------------------------------------------------------------------------------------------
# D3: libraries = memo-32 §5's full values; model files = the full values behind the pre-registration's short forms
# (the kit's expected_sha256, as in reports/device/r27_2026-11-18.json).
MODEL_SHA = {'granite': '8df94de731a08a377150e4fcc5dcbd16212a6d102bc0b628cdc905cba0bc0f45',
             'olmoe': '5abf710b775cd64fd89d74a783c8a7056a409ba795cd401b9dd448ef8e444abb'}
LIB_SHA = {'c2': {'cpu': '6f1d859a962f473c7f7b5d1547a2530157a3ca98bbe046524bce3d90b4e3d28e',
                  'llama': '126bfb4f04ec7fff1135cf6e7c1c65e191556aad695c52f718e0e3a645d9ede8'},
           'c1': {'cpu': '53bfee45511d14177fa4165d897a1d44b51310c51bdc83aef8f543e8e7d9d11c',
                  'llama': 'f2f4f88af78d601ae1ba14dc895a06aa23d9320661a55271a8fa5cf3c9c23861'},
           'stock': {'cpu': '604e0f69c12b5972dd2ba46f1b24fd462f6cf9e4fefe6041afcff45b4d68cd0d',
                     'llama': 'de2ccec771c869b15f237e66112830fe43c2391eae807522ee58c0620c45cb8c'}}
P4_MHZ, P7_MHZ = 2419, 2841    # C1 (strict): uncapped = policy 4 >= 2,419 and policy 7 >= 2,841, before and after the run
WRITTEN_MHZ = 2419             # the pre-registration's written reading: cpu 7 max >= 2,419 (cap_before); listed beside
PROBE_CAPPED_MHZ = 2150        # E: power saving on predicts max7 <= 2,150
LONG_WAIT_MS = 60000           # A3
RQ_MS = 100                    # B4
PAGE = 4096                    # A6: set x page size against want_mb (bytes / 10^6)
N_SCORED = {'c2': 62, 'c1': 20, 'stock': 20}   # C6: a complete report
NUM_ENV = {'VOLLIE_SKIP_MASS': '900', 'VOLLIE_HEAD2': '1024'}   # C5: b5's num arm
BAND_MEDIAN = {('olmoe', True): (23.0, 25.4), ('olmoe', False): (21.0, 23.0),      # (model, uncapped): band
               ('granite', True): (22.5, 24.5), ('granite', False): (20.0, 22.0)}
BAND_TOUCH_TPS = {'olmoe': (-0.5, 1.5), 'granite': (-0.5, 1.0)}
BAND_NICE_TPS = {'olmoe': (-0.5, 2.0), 'granite': (-0.5, 1.5)}
BAND_TOUCH_MISSING_PCT = (-90.0, -30.0)
BAND_B5_B1 = (0.4, 1.6)
BAND_BELOW25_B1 = (4, 7)
BAND_REF = {('c1', 'granite'): (19, 25), ('c1', 'olmoe'): (12, 22), ('stock', 'granite'): (8, 18), ('stock', 'olmoe'): (5, 18)}

TOUCH_RE = re.compile(r'c2-touch run=(\S+) pass=(\d+) final=(\d) src=(\w+) at_ms=(\d+) set=(\d+) pages=(\d+) ms=([\d.]+) '
                      r'cpu=(-?\d+) majflt=(-?\d+) minflt=(-?\d+)')
NICE_RE = re.compile(r'c2-nice run=(\S+) want=(\S+) pool=(\d) restored=(\d+)/(\d+)')
NICEW_RE = re.compile(r' w(\d+)=(-?\d+):(-?\d+):s(\d+):f(\d+):e(\d+)')
SCHED_RE = re.compile(r'c2-sched run=(\S+) phase=decode n=(\d+)')
SCHEDW_RE = re.compile(r' w(\d+)=(-?\d+):(-?\d+):(-?\d+):(-?\d+)')
POOL_RE = re.compile(r'c2-pool run=(\S+) pool=(\d)')
POOLW_RE = re.compile(r' w(\d+)=(-?\d+):0x')
IO_RE = re.compile(r'c2-io run=(\S+) .*? majflt=(-?\d+)/(-?\d+)/(-?\d+)')
SKIP_RE = re.compile(r'c2-skip-run: run=(\S+) .*?decode_pairs_skipped=([\d.]+)%')

AMB = []


def amb(item):
    if item not in AMB:
        AMB.append(item)
    return f'AMBIGUOUS: {item}'


def load(path):
    with open(path, encoding='utf-8') as f:
        s = f.read().strip()
    if s.startswith('c2gz:'):
        s = gzip.decompress(base64.b64decode(s[5:])).decode('utf-8')
    return json.loads(s)


# --- per-run readers (the parser's definitions) -----------------------------------------------------------------------
def env(x):
    return x.get('env') or x.get('env_set') or {}


def touch_on(x):
    return 'VOLLIE_TOUCH' in env(x)


def nice_on(x):
    return 'VOLLIE_NICE' in env(x)


def cell(x):
    return ('T' if touch_on(x) else 't') + ('N' if nice_on(x) else 'n')


def power(p):
    p = p or {}
    lp = p.get('lowPower')
    return 'on' if p.get('powerSave') is True or (isinstance(lp, int) and lp > 0) else 'off' if p.get('powerSave') is False else '?'


def caps(x, k):
    m = re.search(r'cap=(\d+)/(\d+)', x.get(k) or '')
    return (int(m.group(1)), int(m.group(2))) if m else None


def uncapped(x):
    """C1 (lead: strict): cap_before and cap_after both read policy 4 >= 2,419 and policy 7 >= 2,841."""
    return all(c is not None and c[0] >= P4_MHZ and c[1] >= P7_MHZ for c in (caps(x, 'cap_before'), caps(x, 'cap_after')))


def uncapped_written(x):
    c = caps(x, 'cap_before')
    return c is not None and c[1] >= WRITTEN_MHZ


def tagged(x, rx):
    return [(m, l) for l in x.get('c2') or [] for m in [rx.search(l)] if m and m.group(1) == x.get('run_tag', m.group(1))]


def touch_final(x):
    """The run's last final c2-touch pass: {src, set, pages, ms, majflt, minflt}, or None."""
    f = [m for m, _ in tagged(x, TOUCH_RE) if m.group(3) == '1']
    if not f:
        return None
    m = f[-1]
    return {'src': m.group(4), 'set': int(m.group(6)), 'pages': int(m.group(7)), 'ms': float(m.group(8)),
            'majflt': int(m.group(10)), 'minflt': int(m.group(11))}


def nice_line(x):
    """The run's last c2-nice line: {want, restored_failed, workers [(tid, value read back, sets, fails, errno)]} or None."""
    t = tagged(x, NICE_RE)
    if not t:
        return None
    m, l = t[-1]
    return {'want': m.group(2), 'restored_failed': int(m.group(5)),
            'workers': [(int(w.group(2)), int(w.group(3)), int(w.group(4)), int(w.group(5)), int(w.group(6))) for w in NICEW_RE.finditer(l[m.end():])]}


def pool_tids(x):
    t = tagged(x, POOL_RE)
    tids = {int(w.group(2)) for m, l in t for w in POOLW_RE.finditer(l[m.end():])}
    n = nice_line(x)
    return tids | ({w[0] for w in n['workers']} if n else set())


def after_nice(x):
    """workers_after as {tid: nice} (rows: tid, nice, priority, policy, cpu, cgroup)."""
    return {th[0]: th[1] for th in x.get('workers_after') or [] if isinstance(th, (list, tuple)) and len(th) > 1}


def rq(x):
    """B4: run-queue wait of the decode, ms: sum over workers of wq_us / 1000 (c2-sched phase=decode)."""
    t = tagged(x, SCHED_RE)
    if not t:
        return None
    m, l = t[-1]
    return sum(int(w.group(3)) for w in SCHEDW_RE.finditer(l[m.end():])) / 1000


def dec_majflt(x):
    t = tagged(x, IO_RE)
    if not t:
        return None
    a, b = int(t[-1][0].group(3)), int(t[-1][0].group(4))
    return None if a < 0 or b < 0 else b - a


def memv(s, k):
    m = re.search(k + r'=(\d+)MB', s or '')
    return int(m.group(1)) if m else None


def smaps_mb(x):
    """A1: the model's smaps Rss by inode, kB -> MB of 10^6 bytes (want_mb's unit)."""
    d = x.get('smaps_before')
    return d['model'][1] * 1024 / 1e6 if isinstance(d, dict) and d.get('model') else None


def missing_resident(x):   # memo-31's "missing": reported beside, not scored (A1)
    w, r = x.get('want_mb'), x.get('res_before')
    r = r.get('total_mb') if isinstance(r, dict) else None
    return None if w is None or r is None else max(0, w - r)


def md5(t):
    return hashlib.md5(t.encode()).hexdigest()[:8]


# --- statistics --------------------------------------------------------------------------------------------------------
def med(v):
    v = [a for a in v if a is not None]
    return st.median(v) if v else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def band(v, lo_hi):
    lo, hi = lo_hi
    return 'n/a' if v is None else f"{'inside' if lo <= v <= hi else 'MISS'} ({lo:g} to {hi:g})"


def paired(quads, on, f):
    """A5: per quad mean(on runs) - mean(off runs); (mean over quads, SE = sd / sqrt(quads) or None, quads used)."""
    d = []
    for q in quads:
        a, b = [f(x) for x in q if on(x)], [f(x) for x in q if not on(x)]
        if a and b and None not in a + b:
            d.append(st.mean(a) - st.mean(b))
    if not d:
        return None, None, 0
    return st.mean(d), (st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else None), len(d)


def pm(e, nd=2):
    return '-' if e[0] is None else f"{e[0]:+.{nd}f} ± {'-' if e[1] is None else f'{e[1]:.{nd}f}'} (n {e[2]} quads)"


def beyond(e):
    """-1 / +1: negative / positive beyond 2 x SE; 0 inside; None when there is no SE."""
    if e[0] is None or e[1] is None:
        return None
    return -1 if e[0] < -2 * e[1] else 1 if e[0] > 2 * e[1] else 0


def gkey(g):
    return int(str(g).lstrip('b') or 0)


def short(h):
    return f'{h[:8]}…{h[-8:]}' if isinstance(h, str) and len(h) > 16 else repr(h)


# --- one kit report ----------------------------------------------------------------------------------------------------
class Session:
    def __init__(self, r, path):
        self.r, self.path = r, path
        self.rt = r.get('runtime') or (r.get('build') or {}).get('runtime') or 'c2'
        lab = r.get('session_label')
        self.name = f'R-{self.rt}' if self.rt != 'c2' else 'A' if lab == 'deep-sleeping list on' else 'B' if lab == 'deep-sleeping list off' else '?'
        self.runs = [x for x in r.get('runs', []) if x.get('proto') == 'r28']
        self.scored = [x for x in self.runs if x.get('phase') == 'on']
        self.models = list(dict.fromkeys(x.get('model') for x in self.runs))
        self.complete = r.get('kit') == 'r28' and not r.get('stopped_early') and len(self.scored) == N_SCORED.get(self.rt)
        # D1: power saving on -> not scored; '?' is scored only if the session reads off at start and end. D2 / R63 §4: plugged.
        soff = power(r.get('power_start')) == 'off' and power(r.get('power_end')) == 'off'
        self.dropped = {}
        for x in self.scored:
            p = power(x.get('power'))
            if p == 'on' or (p == '?' and not soff):
                self.dropped[id(x)] = 'power saving on' if p == 'on' else 'power state unread'
            elif x.get('plugged_run') or x.get('plugged'):
                self.dropped[id(x)] = 'charger'
        for g in self.groups('quad'):           # D2: a quad with any invalid run is dropped whole
            for q in self.raw_quads(g):
                if any(id(x) in self.dropped for x in q):
                    for x in q:
                        self.dropped.setdefault(id(x), 'in a dropped quad')
        self.void_blocks = {g for g in self.groups('claim', 'ref') if len(self.valid(g)) < 8}
        self.res = {}                           # per-model verdicts, read by the cross-session part

    def groups(self, *kinds):
        return sorted({x.get('group') for x in self.scored if x.get('block') in kinds}, key=gkey)

    def block(self, g):
        return [x for x in self.scored if x.get('group') == g]

    def valid(self, g):
        return [x for x in self.block(g) if id(x) not in self.dropped]

    def raw_quads(self, g):
        b = self.block(g)
        return [b[i:i + 4] for i in range(0, len(b) - 3, 4)]

    def quads(self, m):
        """The model's whole valid quads, in run order (D2)."""
        return [q for g in self.groups('quad') if self.block(g)[0].get('model') == m for q in self.raw_quads(g)
                if not any(id(x) in self.dropped for x in q)]

    def claim_stats(self, g):
        b, v = self.block(g), self.valid(g)
        tps = [x.get('decode_tps') for x in v if x.get('decode_tps') is not None]
        return {'g': g, 'id': b[0].get('id'), 'model': b[0].get('model'), 'arm': b[0].get('arm'), 'n': len(b), 'valid': len(v),
                'median': med(tps), 'below': sum(1 for t in tps if t < 25.0), 'uncapped': sum(1 for x in v if uncapped(x)),
                'written': sum(1 for x in v if uncapped_written(x)), 'void': g in self.void_blocks, 'runs': v,
                'disagree': [f"{x.get('run_tag')} ({(x.get('cap_before') or '-').replace('cap=', '')} -> {(x.get('cap_after') or '-').replace('cap=', '')})"
                             for x in v if uncapped(x) != uncapped_written(x)],
                'states': sorted({f"{(x.get('cap_before') or '-').replace('cap=', '')}->{(x.get('cap_after') or '-').replace('cap=', '')}" for x in v})}


def void_checks(s, out):
    r = s.r
    out.append('  void checks (§4, kit not lever; D1 to D3):')
    for m in s.models:
        want = MODEL_SHA.get(m)
        if want is None:
            out.append('    ' + amb(f'model {m!r} has no hash in the pre-registration; not checked'))
            continue
        got = ((r.get('models') or {}).get(m) or {}).get('sha256_end')
        copy = (((r.get('storage') or {}).get('models') or {}).get(m) or {}).get('sha256')
        out.append(f"    D3 model {m}: models.sha256_end {short(got)}: {'ok' if got == want else 'VOID (model hash differs or is missing)'}; "
                   f"storage.models.sha256 {short(copy)}: {'ok' if copy == want else 'VOID (internal copy fails its sha256 or has none)'}")
    want = LIB_SHA.get(s.rt)
    libsets = [('build', (r.get('build') or {}).get('libs') or {})] + [(x.get('run_tag'), x['libs']) for x in s.runs if x.get('libs')]
    lib_bad, seen = [], 0
    for where, libs in libsets:
        for name, h in libs.items():
            k = 'cpu' if name.startswith('libggml-cpu') else 'llama' if name == 'libllama.so' else None
            if k and want:
                seen += 1
                if h != want[k]:
                    lib_bad.append(f'{where}:{name}={short(h)}')
    out.append(f"    D3 libraries (runtime {s.rt}, memo-32 §5's full values): {seen} hashes in build.libs + {len(libsets) - 1} load rows: "
               f"{'none recorded: VOID cannot be excluded' if not seen else 'ok' if not lib_bad else 'VOID (library hash differs): ' + ', '.join(lib_bad)}")
    pw = [x for x in s.runs if power(x.get('power')) == 'on']
    unk = [x for x in s.runs if power(x.get('power')) == '?']
    out.append(f"    D1 power saving: at start {power(r.get('power_start'))}, at end {power(r.get('power_end'))}, session_power "
               f"{r.get('session_power')!r}; reads on in {len(pw)} of {len(s.runs)} runs ({', '.join(str(x.get('run_tag')) for x in pw) or 'none'}); "
               f"unread ('?') in {len(unk)} (scored only when start and end both read off)")
    out.append(f"    prediction 'power saving reads off in every run' (P 0.95): "
               f"{'hit' if not pw and not unk and power(r.get('power_start')) == 'off' and power(r.get('power_end')) == 'off' else 'MISS'}")
    why = {}
    for x in s.scored:
        if id(x) in s.dropped:
            why.setdefault(s.dropped[id(x)], []).append(str(x.get('run_tag')))
    out.append(f"    D2 scored runs left out of every statistic: {sum(len(v) for v in why.values())} of {len(s.scored)}"
               + ''.join(f"; {k}: {', '.join(v)}" for k, v in why.items()))
    for g in sorted(s.void_blocks, key=gkey):
        b = s.block(g)
        out.append(f"    D2 VOID block {g} ({b[0].get('id')}): {len(s.valid(g))} valid runs of {len(b)} (fewer than 8)")
    if (r.get('storage') or {}).get('used') != 'internal':
        out.append(f"    storage.used = {(r.get('storage') or {}).get('used')!r}: not the internal copies the pre-registration names")


def identity(s, out):
    """D4: per model and arm one md5 over all rows, warm-ups included; rows without a text are listed."""
    out.append('  identity of the greedy text (§4, D4): md5 of text per model and arm, all r28 rows:')
    ok = {}
    for m in s.models:
        for arm in sorted({str(x.get('arm')) for x in s.runs if x.get('model') == m}):
            rs = [x for x in s.runs if x.get('model') == m and str(x.get('arm')) == arm]
            none = [str(x.get('run_tag')) for x in rs if x.get('text') is None]
            by = {}
            for x in rs:
                if x.get('text') is not None:
                    by.setdefault(md5(x['text']), []).append(x)
            ok[(m, arm)] = len(by) == 1
            tail = f"; rows without a text: {', '.join(none)}" if none else ''
            if len(by) <= 1:
                out.append(f"    {m} {arm}: {len(rs)} rows, {sorted(by)}: {'same' if by else 'NO TEXT'}{tail}")
                continue
            if any(x.get('block') == 'claim' and touch_on(x) for x in rs):
                knobs = 'num, touch and nice (inside b5)'
            else:
                sep = []
                for name, on in (('touch', touch_on), ('nice', nice_on)):
                    lv = [{on(x) for x in v} for v in by.values()]
                    if all(len(l) == 1 for l in lv) and len(set().union(*lv)) == 2:
                        sep.append(name)
                knobs = ' + '.join(sep) if sep else 'touch and nice (no single level separates the texts)'
            out.append(f"    {m} {arm}: {len(rs)} rows, { {k: sorted({cell(x) for x in v}) for k, v in sorted(by.items())} }: DIFFER -> kills "
                       f"(suspends) the knob(s) involved until explained: {knobs}{tail}")
    return ok


def claim_blocks(s, out):
    out.append('  claim-format blocks (descriptive; D2: valid runs only; C1 strict: uncapped = policy 4 >= 2419 and policy 7 >= 2841 in '
               'cap_before and cap_after):')
    stats = {}
    for g in s.groups('claim', 'ref'):
        c = s.claim_stats(g)
        stats[g] = c
        cap_n = c['valid'] - c['uncapped']
        out.append(f"    {g} {c['model']} {c['arm']} ({c['id']}): valid {c['valid']} of {c['n']}, median {fmt(c['median'])} tok/s, below 25.0: "
                   f"{c['below']}, uncapped {c['uncapped']} of {c['valid']} (written reading, cpu 7 before >= {WRITTEN_MHZ}: {c['written']}); "
                   f"C2c block is {'capped' if cap_n >= 2 else 'not capped'} ({cap_n} capped runs); cap states {c['states']}"
                   f"{' VOID (<8 valid)' if c['void'] else ''}")
        if c['disagree']:
            out.append(f"      C1 runs where the strict and the written reading disagree: {', '.join(c['disagree'])}")
        if c['valid'] != 10 and (c['uncapped'] >= 9 or cap_n >= 9):
            out.append('      ' + amb(f"C3 chooses the band by '9 of 10' runs; {g} has {c['valid']} valid runs; the counts of valid runs are used"))
        c['state'] = 'uncapped' if c['uncapped'] >= 9 else 'capped' if cap_n >= 9 else 'mixed'
        if s.rt == 'c2' and g in ('b1', 'b2'):
            if c['state'] == 'mixed':
                out.append(f"      C3 level band: mixed block, neither band scored (uncapped band {band(c['median'], BAND_MEDIAN[(c['model'], True)])}; "
                           f"capped band {band(c['median'], BAND_MEDIAN[(c['model'], False)])})")
            else:
                out.append(f"      C3 prediction, median {c['state']}: {band(c['median'], BAND_MEDIAN[(c['model'], c['state'] == 'uncapped')])}")
        if s.rt == 'c2' and g == 'b1':
            out.append(f"      prediction, runs below 25.0 in b1: {c['below']}: {band(c['below'], BAND_BELOW25_B1)}")
            out.append(f"      C2a prediction, uncapped in at least 9 of 10 runs of b1 (P 0.75): {'hit' if c['uncapped'] >= 9 else 'MISS'} "
                       f"({c['uncapped']}; written reading {c['written']})")
        if s.rt != 'c2':
            bd = BAND_REF.get((s.rt, c['model']))
            out.append(f"      reference block {s.rt} {c['model']}: C3 prediction (written for uncapped): "
                       f"{band(c['median'], bd) if bd and c['state'] == 'uncapped' else 'not scored (' + c['state'] + ' block)' if bd else 'no band'}"
                       f"{'' if not bd or c['state'] == 'uncapped' else '; band read anyway: ' + band(c['median'], bd)}")
    if s.rt == 'c2' and 'b1' in stats and 'b5' in stats:
        a, b = stats['b5']['median'], stats['b1']['median']
        d = None if a is None or b is None else a - b
        out.append(f"    C5 b5 median minus b1 median (OLMoE num + touch + nice; position confounded: b5 last, b1 first): {fmt(a)} - {fmt(b)} = "
                   f"{'-' if d is None else f'{d:+.2f}'} tok/s: {band(d, BAND_B5_B1)}")
    return stats


def touch(s, m, out):
    qs = s.quads(m)
    runs = [x for q in qs for x in q]
    if len(qs) < 3:
        out.append(f"    VERDICT touch, {m}, session {s.name}: VOID factor (D2: {len(qs)} whole quads, fewer than 3)")
        return 'void'
    e = paired(qs, touch_on, lambda x: x.get('decode_tps'))
    b = beyond(e)
    out.append(f"    A5 touch (T - t), tok/s paired per quad: {pm(e)}: prediction {band(e[0], BAND_TOUCH_TPS.get(m, (0, 0)))}; "
               f"{'no SE' if b is None else 'negative beyond 2 x SE' if b < 0 else 'not negative beyond 2 x SE'}")
    # A6: per-run end-pass check, every gated T run of the model (quad T cells and b5)
    tr = [x for x in s.runs if x.get('model') == m and x.get('gate') and touch_on(x)]
    bad6 = []
    for x in tr:
        f, w = touch_final(x), x.get('want_mb')
        if not f:
            bad6.append(f"{x.get('run_tag')}(no final pass)")
        elif f['src'] != 'want' or not f['set'] or f['pages'] / f['set'] < 0.90 or not w or abs(f['set'] * PAGE / 1e6 - w) > 0.01 * w:
            bad6.append(f"{x.get('run_tag')}(src={f['src']} pages/set={f['pages'] / f['set'] if f['set'] else 0:.3f} "
                        f"set={f['set'] * PAGE / 1e6:.0f}MB want={w})")
    s.res.setdefault('a6_bad', {})[m] = bad6
    out.append(f"    A6 end pass (src=want, pages / set >= 0.90, set x 4096 within 1 % of want_mb) in every gated T run: {len(tr) - len(bad6)} of "
               f"{len(tr)}: prediction {'hit' if tr and not bad6 else 'MISS' if tr else 'n/a'}{'; failing: ' + ', '.join(bad6) if bad6 else ''}")
    # A1: one source per session and model
    use_smaps = bool(runs) and all(smaps_mb(x) for x in runs)
    if use_smaps:
        miss = lambda x: None if x.get('want_mb') is None else max(0, x['want_mb'] - smaps_mb(x))
    else:
        miss = lambda x: None if x.get('want_mb') is None or memv(x.get('mem_before'), 'file') is None else max(0, x['want_mb'] - memv(x.get('mem_before'), 'file'))
        out.append('    ' + amb("A1 fallback in use: RssFile is read as printed in mem_before (file=<n>MB); its MB unit is not stated, and it "
                                "counts the libraries as well"))
    out.append(f"    A1 source of 'mapped' for {m}: {'smaps by inode (> 0 in every scored quad run)' if use_smaps else 'RssFile for every run (smaps by inode reads 0 or is absent in some quad run)'}")
    # A3 / A4
    lw = lambda x: (x.get('wait_ms') or 0) >= LONG_WAIT_MS
    T, t = [x for x in runs if touch_on(x)], [x for x in runs if not touch_on(x)]
    pairs = sum(1 for q in qs if any(lw(x) and touch_on(x) for x in q) and any(lw(x) and not touch_on(x) for x in q))
    out.append(f"    A3 waits >= 60 s (wait_ms): T {sum(map(lw, T))} of {len(T)} runs, t {sum(map(lw, t))} of {len(t)}; A4 quads holding a T and a t "
               f"run over 60 s: {pairs} of {len(qs)}")
    res = {}
    for name, f in (('want - mapped, scored', miss), ('want - resident, reported only', missing_resident)):
        a, c = med(f(x) for x in T if lw(x)), med(f(x) for x in t if lw(x))
        eff = None if a is None or not c else 1 - a / c
        res[name] = (a, c, eff)
        out.append(f"    A1/A4 missing MB at run start ({name}): median T {fmt(a, 0)} / t {fmt(c, 0)}: effect "
                   f"{'-' if eff is None else f'{eff:.2f}'} (= {'-' if eff is None else f'{-100 * eff:+.0f} %'})"
                   + (f"; prediction {band(None if eff is None else -100 * eff, BAND_TOUCH_MISSING_PCT)}; §3 mechanism 'lower by at least half': "
                      f"{'-' if eff is None else 'hit' if eff >= 0.5 else 'MISS'}" if name.endswith('scored') else ''))
    a, c, eff = res['want - mapped, scored']
    testable = pairs >= 3 and bool(c)
    # A2: unscored reads of "did the wait cost less"
    lost = paired(qs, touch_on, lambda x: x.get('wait_lost_mb'))
    out.append(f"    A2 unscored: wait_lost_mb (before the hold) T - t {pm(lost, 0)}; TTFT ms T - t {pm(paired(qs, touch_on, lambda x: x.get('ttft_ms')), 0)}; "
               f"decode majflt T - t {pm(paired(qs, touch_on, dec_majflt), 0)}; final pass medians: ms "
               f"{fmt(med((touch_final(x) or {}).get('ms') for x in T), 1)}, majflt {fmt(med((touch_final(x) or {}).get('majflt') for x in T), 0)}, "
               f"minflt {fmt(med((touch_final(x) or {}).get('minflt') for x in T), 0)}")
    if b is None:
        v, key = 'not scoreable (no SE)', 'void'
    elif not testable:
        why = f'{pairs} quads with a T and a t run over 60 s, fewer than 3' if pairs < 3 else 'the t median of missing MB is 0 or absent'
        out.append(f"    A4/A7 loss endpoint: not testable this session ({why})")
        v, key = ('KILL (tok/s negative beyond 2 x SE; loss not testable)', 'kill') if b < 0 else \
            ("loss not testable, speed not negative (neither pass nor kill; §7's 'it stays on' does not follow)", 'untestable')
    elif eff >= 0.30 and b >= 0:
        v, key = 'PASS (missing MB lower by at least 30 % with T, tok/s not negative beyond 2 x SE)', 'pass'
    else:
        v, key = 'KILL (' + ' and '.join((['missing MB not lower by 30 %'] if eff < 0.30 else []) + (['tok/s negative beyond 2 x SE'] if b < 0 else [])) + ')', 'kill'
    out.append(f"    VERDICT touch, {m}, session {s.name}: {v}")
    out.append("      A2 (lead): the loss rule is read after touch's own final pass, so it is a mechanism check of the final pass.")
    if lost[0] is None:
        out.append('      ' + amb("A2: wait_lost_mb is absent, so 'not higher with T' cannot be read; the 'stays on' condition is left open"))
    else:
        out.append(f"      A2 §7 'it stays on' needs: loss rule pass ({'yes' if key == 'pass' else 'no'}), tok/s condition of §4 "
                   f"({'yes' if b is not None and b >= 0 else 'no'}), wait_lost_mb not higher with T ({'yes' if lost[0] <= 0 else 'NO'}): "
                   f"{'holds' if key == 'pass' and lost[0] <= 0 else 'does not hold'} in this session")
        out.append('      ' + amb("A2 'wait_lost_mb not higher with T' has no stated statistic; read as the paired per-quad mean (T - t) <= 0, "
                                "without an SE margin"))
    return key


def nice_permission(s, out):
    """B1 / B2: session level, every quad N run of both models. Returns 'permitted', 'partly permitted', 'not permitted' or 'no N runs'."""
    nr = [x for m in s.models for q in s.quads(m) for x in q if nice_on(x)]
    full, some, errs, pattern, missing_after = 0, 0, {}, {}, 0
    for x in nr:
        n = nice_line(x)
        ws = n['workers'] if n else []
        aft = after_nice(x)
        st_ = []
        for i, w in enumerate(ws):
            good = w[1] == -10 and w[3] == 0 and aft.get(w[0], -10) == -10
            missing_after += w[0] not in aft
            st_.append(good)
            if w[4]:
                errs[w[4]] = errs.get(w[4], 0) + 1
            pattern.setdefault(f'w{i}', [0, 0])[0 if good else 1] += 1
        full += bool(st_) and all(st_)
        some += any(w[1] == -10 for w in ws)
    state = 'no N runs' if not nr else 'permitted' if full == len(nr) else 'not permitted' if some == 0 else 'partly permitted'
    out.append(f"  B1/B2 nice permission (session, every quad N run of both models): N runs {len(nr)}; every pool thread at -10 with f0 (and -10 in "
               f"workers_after) in {full}; any thread at -10 in {some}; per thread ok / not ok {dict(sorted((k, tuple(v)) for k, v in pattern.items()))}; "
               f"errnos {errs or 'none'} (13 = EACCES): {state}; prediction 'the set succeeds' (P 0.5): "
               f"{'hit' if state == 'permitted' else 'MISS' if nr else 'n/a'}")
    if missing_after and state == 'permitted':
        out.append('  ' + amb(f"B1 wants workers_after to show -10 'for the same tids'; {missing_after} pool tids of N runs are not listed in "
                              f"workers_after; a tid that is not listed is not counted against the permission"))
    return state


def nice(s, m, perm, out):
    qs0 = s.quads(m)
    # B3: drop a quad whose n run failed a restore or still reads -10 on a pool tid
    qs, b3 = [], []
    for q in qs0:
        bad = [x.get('run_tag') for x in q if not nice_on(x) and (
            ((nice_line(x) or {}).get('restored_failed') or 0) > 0 or any(after_nice(x).get(t) == -10 for t in pool_tids(x)))]
        (b3 if bad else qs).append(bad if bad else q)
    out.append(f"    B3 quads dropped from the nice factor (restore failed or an n run's pool thread still at -10): {len(b3)} of {len(qs0)}"
               f"{' (' + ', '.join(str(t) for b in b3 for t in b) + ')' if b3 else ''}")
    if b3 and len(qs) < 3:
        out.append('    ' + amb("D2's 'fewer than 3 whole quads voids the factor' is written for invalid runs; applied to quads dropped by B3 too"))
    if len(qs) < 3:
        out.append(f"    VERDICT nice, {m}, session {s.name}: VOID factor ({len(qs)} whole quads, fewer than 3)")
        return 'void'
    e = paired(qs, nice_on, lambda x: x.get('decode_tps'))
    b = beyond(e)
    out.append(f"    B5 nice (N - n), tok/s paired per quad: {pm(e)}: prediction (if permitted) {band(e[0], BAND_NICE_TPS.get(m, (0, 0)))}")
    # B4 (lead): rq per decode, pairs within the same touch level, >= 3 qualifying pairs, median ratio <= 0.5, mirror control
    pr = []
    for q in qs:
        for lv in (True, False):
            off, on = [x for x in q if touch_on(x) == lv and not nice_on(x)], [x for x in q if touch_on(x) == lv and nice_on(x)]
            if len(off) == 1 and len(on) == 1 and None not in (rq(off[0]), rq(on[0])):
                pr.append((rq(off[0]), rq(on[0])))
    ratio = lambda a, c: c / a if a else float('inf')
    sel = [ratio(a, c) for a, c in pr if a > RQ_MS]            # selected on the n run: N / n
    mir = [ratio(c, a) for a, c in pr if c > RQ_MS]            # mirror, selected on the N run: n / N
    r1, r2 = med(sel), med(mir)
    route = len(sel) >= 3 and r1 <= 0.5
    mirror = r2 is not None and r2 <= 0.5
    out.append(f"    B4 run-queue route (rq = sum over workers of wq_us / 1000 per decode; same-touch pairs: {len(pr)}): pairs with rq(n) > {RQ_MS} ms: "
               f"{len(sel)} (needs 3), median rq(N) / rq(n) {fmt(r1)} (needs <= 0.5): {'met' if route else 'not met'}; mirror control, pairs with "
               f"rq(N) > {RQ_MS} ms: {len(mir)}, median rq(n) / rq(N) {fmt(r2)}: {'ALSO <= 0.5, the route does not pass' if mirror else 'not <= 0.5'}; "
               f"runs above {RQ_MS} ms: n {sum(1 for a, _ in pr if a > RQ_MS)}, N {sum(1 for _, c in pr if c > RQ_MS)}")
    if mirror and len(mir) < 3:
        out.append('    ' + amb(f"B4's mirror control has no minimum pair count; it reads <= 0.5 on {len(mir)} pair(s) and is applied as written"))
    if perm == 'not permitted':
        v, key = 'VOID, not permitted (the set fails on the phone); §7: ADPF is priced again', 'void'
    elif perm == 'partly permitted':
        v, key = 'partly permitted: no pass or kill; the tok/s effect above is reported with the per-thread pattern; §7 treats it as void (ADPF priced again)', 'void'
    elif perm != 'permitted':
        v, key = 'n/a (no N runs)', 'void'
    elif b is None:
        v, key = 'not scoreable (no SE)', 'void'
    elif b < 0:
        if route and not mirror:
            out.append('    ' + amb("nice meets the run-queue pass route and the kill rule (tok/s negative beyond 2 x SE) at once; §4 gives no "
                                    "order; scored as the kill, with the route reported"))
        v, key = 'KILL (tok/s negative beyond 2 x SE)', 'kill'
    elif b > 0:
        v, key = 'PASS (tok/s positive beyond 2 x SE)', 'pass'
    elif route and not mirror:
        v, key = 'PASS (run-queue wait lower by at least half where the off cell waited above 100 ms; mirror control clear)', 'pass'
    else:
        v, key = 'permitted, speed not shown' + (' (run-queue route met, but the mirror control also reads <= 0.5)' if route else ''), 'not shown'
    out.append(f"    VERDICT nice, {m}, session {s.name}: {v}")
    return key


def score_session(s):
    r = s.r
    out = [f"== session {s.name} ({os.path.basename(s.path)}): kit {r.get('kit')}, runtime {s.rt}, label {r.get('session_label')!r}, "
           f"started {r.get('started')}, r28 runs {len(s.runs)} (scored {len(s.scored)}) =="]
    void_checks(s, out)
    s.stats = claim_blocks(s, out)
    if s.rt != 'c2':
        t = {}
        for x in s.runs:
            if x.get('text') is not None:
                t.setdefault(x.get('model'), set()).add(md5(x['text']))
        out.append(f"  texts per model (reported; the identity rule covers c2 cells): { {m: sorted(v) for m, v in t.items()} }")
        return out
    s.res['identity'] = identity(s, out)
    s.res['perm'] = nice_permission(s, out)
    s.res['touch'], s.res['nice'] = {}, {}
    for m in s.models:
        if not any(x.get('block') == 'quad' and x.get('model') == m for x in s.scored):
            continue
        out.append(f"  {m}: factors (D2: {len(s.quads(m))} whole valid quads)")
        s.res['touch'][m] = touch(s, m, out)
        s.res['nice'][m] = nice(s, m, s.res['perm'], out)
    return out


def combine(v):
    """A8: a kill in either session is a kill; a pass needs a pass in every session where the lever is testable; else mixed."""
    if 'kill' in v:
        return 'KILL'
    return 'PASS' if v and all(x == 'pass' for x in v if x in ('pass', 'kill')) and 'pass' in v and not any(x == 'not shown' for x in v) else 'mixed'


def cross(A, B, out):
    ss = [s for s in (A, B) if s]
    out.append(f"== levers over the c2 sessions scored ({', '.join(s.name for s in ss)}) (A8) ==")
    if len(ss) < 2:
        out.append('  one c2 session supplied: A8 needs both; the lines below are provisional')
    for lever in ('touch', 'nice'):
        for m in dict.fromkeys(m for s in ss for m in s.res[lever]):
            per = {s.name: s.res[lever].get(m) for s in ss}
            out.append(f"  VERDICT {lever}, {m}, both sessions: {combine([v for v in per.values() if v])} (per session {per})")
    out.append('  ' + amb("A8 is listed under touch in amendment 2; its wording ('a lever') is applied to nice as well"))
    out.append('== go rule (memo-31 §4, R68 §4; C2, C4, C5) ==')
    if not B:
        out.append('  no scoreable session B (deep-sleeping list off): the go rule cannot be read')
        return
    touch_killed = [s.name for s in ss if s.res['touch'].get('olmoe') == 'kill']
    met = {}
    for g in ('b1', 'b5', 'b2'):
        c = B.stats.get(g)
        if not c:
            continue
        okb = c['valid'] == 10 and c['uncapped'] == 10 and c['median'] is not None and c['median'] >= 25.7 and c['below'] <= 2
        line = (f"  {g} {c['id']} ({c['model']} {c['arm']}): B: valid {c['valid']} (10: {'yes' if c['valid'] == 10 else 'no'}), uncapped "
                f"{c['uncapped']} (C2b 10 of 10: {'yes' if c['uncapped'] == 10 and c['valid'] == 10 else 'no'}), median {fmt(c['median'], 3)} "
                f"(>= 25.7: {'yes' if c['median'] is not None and c['median'] >= 25.7 else 'no'}), below 25.0: {c['below']} (at most 2: "
                f"{'yes' if c['below'] <= 2 else 'no'})")
        a = A.stats.get(g) if A else None
        oka = None
        if a:
            oka = a['median'] is not None and a['median'] >= 25.0 and not a['void']
            line += f"; A: median {fmt(a['median'], 3)} (>= 25.0: {'yes' if oka else 'no'}; valid {a['valid']}, uncapped {a['uncapped']})"
            if a['valid'] != 10 and oka:
                out.append('  ' + amb(f"C4 asks session A only for 'a median >= 25.0'; {g} has {a['valid']} valid runs in A; scored on the valid "
                                      f"runs (a void block, fewer than 8, does not meet it)"))
        else:
            line += '; A: not supplied'
        out.append(line)
        inel = []
        if g == 'b5':                                           # C4 / C5 eligibility
            runs = [x for s in ss for x in s.stats.get('b5', {}).get('runs', [])]
            if any(not s.res['identity'].get((c['model'], str(c['arm'])), False) for s in ss):
                inel.append('its texts are not one (D4)')
            if touch_killed:
                inel.append(f"touch is killed on OLMoE in session {', '.join(touch_killed)}")
            if any(not smaps_mb(x) for x in runs):
                inel.append('smaps by inode reads 0 or is absent in its runs (memory state not readable)')
            if any(s.res['perm'] == 'partly permitted' for s in ss):
                inel.append('nice is partly permitted (C5)')
            a6 = [t for s in ss for t in s.res.get('a6_bad', {}).get('olmoe', []) if '_claim' in t]
            if a6:
                inel.append(f"touch fails A6 in b5 runs ({', '.join(a6)})")
            env_bad = [str(x.get('run_tag')) for x in runs if any((x.get('env_set') or env(x)).get(k) != v for k, v in NUM_ENV.items())]
            skip_bad = [str(x.get('run_tag')) for x in runs if not any(float(mm.group(2)) > 0 for mm, _ in tagged(x, SKIP_RE))]
            out.append(f"    C5 b5 ran its num arm: VOLLIE_SKIP_MASS=900 and VOLLIE_HEAD2=1024 in env_set in {len(runs) - len(env_bad)} of {len(runs)} "
                       f"valid runs; decode skip counter above 0 in {len(runs) - len(skip_bad)}"
                       f"{'' if not (env_bad or skip_bad) else '; failing: ' + ', '.join(sorted(set(env_bad + skip_bad)))}")
            if env_bad or skip_bad:
                inel.append('the num arm did not run in every run')
                out.append('    ' + amb("C5 says to check that b5 ran its num arm and names no consequence; a failing run makes b5 not eligible "
                                        "(R63 §4 would void it)"))
            if any(s.res['perm'] == 'not permitted' for s in ss):
                out.append("    C5: nice not permitted, so b5 is 'num + touch (nice asked, refused)', still eligible")
            out.append(f"    C4/C5 b5 eligible for the go rule: {'yes' if not inel else 'NO: ' + '; '.join(inel)}")
        if g == 'b2':
            out.append(f"    Granite b2 is not a claim candidate under C4 (b1 or b5); reported per model: session B part {'met' if okb else 'not met'}")
            continue
        met[g] = 'no' if (not okb or oka is False or inel) else 'pending A' if oka is None else 'yes'
        out.append(f"    go rule, {g}: {'MET' if met[g] == 'yes' else 'session B part met, session A pending' if met[g] == 'pending A' else 'MISSED'}"
                   f"{' (not eligible)' if inel and okb and oka is not False else ''}")
    if 'yes' in met.values():
        g = 'b1' if met.get('b1') == 'yes' else 'b5'
        out.append(f"  VERDICT go rule: MET by {g} ({B.stats[g]['id']}{', exactly as run' if g == 'b5' else '; b1 is the claim configuration, no budget spent'}): "
                   f"opens a claim round (R69, its own pre-registration)")
    elif 'pending A' in met.values():
        out.append(f"  VERDICT go rule: session B part met by {', '.join(g for g, v in met.items() if v == 'pending A')}; session A pending")
    else:
        out.append('  VERDICT go rule: MISSED; reported with these numbers')
    claimed = [g for g, v in met.items() if v != 'no'] or []
    capped = [f"{s.name} {g} ({s.stats[g]['valid'] - s.stats[g]['uncapped']} capped runs)" for s in ss for g in dict.fromkeys(['b1'] + claimed)
              if g in s.stats and s.stats[g]['valid'] - s.stats[g]['uncapped'] >= 2]
    out.append(f"  C2c third bullet (b1 or the claimed block capped, 2 or more capped runs, in either session): "
               + (f"FIRES for {', '.join(capped)}: \"25 is reached only in a clock state we do not control\", with these numbers, and the research "
                  f"goes to the cap's cause" if capped else 'does not fire'))
    if len(ss) == 2:                                           # D4: the same md5 in A and B
        t = {}
        for s in ss:
            for x in s.runs:
                if x.get('text') is not None:
                    t.setdefault((x.get('model'), str(x.get('arm'))), {}).setdefault(md5(x['text']), set()).add(s.name)
        for k, v in sorted(t.items()):
            out.append(f"  D4 texts across A and B, {k[0]} {k[1]}: { {h: sorted(n) for h, n in sorted(v.items())} }: {'same' if len(v) == 1 else 'DIFFER'}")


def cap_probes(probes, out):
    out.append("== T-A cap probes (E; P 0.65 that the cap follows power saving; 'most' = more than half of all samples; off predicts max7 >= 2841 "
               "by C1's strict reading) ==")
    first = {}
    for path, p in sorted(probes, key=lambda q: str(q[1].get('started'))):
        lab = 'on' if 'on' in str(p.get('label')).split()[-1:] else 'off' if 'off' in str(p.get('label')).split()[-1:] else power(p.get('power_start'))
        f = p.get('fields') or []
        i, j = (f.index('max7') if 'max7' in f else 3), (f.index('spin_load') if 'spin_load' in f else 15)
        smp = p.get('samples') or []
        hit = lambda v: v is not None and (v <= PROBE_CAPPED_MHZ if lab == 'on' else v >= P7_MHZ)
        share = lambda rows: sum(1 for q in rows if hit(q[i])) / len(rows) if rows else None
        whole = share(smp)
        scored = len(smp) == 360 and lab not in first
        if scored:
            first[lab] = whole is not None and whole > 0.5
        out.append(f"  {os.path.basename(path)} (started {p.get('started')}): power saving {lab} ({p.get('label')!r}, {p.get('session_label')!r}), samples "
                   f"{len(smp)}; share of samples as predicted ({'max7 <= 2150' if lab == 'on' else 'max7 >= 2841'}): {fmt(whole)} (no load "
                   f"{fmt(share([q for q in smp if not q[j]]))}, load {fmt(share([q for q in smp if q[j]]))}); written reading (>= 2419) share "
                   f"{fmt(sum(1 for q in smp if q[i] and q[i] >= WRITTEN_MHZ) / len(smp) if smp else None)}; max7 values {sorted({q[i] for q in smp})}: "
                   f"{'scored' if scored else 'reported, not scored (' + ('not 360 samples' if len(smp) != 360 else 'not the first complete probe of its state') + ')'}")
    if 'on' in first and 'off' in first:
        out.append(f"  prediction 'cap follows power saving': {'hit' if first['on'] and first['off'] else 'MISS'} (on {first['on']}, off {first['off']})")
    else:
        out.append('  prediction not scoreable: it needs a complete probe with power saving on and one with it off')


def run(paths):
    AMB.clear()
    out, sessions, probes = [], [], []
    for p in paths:
        r = load(p)
        if r.get('kind') == 'cap_probe':
            probes.append((p, r))
        else:
            sessions.append(Session(r, p))
    # C6: the first complete report of each kind is scored; a stopped or repeated one is reported, not scored
    sessions.sort(key=lambda s: str(s.r.get('started')))
    pick = {}
    for s in sessions:
        ok = s.complete and s.name != '?'
        if ok and s.name == 'B' and not ('2026-11-18T18' <= str(s.r.get('started')) < '2026-11-19T12'):
            out.append(amb(f"C6 wants session B 'started on the night of 2026-11-18'; {os.path.basename(s.path)} started {s.r.get('started')}; "
                           f"it is scored as B and the date is for the lead to accept"))
        if ok and s.name not in pick:
            pick[s.name] = s
            out += score_session(s)
        else:
            why = 'label unset or unknown' if s.name == '?' else f"not complete (kit {s.r.get('kit')}, stopped_early {s.r.get('stopped_early')}, scored runs " \
                f"{len(s.scored)} of {N_SCORED.get(s.rt)})" if not s.complete else f'a repeat: an earlier complete session {s.name} is scored'
            out.append(f"== {os.path.basename(s.path)}: session {s.name}, started {s.r.get('started')}: REPORTED, NOT SCORED (C6): {why} ==")
    if sum(1 for s in sessions if s.name in pick and pick[s.name] is not s and s.complete) > 0:
        out.append(amb("C6 'the first such report': read as the earliest by the report's 'started' time, not by argument order"))
    if pick.get('A') or pick.get('B'):
        cross(pick.get('A'), pick.get('B'), out)
    if probes:
        cap_probes(probes, out)
    out.append(f'== still open after amendment 2 (met in this scoring): {len(AMB)} ==')
    out += [f'  {i + 1}. {a}' for i, a in enumerate(AMB)]
    return out


# --- synthetic reports (r28 field layout, after the parser's selftest rows) -------------------------------------------
ROWS = [[1, 2, 4, 3], [2, 3, 1, 4], [3, 4, 2, 1], [4, 1, 3, 2]]           # Williams rows; cells below as (touch, nice)
CELLS = {1: (True, False), 2: (False, True), 3: (True, True), 4: (False, False)}


def synth(rt='c2', list_on=False, base=None, nice_ok=True, touch_tps=0.4, nice_tps=0.8, cap='cap=2419/2841', started='2026-11-18T23:10:00'):
    """A report with the r28 layout. Synthetic effects: touch +touch_tps tok/s and fewer missing MB, nice +nice_tps and less rq wait."""
    base = base or {'olmoe': 24.6, 'granite': 23.4}
    libs = {'libggml-cpu-android_armv8.2_2.so': LIB_SHA[rt]['cpu'], 'libllama.so': LIB_SHA[rt]['llama'], 'libggml.so': 'e' * 64}
    runs = []

    def row(m, g, i, arm, block, tc, nc, phase='on', jit=0.0):
        sfx = '_wu' if phase == 'warmup' else '_claim' if block in ('claim', 'ref') else '_q' + ('_touch' if tc else '') + ('_nice' if nc else '')
        tag = f'{m}_{arm}{sfx}.{i}'
        e = {'VOLLIE_POOL': '1', 'VOLLIE_EXEC_MAP': '1', 'VOLLIE_KV_DONTNEED': '1', 'VOLLIE_SCHED': '1'} if rt == 'c2' else {}
        e.update(NUM_ENV if (m, arm) == ('olmoe', 'num') else {})
        e.update({'VOLLIE_TOUCH': '1'} if tc else {})
        e.update({'VOLLIE_NICE': '-10'} if nc else {})
        tps = base[m] + (0.9 if (m, arm) == ('olmoe', 'num') else 0) + (touch_tps if tc else 0) + (nice_tps if nc and nice_ok else 0) + jit
        gate = phase == 'on'
        long_ = gate and i % 4 != 3
        at10 = nc and nice_ok
        x = {'proto': 'r28', 'id': f'{m}_{arm}{sfx}', 'run_tag': tag, 'model': m, 'group': g, 'phase': phase, 'arm': arm, 'block': block,
             'env': e, 'env_set': e, 'gate': 0.65 if gate else None, 'gate_rule': 'r16' if gate else 'none', 'decode_tps': tps,
             'wait_ms': 90500 if long_ else 4, 'wait_end': 'timeout' if long_ else 'gate', 'idle_ms': 95000, 'ttft_ms': 4000.0 - (300 if tc else 0),
             'want_mb': 2700, 'wait_lost_mb': (300 if tc else 700) if long_ else 0, 'res_before': {'total_mb': 2650 if tc else 2500},
             'cap_before': cap, 'cap_after': cap, 'power': {'powerSave': False, 'lowPower': 0, 'status': 0}, 'plugged_run': False,
             'mem_before': f"avail=2638MB cached=2447MB swapfree=0MB anon=480MB file={2560 if tc else 2150}MB swap=437MB",
             'smaps_before': {'model': [3000000, (2441406 if tc else 2050781) if m == 'olmoe' else 0, 0, 0]},   # 2500 / 2100 MB of 10^6
             'workers_after': [[100 + k, -10 if at10 else 0, 110 if at10 else 120, 0, 4 + k, '0::/'] for k in range(3)],
             'text': 'The sea' if (m, arm) == ('olmoe', 'num') else 'The ocean ' + m,
             'c2': [f'c2-io run={tag} prompt_graphs=1 decode_graphs=32 us=1/2/3 read_bytes=-1/-1/-1 rchar=-1/-1/-1 majflt=100/200/{230 if tc else 290} minflt=1/2/3',
                    f'c2-pool run={tag} pool=1 mask=0xf0 threads=3 graphs=1/32 pool_graphs=1/32 ' + ' '.join(f'w{k}=10{k}:0xf0:s1:e0:c{4 + k}=33' for k in range(3)),
                    f'c2-sched run={tag} phase=prefill n=1 w0=1:2:3:4 w1=1:2:3:4 w2=1:2:3:4',
                    f'c2-sched run={tag} phase=decode n=32 w0=1000:{60000 if at10 else 150000}:3:4 w1=1000:0:3:4 w2=1000:0:3:4']}
        if (m, arm) == ('olmoe', 'num'):
            x['c2'].append(f'c2-skip-run: run={tag} sum=0 max=0 mass=900 prefill_tokens=177 decode_tokens=32 decode_pairs_skipped=31.5000% '
                           f'decode_expert_bytes_skipped=31.5000% decode_MB_per_token_selected=243.3 decode_MB_per_token_skipped=96.4 decode_bytes_unread=31.5%')
        if tc:
            n = 2700 * 10 ** 6 // PAGE
            x['c2'].append(f'c2-touch run={tag} pass=1 final=0 src=want at_ms=12000 set={n} pages={n} ms=700.0 cpu=1 majflt=0 minflt=10 sum=1')
            x['c2'].append(f'c2-touch run={tag} pass=2 final=1 src=want at_ms=90000 set={n} pages={n - 5000} ms=800.0 cpu=2 majflt=5 minflt=3000 sum=1')
        if nc:
            x['c2'].append(f'c2-nice run={tag} want=-10 pool=1 restored=0/0 ' + ' '.join(
                f'w{k}=10{k}:-10:s1:f0:e0' if nice_ok else f'w{k}=10{k}:0:s0:f1:e13' for k in range(3)))
        if i == 90:
            x['libs'] = dict(libs)
        return x

    if rt == 'c2':
        plan = [('olmoe', 'claim', 'off'), ('granite', 'claim', 'num'), ('olmoe', 'quad', 'off'), ('granite', 'quad', 'num'), ('olmoe', 'claim', 'num')]
    else:
        plan = [('granite', 'ref', rt), ('olmoe', 'ref', rt)]
    for b, (m, blk, arm) in enumerate(plan, 1):
        mi = ['granite', 'olmoe'].index(m)
        last = rt == 'c2' and b == 5
        runs += [row(m, f'b{b}', 90 + j, arm, blk, False, False, 'warmup') for j in range(2)]
        if blk != 'quad':
            runs += [row(m, f'b{b}', j, arm, blk, last, last, jit=0.05 * ((j * 7) % 5 - 2)) for j in range(10)]   # median = level
        else:
            for k in range(4):
                runs += [row(m, f'b{b}', 4 * k + j, arm, blk, *CELLS[c], jit=0.125 * k) for j, c in enumerate(ROWS[(k + 2 * mi) % 4])]
    return {'kit': 'r28', 'app': '1.28.0+28', 'runtime': rt, 'started': started, 'stopped_early': False,
            'session_label': f"deep-sleeping list {'on' if list_on else 'off'}", 'deep_sleeping_list': list_on,
            'power_start': {'powerSave': False, 'lowPower': 0}, 'power_end': {'powerSave': False, 'lowPower': 0},
            'session_power': 'power-save off', 'build': {'runtime': rt, 'libs': {k: v for k, v in libs.items() if k != 'libggml.so'}},
            'models': {m: {'expected_sha256': h, 'sha256_end': h} for m, h in MODEL_SHA.items()},
            'storage': {'used': 'internal', 'models': {m: {'sha256': h, 'action': 'present'} for m, h in MODEL_SHA.items()}}, 'runs': runs}


def synth_probe(on, started='2026-11-18T22:00:00'):
    return {'kind': 'cap_probe', 'kit': 'r28', 'started': started, 'label': f"power-save {'on' if on else 'off'}",
            'session_label': 'deep-sleeping list off', 'power_start': {'powerSave': on, 'lowPower': int(on)},
            'fields': ['ms', 'max0', 'max4', 'max7', 'cur0', 'cur4', 'cur7', 'headroom', 'thermal_status', 'power_save', 'low_power',
                       'battery_pct', 'battery_c', 'plugged', 'spinner', 'spin_load'],
            'samples': [[1000 * i, 1804, 1766 if on else 2419, (2035 if on else 2841) if i % 10 else (2841 if on else 2150), 0, 0, 0, None, 0, on,
                         int(on), 80, 27.0, False, int((i // 60) % 2 == 0), int(i >= 180)] for i in range(360)]}


def write_synthetic(d):
    os.makedirs(d, exist_ok=True)
    files = {'synthetic-r28-B.json': synth(), 'synthetic-r28-A.json': synth(list_on=True, started='2026-11-19T15:00:00'),
             'synthetic-r28-c1.json': synth('c1', base={'olmoe': 17.0, 'granite': 22.5}, started='2026-11-20T10:00:00'),
             'synthetic-r28-stock.json': synth('stock', base={'olmoe': 10.0, 'granite': 13.0}, started='2026-11-20T11:00:00'),
             'synthetic-capprobe-on.json': synth_probe(True), 'synthetic-capprobe-off.json': synth_probe(False, '2026-11-18T22:10:00')}
    for n, r in files.items():
        with open(os.path.join(d, n), 'w', encoding='utf-8') as f:
            json.dump(r, f)
    return [os.path.join(d, n) for n in files]


def selftest():
    import tempfile

    def score(*reports):
        with tempfile.TemporaryDirectory() as d:
            ps = []
            for i, r in enumerate(reports):
                ps.append(os.path.join(d, f'{i}.json'))
                with open(ps[-1], 'w', encoding='utf-8') as f:
                    json.dump(r, f)
            return run(ps)

    def has(t, *parts):
        assert any(all(p in l for p in parts) for l in t), (parts, '\n'.join(t))

    def quad_runs(r, g):
        return [x for x in r['runs'] if x['group'] == g and x['phase'] == 'on']

    A = lambda **k: synth(list_on=True, started='2026-11-19T15:00:00', **k)
    # 1. B alone. OLMoE 24.6: b1 median 24.6, 10 below 25.0; b5 = 24.6 + 0.9 + 0.4 + 0.8 = 26.7
    t = score(synth())
    has(t, 'b1 olmoe off (olmoe_off_claim): valid 10 of 10, median 24.60 tok/s, below 25.0: 10, uncapped 10 of 10', 'not capped')
    has(t, 'C3 prediction, median uncapped: inside (23 to 25.4)')
    has(t, 'runs below 25.0 in b1: 10: MISS (4 to 7)')
    has(t, 'C5 b5 median minus b1 median', '= +2.10 tok/s: MISS (0.4 to 1.6)')
    has(t, 'A5 touch (T - t), tok/s paired per quad: +0.40 ± 0.00 (n 4 quads)')
    has(t, 'B5 nice (N - n), tok/s paired per quad: +0.80 ± 0.00 (n 4 quads)')
    has(t, 'A6 end pass', '18 of 18: prediction hit')
    has(t, "A1 source of 'mapped' for olmoe: smaps by inode")
    has(t, "A1 source of 'mapped' for granite: RssFile for every run")
    has(t, 'missing MB at run start (want - mapped, scored): median T 200 / t 600: effect 0.67 (= -67 %)', 'inside', 'hit')   # olmoe, unit-corrected
    has(t, 'missing MB at run start (want - mapped, scored): median T 140 / t 550: effect 0.75')                                  # granite, RssFile
    has(t, 'missing MB at run start (want - resident, reported only): median T 50 / t 200: effect 0.75')
    has(t, 'A3 waits >= 60 s (wait_ms): T 6 of 8 runs, t 6 of 8; A4 quads holding a T and a t run over 60 s: 4 of 4')
    has(t, 'A2 unscored: wait_lost_mb (before the hold) T - t -300 ± 144 (n 4 quads)')
    has(t, 'VERDICT touch, olmoe, session B: PASS')
    has(t, "A2 §7 'it stays on' needs: loss rule pass (yes), tok/s condition of §4 (yes), wait_lost_mb not higher with T (yes): holds")
    has(t, 'B1/B2 nice permission', 'N runs 16', ': permitted;')
    has(t, 'B4 run-queue route', 'pairs with rq(n) > 100 ms: 8 (needs 3), median rq(N) / rq(n) 0.40 (needs <= 0.5): met; mirror control, pairs with rq(N) > 100 ms: 0')
    has(t, 'VERDICT nice, olmoe, session B: PASS (tok/s positive beyond 2 x SE)')
    has(t, 'olmoe off: 30 rows', ': same')
    has(t, 'go rule, b1: MISSED')
    has(t, 'C4/C5 b5 eligible for the go rule: yes')
    has(t, 'VERDICT go rule: session B part met by b5; session A pending')
    has(t, 'C2c third bullet', 'does not fire')
    assert not any('VOID' in l for l in t), [l for l in t if 'VOID' in l]
    # 2. A + B + references + probes, given out of order: C6 sorts them; b5 meets the rule in both sessions
    t = score(synth('stock', base={'olmoe': 3.3, 'granite': 13.0}, started='2026-11-20T11:00:00'), A(base={'olmoe': 24.3, 'granite': 23.4}),
              synth('c1', base={'olmoe': 17.0, 'granite': 22.5}, started='2026-11-20T10:00:00'), synth(), synth_probe(True), synth_probe(False, '2026-11-18T22:10:00'))
    has(t, 'VERDICT go rule: MET by b5 (olmoe_num_claim, exactly as run)')
    has(t, 'VERDICT touch, olmoe, both sessions: PASS')
    has(t, 'reference block c1 granite: C3 prediction (written for uncapped): inside (19 to 25)')
    has(t, 'reference block stock olmoe: C3 prediction (written for uncapped): MISS (5 to 18)')
    has(t, "prediction 'cap follows power saving': hit")
    has(t, 'share of samples as predicted (max7 >= 2841): 0.90 (no load 0.90, load 0.90)')
    has(t, 'D4 texts across A and B, olmoe num', ': same')
    # 2b. both b1 and b5 meet it: b1 is the claim configuration
    t = score(synth(base={'olmoe': 25.9, 'granite': 23.4}), A(base={'olmoe': 25.9, 'granite': 23.4}))
    has(t, 'VERDICT go rule: MET by b1', 'no budget spent')
    # 2c. a repeated B and a stopped A are reported, not scored
    r = A()
    r['stopped_early'] = True
    t = score(synth(), synth(started='2026-11-19T02:00:00'), r)
    assert sum('REPORTED, NOT SCORED (C6)' in l for l in t) == 2, t
    has(t, 'a repeat: an earlier complete session B is scored')
    # 3. C1: 2419/2419 is capped by the strict reading, uncapped by the written one; two such runs make b1 a capped block
    r = synth(base={'olmoe': 25.9, 'granite': 23.4})
    b1 = quad_runs(r, 'b1')
    b1[0]['cap_before'] = b1[0]['cap_after'] = 'cap=2419/2419'
    b1[1]['cap_after'] = 'cap=2150/2150'
    t = score(r)
    has(t, 'b1 olmoe off', 'uncapped 8 of 10 (written reading, cpu 7 before >= 2419: 10)', 'C2c block is capped (2 capped runs)')
    has(t, 'C1 runs where the strict and the written reading disagree: olmoe_off_claim.0 (2419/2419 -> 2419/2419), olmoe_off_claim.1 (2419/2841 -> 2150/2150)')
    has(t, 'C3 level band: mixed block, neither band scored')
    has(t, 'C2a prediction, uncapped in at least 9 of 10 runs of b1 (P 0.75): MISS (8; written reading 10)')
    has(t, 'go rule, b1: MISSED')
    has(t, 'C2c third bullet', 'FIRES for B b1 (2 capped runs)', '25 is reached only in a clock state we do not control')
    # 3b. a fully capped session: the capped band
    t = score(synth(cap='cap=2035/2035', base={'olmoe': 22.0, 'granite': 21.0}))
    has(t, 'C3 prediction, median capped: inside (21 to 23)')
    # 4. nice refused (EACCES): void, b5 still eligible; partly permitted: no verdict, b5 not eligible
    t = score(synth(nice_ok=False))
    has(t, 'errnos {13: 48}', ': not permitted;')
    has(t, 'VERDICT nice, olmoe, session B: VOID, not permitted')
    has(t, "b5 is 'num + touch (nice asked, refused)', still eligible")
    r = synth()
    x = next(x for x in quad_runs(r, 'b3') if nice_on(x))
    x['c2'][-1] = x['c2'][-1].replace('w0=100:-10:s1:f0:e0', 'w0=100:0:s0:f1:e13')
    t = score(r)
    has(t, ': partly permitted;', "'w0': (15, 1)")
    has(t, 'VERDICT nice, granite, session B: partly permitted: no pass or kill')
    has(t, 'C4/C5 b5 eligible for the go rule: NO: nice is partly permitted (C5)')
    # 4b. B3: an n run whose pool thread still reads -10 drops its quad
    r = synth()
    x = next(x for x in quad_runs(r, 'b3') if not nice_on(x))
    x['workers_after'][1][1] = -10
    t = score(r)
    has(t, 'B3 quads dropped from the nice factor', ': 1 of 4 (' + x['run_tag'] + ')')
    has(t, 'B5 nice (N - n), tok/s paired per quad: +0.80 ± 0.00 (n 3 quads)')
    # 4c. B4: no tok/s effect; the route passes only with a clear mirror
    r = synth(nice_tps=0.0)
    t = score(r)
    has(t, 'VERDICT nice, olmoe, session B: PASS (run-queue wait lower by at least half')
    q = quad_runs(r, 'b3')                            # sporadic waits on both levels: regression to the mean, both ratios low
    for k in range(4):
        for j, lv in enumerate((True, False)):
            for x in q[4 * k:4 * k + 4]:
                if touch_on(x) == lv:
                    hi = nice_on(x) == bool((2 * k + j) % 2)
                    x['c2'][3] = re.sub(r'w0=1000:\d+', f"w0=1000:{400000 if hi else 20000}", x['c2'][3])
    t = score(r)
    b4 = [l for l in t if 'B4 run-queue route' in l][0]
    assert 'ALSO <= 0.5, the route does not pass' in b4 and ': met;' in b4, b4
    has(t, 'VERDICT nice, olmoe, session B: permitted, speed not shown (run-queue route met, but the mirror control also reads <= 0.5)')
    # 5. touch slows and maps nothing: kill with both reasons; b5 not eligible
    r = synth(touch_tps=-0.6)
    for x in r['runs']:
        x['smaps_before'] = {'model': [3000000, 2050781, 0, 0]}
    t = score(r)
    has(t, 'VERDICT touch, olmoe, session B: KILL (missing MB not lower by 30 % and tok/s negative beyond 2 x SE)')
    has(t, 'VERDICT touch, olmoe, both sessions: KILL')
    has(t, 'b5 eligible for the go rule: NO: touch is killed on OLMoE in session B')
    # 6. no long waits: loss not testable (A7)
    r = synth()
    for x in r['runs']:
        x['wait_ms'] = 5
    t = score(r)
    has(t, 'A4/A7 loss endpoint: not testable this session (0 quads')
    has(t, 'VERDICT touch, olmoe, session B: loss not testable, speed not negative')
    has(t, 'VERDICT touch, olmoe, both sessions: mixed')
    # 6b. A6: src is not the wanted set in a b5 run
    r = synth()
    x = quad_runs(r, 'b5')[2]
    x['c2'][-2] = x['c2'][-2].replace('src=want', 'src=resident')
    t = score(r)
    has(t, 'A6 end pass', '17 of 18: prediction MISS; failing: olmoe_num_claim.2(src=resident')
    has(t, 'b5 eligible for the go rule: NO: touch fails A6 in b5 runs')
    # 7. D1 to D4: power saving on in three b1 runs (block void) and one b3 run (its quad dropped), hashes, texts
    r = synth()
    for x in quad_runs(r, 'b1')[:3] + quad_runs(r, 'b3')[5:6]:
        x['power'] = {'powerSave': True, 'lowPower': 1}
    r['build']['libs']['libllama.so'] = 'f' * 64
    r['storage']['models']['granite']['sha256'] = '0' * 64
    for x in quad_runs(r, 'b4'):
        if touch_on(x):
            x['text'] = 'other'
    quad_runs(r, 'b5')[0]['text'] = None
    t = score(r)
    has(t, 'D2 VOID block b1 (olmoe_off_claim): 7 valid runs of 10')
    has(t, 'reads on in 4 of 72 runs')
    has(t, 'D2 scored runs left out of every statistic: 7 of 62; power saving on:', 'in a dropped quad:')
    has(t, "prediction 'power saving reads off in every run' (P 0.95): MISS")
    has(t, 'VOID (library hash differs): build:libllama.so=ffffffff…ffffffff')
    has(t, 'D3 model granite', 'VOID (internal copy fails its sha256 or has none)')
    has(t, 'granite num:', 'DIFFER -> kills (suspends) the knob(s) involved until explained: touch')
    has(t, 'olmoe num: 12 rows', 'rows without a text: olmoe_num_claim.0')
    has(t, 'olmoe: factors (D2: 3 whole valid quads)')
    has(t, 'A5 touch (T - t), tok/s paired per quad: +0.40 ± 0.00 (n 3 quads)')
    # 7b. a plugged quad run drops its quad; two dropped quads void the factor
    r = synth()
    q = quad_runs(r, 'b3')
    q[0]['plugged_run'] = q[4]['plugged_run'] = True
    t = score(r)
    has(t, 'VERDICT touch, olmoe, session B: VOID factor (D2: 2 whole quads, fewer than 3)')
    # 7c. b5 without its num env
    r = synth()
    for x in quad_runs(r, 'b5'):
        x['env_set'] = x['env'] = {k: v for k, v in x['env'].items() if k != 'VOLLIE_SKIP_MASS'}
    has(score(r), 'b5 eligible for the go rule: NO: the num arm did not run in every run')
    # 8. the compressed form loads
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'z.json')
        with open(p, 'w') as f:
            f.write('c2gz:' + base64.b64encode(gzip.compress(json.dumps(synth()).encode())).decode())
        has(run([p]), 'VERDICT touch, olmoe, session B: PASS')
    print('selftest OK')
    return 0


def main(argv):
    if '--selftest' in argv:
        return selftest()
    if '--write-synthetic' in argv:
        i = argv.index('--write-synthetic')
        if i + 1 >= len(argv):
            print('--write-synthetic needs a directory', file=sys.stderr)
            return 2
        print('\n'.join(write_synthetic(argv[i + 1])))
        return 0
    if not argv or argv[0].startswith('-'):
        print(__doc__, file=sys.stderr)
        return 2
    print('\n'.join(run(argv)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
