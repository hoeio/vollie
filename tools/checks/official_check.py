"""Vollie r16 (R48): runnable check of tools/research_parse.py --official on a synthetic kit-r16 report with known answers.
Reference story: 120 fault-free tokens at 2112/2035 (47 ms, MUL_MAT 22, MUL_MAT_ID 16) + 80 at 1881/1785 (52 ms).
Profiled official runs (32 token lines each, steady at 2112/2035): groups 2 and 4 carry a fault-free ramp (tok 2-11 +6 ms on
MUL_MAT -> E = E_clean = +60, 100 % dense); groups 1 and 3 carry faults (tok 2-4 majflt 5, +10 ms on MUL_MAT_ID -> E = E_fault
= +30); one token of run 1 sits at 1670/1670 (not in the story) at 47 × 2112/1670 ms -> scaled from the nearest clock
1881/1785 (52 × 1881/1670 = 58.57), flagged, excess +0.87.
Expect: classifier MIXED (ramp 4/8, fault 4/8), E vs idle_ms slope +1.48 ± 0.04 ms/s (reported), group means 60.0 / 30.2;
runs with E ≤ 0 (tok 2-11 at 46 ms) are neither (1 of 8 -> MIXED; 5 of 8 -> NO GAP); line-1 median 19.10 / best 21.00, text ok; a changed text -> VOID;
a story with < 100 fault-free tokens at one clock -> LINE 2 UNSCORABLE. usage: python3 -I lab/checks/official_check.py"""
import json, os, subprocess, sys, tempfile

P = os.path.join(os.path.dirname(__file__), '..', '..', 'tools', 'research_parse.py')

def line(k, ms, mm, mmid, majflt=0, mhz='2112/2035'):
    us = lambda v: int(round(v * 1000))
    return (f"cpu-prof ab={'A' if (k - 1) % 4 in (0, 3) else 'B'} tok={k} us/token={us(ms)} head=3500 router=800 majflt={majflt} "
            f"minflt=30 mhz={mhz} nodes=1775 barriers=666 kern=1 fuse=1 chunk2=1 | MUL_MAT 265:{us(mm)}+0 MUL_MAT_ID 80:{us(mmid)}+0 "
            f"SSM_SCAN 36:2900+100 GLU 76:200+10")

def report(text2='T', ref_clean=120, fast=0):
    ref = [line(k, 47, 22, 16, 0 if k <= ref_clean else 3) for k in range(1, 121)] + \
          [line(k, 52, 24, 18, 0, '1881/1785') for k in range(121, 201)]
    runs = [{'phase': 'baseline', 'name': 'official 32 · 3t', 'decode_tps': 12.0, 'text': 'T', 'majflt': 9000, 'ttft_ms': 9000},
            {'phase': 'warmup', 'name': 'story 200 · profiled', 'prof': ref, 'tokens': 200},
            {'phase': 'ref', 'name': 'story 200 · profiled', 'prof': ref, 'tokens': 200}]
    tps = iter([19.0, 19.5, 18.0, 18.5, 20.5, 21.0, 17.0, 17.5, 19.2, 19.4])
    for g in range(1, 5):
        for i in range(4):
            if i in (1, 2):
                toks = []
                for k in range(1, 33):
                    if (g - 1) * 2 + i - 1 < fast and 2 <= k <= 11: toks.append(line(k, 46, 21, 16))  # below S: E = -10
                    elif g % 2 == 0 and 2 <= k <= 11: toks.append(line(k, 53, 28, 16))
                    elif g % 2 == 1 and 2 <= k <= 4: toks.append(line(k, 57, 22, 26, 5))
                    elif g == 1 and i == 1 and k == 20: toks.append(line(k, 47 * 2112 / 1670, 22 * 2112 / 1670, 16 * 2112 / 1670, 0, '1670/1670'))
                    else: toks.append(line(k, 47, 22, 16))
                runs.append({'phase': 'prof', 'name': 'official 32 · profiled', 'group': g, 'pause_s': 20 if g % 2 == 0 and i == 0 else 0,
                             'prof': toks, 'tokens': 32, 'ttft_ms': 5000, 'idle_ms': 90000 + (20000 if g % 2 == 0 else 0) + 1000 * i, 'duration_ms': 6600, 'majflt': 400, 'decode_tps': 19.0, 'text': 'T'})
            else:
                runs.append({'phase': 'on', 'name': 'official 32 · default', 'group': g, 'pause_s': 20 if g % 2 == 0 and i == 0 else 0,
                             'decode_tps': next(tps), 'majflt': 100, 'ttft_ms': 5000, 'text': text2 if (g, i) == (3, 3) else 'T',
                             'mhz_after': 'mhz=1094,1094,1094,1094,2112,2112,2112,2035'})
    for _ in range(2):
        runs.append({'phase': 'on', 'name': 'official 32 · default', 'decode_tps': next(tps), 'majflt': 100, 'ttft_ms': 5000, 'text': 'T'})
    return {'kit': 'r16-synthetic', 'runs': runs}

def parse(r):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(r, f)
    return subprocess.run([sys.executable, '-I', P, f.name, '--official'], capture_output=True, text=True, check=True).stdout

out = parse(report())
need = ['median 19.10 tok/s', 'best   21.00 tok/s', 'median ≥ 20.0): no', 'identical across all 19 official runs: yes',
        'S(2112/2035) token 47.00', 'modal 2112/2035 120 ≥ 100: ok',         'prof 1 group 1 pause 0 s: tokens 31  E    +30.9  E_fault    +30.0  E_clean     +0.9',
        'E    +60.0  E_fault     +0.0  E_clean    +60.0', 'prof 2 group 1 pause 0 s: tokens 31  E    +30.0  E_fault    +30.0', 'MUL_MAT +60.0 (100 % of E_clean)', 'FLAG scaled 1881/1785->1670/1670',
        'E ≤ 0 in 0/8 (≥ 5 -> NO GAP), E_clean ≥ 0.5 E > 0 in 4/8, E_fault > 0.5 E > 0 in 4/8 (≥ 6 of 8) -> MIXED',
        'slope +1.48 ± 0.04 ms of E per s idle', 'mean E groups 2, 4 (after the 20 s pause) +60.0 (n 4); groups 1, 3 +30.2 (n 4)', 'prefill majflt 385 (run 400, decode 15)']
miss = [n for n in need if n not in out]
assert not miss, f'missing {miss}\n{out}'
o1 = parse(report(fast=1))  # run 1: E = -10 + 0.87 < 0 -> neither; 4 ramp, 3 fault -> MIXED
assert 'E     -9.1  E_fault     +0.0  E_clean     -9.1' in o1 and '-> neither (E ≤ 0)' in o1, o1
assert 'E ≤ 0 in 1/8 (≥ 5 -> NO GAP), E_clean ≥ 0.5 E > 0 in 4/8, E_fault > 0.5 E > 0 in 3/8 (≥ 6 of 8) -> MIXED' in o1, o1
o5 = parse(report(fast=5))
assert 'E ≤ 0 in 5/8 (≥ 5 -> NO GAP)' in o5 and '-> NO GAP' in o5, o5
assert 'NO -> SESSION VOID' in parse(report(text2='U'))
assert 'LINE 2 UNSCORABLE' in parse(report(ref_clean=90))
print('official_check: PASS')
