# memo-08, Bonnie: gap ledger, K1 issue floors, lever prices and stacks. All outputs *estimated*.
# Inputs: shapes (memo-06 bonnie_price.py), host frontiers (tom-split.md), R69 pre-registered bands,
# Arm Cortex-A78 SWOG (published) and Cortex-X2 SWOG as the stand-in for X1 (published, by analogy).
import contextlib, io, itertools, math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memo-06-work'))
sys.dont_write_bytecode = True
with contextlib.redirect_stdout(io.StringIO()):
    from bonnie_price import M, shape, Q4, Q6

# ---------- 1. K1 issue floors (cycles per weight) from the SWOG rows ----------
# pipe-cycles per instruction: A78 (2 V pipes): SMLAL V0 1/c; SCVTF.4s, FCVTL V0 1/2 per c (2 pipe-cycles);
# USHR/SSHL/SSHLL V1 1/c; SDOT, AND, ADD/SUB, FMUL/FMLA either pipe. X2 (4 V pipes): SMLAL V02 2/c; SCVTF, FCVTL
# V02 1/c (2 pipe-cycles); shifts V13 2/c; SDOT, AND, FMLA any pipe.
def floor(core, d, kernel):
    np_ = 2 if core == "A78" else 4; half = np_ // 2
    if kernel == "stock":                       # per 4 rows x 32 inputs = 128 weights
        lo, hi, sh, w = 2 + 2, 4, 8 + 4 + 2, 128      # lo: SCVTF+FCVTL (pipe-cycles), hi: SSHL, sh: SDOT, AND, FMUL, FMLA
    else:                                       # K1 per 128-row strip and 32-input block, 32d kept columns
        n = 32 * d
        magic = kernel == "K1m"                 # SCVTF replaced by exact ADD + FSUB (|isum| < 2^22)
        lo = 16 * n + (0 if magic else 64) + 64         # SMLAL, SCVTF x2, FCVTL x2
        hi = 4 * n + 32                                  # USHR, SSHLL
        sh = 4 * n + 16 + 64 + (64 if magic else 0)      # AND, SUB, FMUL+FMLA, magic ADD+FSUB
        w = 4096 * d
    t = max(lo / half, hi / half, (lo + hi + sh) / np_)
    return t / w
F = {(c, kk): floor(c, 0.58, kk) for c in ("A78", "X1") for kk in ("stock", "K1", "K1m")}
assert abs(F["A78", "stock"] - 0.0859) < 1e-3 and abs(F["X1", "stock"] - 0.0430) < 1e-3
HOT = {"A78": 0.105, "X1": 0.078}               # R69 P2hot centres (pre-registered), stock cycles per weight
GHZ = {"A78": 2.419, "X1": 2.841}
def keff(kA, kX, nA=2):                         # dynamic chunks: rates add
    rX, rA = GHZ["X1"] / HOT["X1"], GHZ["A78"] / HOT["A78"]
    return (rX + nA * rA) / (rX / kX + nA * rA / kA)
print("## K1 issue floors at d 0.58 (cycles per weight)")
for kk in ("stock", "K1", "K1m"):
    print(kk, "A78 %.4f  X1 %.4f" % (F["A78", kk], F["X1", kk]))
KB = {}
for kk in ("K1", "K1m"):
    # A78: between the additive and the same-efficiency case; X1: between full overlap and additive
    aL, aH = sorted((F["A78", kk] / F["A78", "stock"], (F["A78", kk] + HOT["A78"] - F["A78", "stock"]) / HOT["A78"]))
    xL, xH = max(F["X1", kk], HOT["X1"]) / HOT["X1"], (F["X1", kk] + HOT["X1"] - F["X1", "stock"]) / HOT["X1"]
    aC, xC = (aL + aH) / 2, math.sqrt(xL * xH)
    KB[kk] = (keff(aC, xC), keff(aL, xL), keff(aH, xH), keff(aC, xC, 3))
    print("%s: A78 k %.2f to %.2f, X1 k %.2f to %.2f; 3 cores %.2f (%.2f to %.2f); 4 cores %.2f"
          % (kk, aL, aH, xL, xH, *KB[kk]))

# ---------- 2. Gap ledger ----------
SUBJ = {"Yi-1.5-6B-Chat": (0.42, False), "Llama-2-7b-chat": (0.40, True), "deepseek-llm-7b-chat": (0.40, True)}
CTX = 171                                       # official prompt 155 tokens (Granite) + 16: mean decode position
def body(name):
    m = next(x for x in M if x[0] == name); WL, WH, _ = shape(m)
    _, _, _, L, dm, f, V, h, kvh, hd, _ = m
    kvw = 2 * dm * kvh * hd * L / 1e9 if SUBJ[name][1] else 0.0
    kvpos = 2 * L * kvh * hd * 2 / 1e9                           # GB of F16 KV per position
    attn = 2 * L * h * hd / 1e9 * 2                              # G weight-equivalents per position (cost 2 per MAC)
    return dict(WL=WL, WH=WH, kv=kvw, WS=WL - kvw, V=V, d=dm, kvpos=kvpos, attn=attn, L=L)
Q6K = 1.4                                       # Q6_K head cost per weight vs Q4_0x4 (Ebony L5 centre; R72 block G)
T12 = 950 / 12                                  # steady ms for 12 printed
def cost(b, s, k, ctx=CTX, kvq=1.0, head="q6", ds=0.0):
    s = s + ds
    if head == "q6": HB, HW = b["WH"] * Q6, b["WH"] * Q6K
    else:                                       # 1-bit sign screen + 2048 exact Q6_K rows
        HB, HW = b["V"] * b["d"] / 8e9 + 2048 * b["d"] * Q6 / 1e9, b["WH"] * 0.30 + 2048 * b["d"] / 1e9
    B = HB + b["kv"] * Q4 + b["WS"] * Q4 * (1 - 0.889 * s) + b["kvpos"] * ctx * kvq
    W = HW + b["kv"] + b["WS"] * (1 - s) * k + b["attn"] * ctx
    return B, W
print("\n## Gap ledger (O 8; 12 printed = %.1f ms of max(bytes, compute) after O)" % (T12 - 8))
for nm, (s, kvd) in SUBJ.items():
    b = body(nm)
    B0, W0 = cost(b, 0, 1); B1, W1 = cost(b, s, KB["K1"][0]); B2, W2 = cost(b, s, KB["K1m"][0])
    print("%s: stock B %.2f GB W %.2f G | frontier+K1 B %.2f W %.2f | K1m W %.2f | KV %.0f MB at %d, %.0f MB at 512"
          % (nm, B0, W0, B1, W1, W2, b["kvpos"] * CTX * 1e3, CTX, b["kvpos"] * 512 * 1e3))
    for r in (19, 24, 30, 35):
        print("   r %d: byte line %.2f GB -> bytes gap %+.0f %% ; r needed %.1f" % (r, r * (T12 - 8) / 1e3,
              100 * (B1 / (r * (T12 - 8) / 1e3) - 1), B1 / (T12 - 8) * 1e3))
    for n, cl in ((3, 71.8 * (T12 - 8) / 1e3), (4, 71.8 * 1.15 * (T12 - 8) / 1e3)):
        print("   %d cores: weight line %.2f G -> compute gap K1 %+.0f %%, K1m %+.0f %%" % (n, cl, 100 * (W1 / cl - 1), 100 * (W2 / cl - 1)))

# ---------- 3. Monte Carlo over the pre-registered bands ----------
Z = 1.2816
def ln(c, lo, hi): return lambda: c * math.exp(random.gauss(0, math.log(hi / lo) / (2 * Z)))
LEV = ("K1m", "head1", "kvq8", "core4", "gpu", "thp", "groups", "a55", "tiers", "fallback")
COST = dict(K1m=1, head1=2, kvq8=0.5, core4=0, gpu=5, thp=1, groups=1.5, a55=2, tiers=3, fallback=2)
STEP = {"Yi-1.5-6B-Chat": 3.4, "Llama-2-7b-chat": 4.2, "deepseek-llm-7b-chat": 4.2}
def draw():
    u = random.random()                         # capacity: one phone, same draw for every subject
    cap = 4.6 if u < 0.06 else 4.2 if u < 0.12 else 3.4 if u < 0.35 else 3.0
    return dict(r=ln(24, 19, 30)(), r4=ln(1.06, 0.95, 1.20)(), cb=ln(40.4, 35, 45)(), c4=ln(1.10, 0.95, 1.22)(),   # Ebony's 4-core bands (R18)
                O=ln(8, 6, 12)(), S=ln(1.25, 0.95, 1.6)(), gpu_ok=random.random() < 0.35, gO=ln(3, 1.5, 6)(),
                k=random.uniform(KB["K1"][1], KB["K1"][2]), km=random.uniform(KB["K1m"][1], KB["K1m"][2]),
                head_ok=random.random() < 0.35, thp=1 + (random.uniform(0, 0.15) if random.random() < 0.3 else 0),
                grp=random.uniform(0, 0.06) if random.random() < 0.5 else 0.0,
                a55=random.uniform(0.95, 1.12), cap=cap, l2=random.random() < 0.8,
                tiers_ok=random.random() < 0.20, fb_ok=random.random() < 0.25)
def resident(b, stack):                         # whole file (columns do not shrink it, memo-05 T5) + screen + KV at n_ctx 512
    return (b["WH"] * Q6 + b["WL"] * Q4 + (b["V"] * b["d"] / 8e9 if "head1" in stack else 0)
            + b["kvpos"] * 512 * (0.53 if "kvq8" in stack else 1.0))
def printed(nm, stack, x):
    b = body(nm); s = SUBJ[nm][0]
    if nm == "Llama-2-7b-chat" and not x["l2"]: s = 0.35          # full-set check may move 40 % to 35 %
    k = x["km"] if "K1m" in stack else x["k"]
    if "core4" in stack: k *= 1.04                                 # third A78 shifts the mix to the slower-k core
    tiers = "tiers" in stack and x.get("tiers_ok", True); fb = "fallback" in stack and x.get("fb_ok", True)
    if tiers: k *= 1.22                                            # plane merge on the full and middle columns (my count)
    hd = "1bit" if ("head1" in stack and x["head_ok"]) else "q6"; kvq = 0.53 if "kvq8" in stack else 1.0
    ds = (x["grp"] if "groups" in stack else 0) + (0.10 if fb else 0)
    B, W = cost(b, s, k, kvq=kvq, head=hd, ds=ds)
    if tiers: B -= b["WS"] * Q4 * 0.889 * 0.30 * 0.5               # Ebony 2.1: 0.30 of columns read at half bytes
    if fb:                                                         # 10 % of tokens re-run dense (Ebony 1.2)
        Bd, Wd = cost(b, 0, 1, kvq=kvq, head=hd); B += 0.10 * (Bd - B); W += 0.10 * (Wd - W)
    r = x["r"] * (x["r4"] if "core4" in stack else 1) * (x["thp"] if "thp" in stack else 1)
    c = x["cb"] / Q4 * (x["c4"] if "core4" in stack else 1) * (x["a55"] if "a55" in stack else 1)
    O = x["O"]; tb = B / r
    if "gpu" in stack and x["gpu_ok"] and x["S"] > 1:   # GPU reads only dense rows (k, v, head) at full bytes
        g = (x["S"] - 1) * x["r"]; D = b["kv"] * Q4 + (b["WH"] * Q6 if not ("head1" in stack and x["head_ok"]) else 0)
        Dg = min(D, B * g / (r + g)); tb = max((B - Dg) / r, Dg / g)
        c *= 1 + min(Dg / B, 0.5) * 0.5; O += x["gO"]                   # GPU takes a matching share of compute
    res = resident(b, stack)
    if res > x["cap"]: return 3.0                                  # not resident: the memo-07 non-resident price
    return 0.95 * 1000 / (max(tb, W / c) * 1000 + O)
def central(nm, stack):
    x = dict(r=24, r4=1.13, cb=40.4, c4=1.15, O=8, S=1.25, gpu_ok=True, gO=3, k=KB["K1"][0], km=KB["K1m"][0],
             head_ok=True, thp=1.0, grp=0.03, a55=1.0, cap=4.6, l2=True)
    return printed(nm, stack, x)
random.seed(8); DR = [draw() for _ in range(20000)]
def P12(nm, stack): return sum(printed(nm, stack, x) >= 12 for x in DR) / len(DR)
print("\n## Stacks")
for nm in SUBJ:
    rows = []
    for n in range(len(LEV) + 1):
        for st in itertools.combinations(LEV, n):
            rows.append((sum(COST[l] for l in st), st, central(nm, st)))
    ok = sorted([r for r in rows if r[2] >= 12], key=lambda r: (r[0], -r[2]))
    best_c = max(rows, key=lambda r: r[2])
    print(nm, "| cols only central %.1f | best central %.1f %s" % (central(nm, ()), best_c[2], "+".join(best_c[1])))
    print("   cheapest central >= 12:", ("%s cost %.1f central %.1f" % ("+".join(ok[0][1]), ok[0][0], ok[0][2])) if ok else "none")
    full = tuple(l for l in LEV if l != "a55")
    cand = [(), ("K1m",), ("K1m", "core4"), ("K1m", "core4", "head1"), ("K1m", "core4", "head1", "kvq8"),
            ("K1m", "core4", "head1", "kvq8", "gpu"), full, tuple(LEV)] + ([ok[0][1]] if ok else [])
    for st in cand:
        print("   %-40s central %5.1f  P(>=12) %.3f" % ("+".join(st) or "columns + K1 only", central(nm, st), P12(nm, st)))
    print("   P(resident) %.2f; resident GB: columns only %.3f, with head1 %.3f, head1+kvq8 %.3f" % (sum(x["cap"] >= STEP[nm] for x in DR) / len(DR),
          resident(body(nm), ()), resident(body(nm), ("head1",)), resident(body(nm), ("head1", "kvq8"))))
# r needed at the full stack (central everything else), for the what-to-measure section
print("\n## r (3-core q4x4 read rate) at which the full stack reaches 12 centrally")
for nm in SUBJ:
    lo, hi = 10.0, 80.0
    for _ in range(40):
        mid = (lo + hi) / 2
        x = dict(r=mid, r4=1.13, cb=40.4, c4=1.15, O=8, S=1.25, gpu_ok=True, gO=3, k=KB["K1"][0], km=KB["K1m"][0],
                 head_ok=True, thp=1.0, grp=0.03, a55=1.0, cap=4.6, l2=True)
        st = ("K1m", "core4", "head1", "kvq8", "gpu", "groups")
        lo, hi = (lo, mid) if printed(nm, st, x) >= 12 else (mid, hi)
    x["gpu_ok"] = False; lo2, hi2 = 10.0, 80.0
    for _ in range(40):
        mid = (lo2 + hi2) / 2; x["r"] = mid
        lo2, hi2 = (lo2, mid) if printed(nm, st, x) >= 12 else (mid, hi2)
    print("%s: with GPU lane r >= %.1f; without GPU lane r >= %.1f" % (nm, hi, hi2))

print("\n## Central printed tok/s at r 19 / 24 / 30 / 35 (conventions; residency assumed held)")
KEY = [(), ("K1m",), ("K1m", "core4"), ("K1m", "core4", "head1", "kvq8"), ("K1m", "core4", "gpu"),
       ("K1m", "core4", "head1", "kvq8", "gpu", "groups"), ("K1m", "core4", "tiers"), ("K1m", "core4", "fallback"),
       tuple(l for l in LEV if l != "a55")]
for nm in SUBJ:
    for st in KEY:
        v = []
        for r in (19, 24, 30, 35):
            x = dict(r=r, r4=1.13, cb=40.4, c4=1.15, O=8, S=1.25, gpu_ok=True, gO=3, k=KB["K1"][0], km=KB["K1m"][0],
                     head_ok=True, thp=1.0, grp=0.03, a55=1.0, cap=4.6, l2=True)
            v.append("%.1f" % printed(nm, st, x))
        print("%-22s %-36s %s" % (nm[:22], "+".join(st) or "columns + K1", " / ".join(v)))
for st in [tuple(l for l in LEV if l != "a55"), ("K1m","core4","head1","kvq8","gpu","thp","groups"), ("K1m","core4","gpu"), ("K1m","core4")]:
    print("joint P(all three >= 12) %-40s %.3f" % ("+".join(st), sum(all(printed(nm, st, x) >= 12 for nm in SUBJ) for x in DR) / len(DR)))
for nm in SUBJ:
    st = tuple(l for l in LEV if l != "a55"); res = [x for x in DR if x["cap"] >= STEP[nm]]
    print("P(>=12 | resident) %s %.3f" % (nm, sum(printed(nm, st, x) >= 12 for x in res) / len(res)))

print("\n## Lever deltas on the two lines, against columns + K1 (k %.2f), 3 cores: dB GB, dW G-equivalent" % KB["K1"][0])
for nm in SUBJ:
    b = body(nm); s = SUBJ[nm][0]; k = KB["K1"][0]; B, W = cost(b, s, k)
    rows = dict(K1m=cost(b, s, KB["K1m"][0]), head1=cost(b, s, k, head="1bit"), kvq8=cost(b, s, k, kvq=0.53),
                kvq8_512=(cost(b, s, k, ctx=512, kvq=0.53)[0] - cost(b, s, k, ctx=512)[0] + B, W),
                groups=cost(b, s, k, ds=0.03), fallback_s=cost(b, s, k, ds=0.10))
    tb = b["WS"] * Q4 * 0.889 * 0.15
    print("%s: B %.2f W %.2f | " % (nm[:12], B, W) + " ".join("%s dB %+.3f dW %+.2f;" % (n, v[0] - B, v[1] - W) for n, v in rows.items())
          + " tiers dB %+.3f dW %+.2f; head W at Q6K %.2f; attn W %.3f; KV GB %.3f" % (-tb, b["WS"] * (1 - s) * k * 0.22, b["WH"] * Q6K, b["attn"] * CTX, b["kvpos"] * CTX))
