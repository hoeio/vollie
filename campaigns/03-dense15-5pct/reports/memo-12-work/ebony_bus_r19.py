# Ebony r19: (1) the lahaina static maps as data, with the kernel's lookup rule, against the caps R75 x4s printed;
# (2) prices for the forms r19 adds, on the committed stage model through ebony_split_r18 (unchanged).
# Run: python3 -I -B ebony_bus_r19.py <repo root>.  Nothing here is a phone measurement except where a line says "device".
# Tables: Samsung SM-F7110 kernel tree (mirror xiliuya/android_kernel_samsung_b2q @ 31574e5bd1),
#   arch/arm64/boot/dts/vendor/qcom/lahaina-v2.dtsi (CPU maps, LPDDR5), lahaina-gpu.dtsi + lahaina-v2-gpu.dtsi (GPU, speed bin 0).
import sys, os, importlib.util

# (core kHz ceiling, target MHz); lookup = first row whose core kHz >= the measured kHz, else the last row (governor_memlat.c core_to_dev_freq)
GOLD_DDR = [(300000, 200), (710400, 451), (844800, 547), (1075200, 768), (1881600, 1555), (2592000, 2092), (2841600, 3196)]
PRIME_DDR_FLOOR = [(2592000, 200), (2841600, 3196)]
GOLD_L3 = [(300000, 300.0), (710400, 499.2), (960000, 614.4), (1209600, 883.2), (1440000, 1056.0), (1766400, 1248.0), (2112000, 1324.8), (2419200, 1516.8), (2841600, 1593.6)]
GOLD_LLCC = [(300000, 150), (710400, 300), (1075200, 466), (1324800, 600), (1881600, 806), (2592000, 933), (2841600, 1000)]
DDR_OPP = [200, 451, 547, 681, 768, 1017, 1555, 1708, 2092, 2736, 3196]            # LPDDR5 rows of ddr_bw_opp_table
GPU_BUS8 = [0, 200, 451, 547, 681, 768, 1017, 1555, 1708, 2092, 2736, 3196]         # qcom,bus-table-ddr8
GPU_LVL = {840: (11, 11, 11), 778: (11, 11, 11), 738: (11, 11, 11), 676: (10, 10, 11), 608: (7, 7, 11), 540: (7, 7, 9),
           491: (7, 5, 7), 443: (7, 5, 7), 379: (6, 5, 7), 315: (2, 2, 7)}           # MHz -> (bus-freq, bus-min, bus-max), ddr8
BWMON_CAP = 1708                                                                    # 6515 MB/s: "echo 6515 > max_freq" in Qualcomm's lahaina post-boot script (not Samsung's; not read)
GBS = lambda mhz: mhz * 2 * 64 / 8 / 1000                                           # LPDDR5, 64 data pins, both edges: theoretical GB/s

def look(tbl, khz):
    for c, t in tbl:
        if c >= khz: return t
    return tbl[-1][1]

def ddr_votes(gold_mhz, prime_mhz, gpu_mhz):
    """Each voter's ceiling in DDR MHz; the bus takes the largest peak vote (icc-rpmh.c: avg summed, peak maxed)."""
    v = {"memlat gold": look(GOLD_DDR, gold_mhz * 1000), "memlat prime": max(look(GOLD_DDR, prime_mhz * 1000), look(PRIME_DDR_FLOOR, prime_mhz * 1000)),
         "bw_hwmon cap": BWMON_CAP}
    if gpu_mhz: f, lo, hi = GPU_LVL[gpu_mhz]; v["gpu (level, max)"] = (GPU_BUS8[f], GPU_BUS8[hi])
    return v

def ddr_range(gold, prime, gpu):
    v = ddr_votes(gold, prime, gpu); cpu = max(v["memlat gold"], v["memlat prime"], v["bw_hwmon cap"])
    g = v.get("gpu (level, max)", (0, 0)); return max(cpu, g[0]), max(cpu, g[1])

def demo():
    assert look(GOLD_DDR, 2419200) == 2092 and look(GOLD_DDR, 2841600) == 3196 and look(GOLD_DDR, 1785600) == 1555 and look(GOLD_DDR, 9e9) == 3196
    assert look(GOLD_L3, 1420800) == 1056.0 and look(GOLD_L3, 1785600) == 1324.8 and look(GOLD_L3, 2150400) == 1516.8
    assert ddr_range(2419, 2841, 840) == (3196, 3196) and ddr_range(1420, 1420, 676) == (2736, 3196) and ddr_range(1420, 1420, 491) == (1708, 1708)
    assert ddr_range(1420, 1420, 0) == (1708, 1708) and abs(GBS(3196) - 51.1) < 0.1 and abs(GBS(1708) - 27.3) < 0.1

if __name__ == "__main__":
    demo()
    print("## Static maps against the caps x4s printed (paper; kHz steps are the phone's cpufreq steps)")
    print("state      gold/prime/GPU MHz   memlat gold  memlat prime  GPU vote (level..max)   DDR MHz on paper   theoretical GB/s   L3 MHz   CPU->LLCC MHz")
    for lab, g, p, gp in (("A cool", 2419.2, 2841.6, 840), ("B onset", 2150.4, 2150.4, 840), ("C capped", 1785.6, 1785.6, 676), ("D capped", 1420.8, 1420.8, 676),
                          ("D pos 20", 1420.8, 1420.8, 491), ("cpu alone, full", 2419.2, 2841.6, 0), ("cpu alone, 2150", 2150.4, 2150.4, 0), ("cpu alone, 1420 (pos 21)", 1420.8, 1420.8, 0)):
        v = ddr_votes(g, p, gp); lo, hi = ddr_range(g, p, gp); gv = v.get("gpu (level, max)", ("-", "-"))
        print(f"{lab:26} {g:6.0f}/{p:6.0f}/{gp:3}   {v['memlat gold']:5}  {v['memlat prime']:5}   {gv[0]}..{gv[1]:<6}  {lo}..{hi}   {GBS(lo):.1f}..{GBS(hi):.1f}   "
              f"{max(look(GOLD_L3, g * 1000), look(GOLD_L3, p * 1000)):.0f}   {max(look(GOLD_LLCC, g * 1000), look(GOLD_LLCC, p * 1000))}")
    print("\n## What the phone drew against the paper ceiling (device arithmetic from R75 x4s; sums are Bonnie r34's estimates)")
    for lab, drawn, lo, hi in (("A pos 1-4, sum", "44.6-47.9 est.", 3196, 3196), ("B pos 5-9, sum", "28.2-34.3 est.", 3196, 3196), ("C pos 12-14, sum", "41.0-42.3 est.", 2736, 3196),
                               ("D pos 16-19, sum", "33.4-36.3 est.", 2736, 3196), ("D pos 20, sum (GPU 491)", "30.1 est.", 1708, 1708), ("pos 21 cpu alone, 1420", "26.77 device", 1708, 1708),
                               ("pos 0 cpu alone, full", "28.57 device", 3196, 3196)):
        print(f"  {lab:28} drew {drawn:16} paper DDR {lo}..{hi} MHz = {GBS(lo):.1f}..{GBS(hi):.1f} GB/s theoretical; at 0.92 practical {0.92 * GBS(lo):.1f}..{0.92 * GBS(hi):.1f}")
    print("  -> pos 20 (30.1 > 27.3) and pos 21 (26.77 of 27.3 = 98 %) do not fit a 1,708 MHz bus: on this phone some vote held the DDR at >= 2,092 MHz under the 1,420 cap.")

    root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
    s = importlib.util.spec_from_file_location("r18", os.path.join(here, "ebony_split_r18.py")); r18 = importlib.util.module_from_spec(s); s.loader.exec_module(r18)
    P, f3 = r18.price, r18.f3
    ref = P("w1", 0, 400, 1.40); G = P("w1", 200, 127, 1.40); F = P("w1", 0, 78, 1.40)
    assert f3(ref) == "7.59 / 6.82 / 6.78" and f3(G) == "8.41 / 7.85 / 7.72" and f3(F) == "9.59 / 8.93 / 8.68"      # r18's rows reproduce
    d = lambda v, b: " / ".join(f"{x - y:+.2f}" for x, y in zip(v, b))
    print("\n## Prices, width 1, O 8, measured head (tok/s Yi / Llama-2 / DeepSeek; estimated)")
    rows = [("as probed, cool (r18): dead 400, sigma 1.40", 0, 400, 1.40, 1.0, ref), ("G, cool (r18)", 200, 127, 1.40, 1.0, ref), ("F per token, rate 1.0 (r18)", 0, 78, 1.40, 1.0, ref),
            ("F per token, rate 0.9", 0, 78, 1.40, 0.9, ref),
            ("F per 8 layers (launch 400 us / 32 split stages): dead 90", 0, 90, 1.40, 1.0, ref), ("F per 4 layers: dead 103", 0, 103, 1.40, 1.0, ref),
            ("F per 2 layers: dead 128", 0, 128, 1.40, 1.0, ref), ("F per layer (r18): dead 178", 0, 178, 1.40, 1.0, ref),
            ("F per 4 layers, rate 0.9", 0, 103, 1.40, 0.9, ref), ("F per 4 layers, rate 0.8", 0, 103, 1.40, 0.8, ref),
            ("G with a warm launch (enq 125 not 200): K 125, dead 127", 125, 127, 1.40, 1.0, ref),
            ("G, launch per 4 layers impossible (CPU needs each y): -", None, None, None, None, None)]
    capC = P("w1", 0, 400, 1.50)
    rows += [("as probed, GPU cap 676 (state C), sigma 1.50", 0, 400, 1.50, 1.0, capC), ("G, state C, sigma 1.50", 200, 127, 1.50, 1.0, capC),
             ("F per token, state C, sigma 1.50", 0, 78, 1.50, 1.0, capC), ("F per 4 layers, state C", 0, 103, 1.50, 1.0, capC),
             ("as probed, state C, sigma 1.60 (bus at the 2,736 step, 42 GB/s)", 0, 400, 1.60, 1.0, capC)]
    capD = P("w1", 0, 400, 1.8, 0.74)
    rows += [("as probed, GPU cap 491 (pos 20): sigma 1.8, GPU rate 0.74", 0, 400, 1.8, 0.74, capD), ("G, GPU cap 491", 200, 127, 1.8, 0.74, capD), ("F per token, GPU cap 491", 0, 78, 1.8, 0.74, capD)]
    for lab, K, J0, sig, rho, b in rows:
        if K is None: continue
        v = P("w1", K, J0, sig, rho); print(f"  {lab:66} {f3(v)}   vs as probed in that state: {d(v, b)}   vs G cool: {d(v, G)}")
    print("\n## rung-7 form (width 2 by two launches, C5 0.93, head x 0.5, O 5.2)")
    for lab, K, J0, sig, rho in (("G", 200, 127, 1.40, 1.0), ("F per token", 0, 78, 1.40, 1.0), ("F per 4 layers", 0, 103, 1.40, 1.0), ("F per 4 layers, rate 0.9", 0, 103, 1.40, 0.9)):
        print(f"  {lab:30} {f3(P('r7', K, J0, sig, rho))}")
