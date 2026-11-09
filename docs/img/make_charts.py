#!/usr/bin/env python3
"""Regenerates docs/img/speed.svg and docs/img/levers.svg (standard library only).

Every number below is copied from the README / reports/03_final.md; nothing is measured here.
usage: python3 docs/img/make_charts.py
"""
import os
from xml.sax.saxutils import escape

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
INK, MUTED, GRID, BAR, BAR2 = "#1f2328", "#59636e", "#d1d9e0", "#2f6feb", "#8c959f"


def svg(width, height, body, title, desc):
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' width='%d' height='%d' role='img' "
        "aria-labelledby='t d'>\n<title id='t'>%s</title><desc id='d'>%s</desc>\n"
        "<rect width='100%%' height='100%%' fill='#ffffff'/>\n%s</svg>\n"
        % (width, height, width, height, escape(title), escape(desc), body)
    )


def text(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return "<text x='%s' y='%s' font-size='%s' fill='%s' text-anchor='%s' font-weight='%s' %s>%s</text>\n" % (
        x, y, size, fill, anchor, weight, FONT, escape(s))


# ---- Chart 1: printed decode speed, official 32-token benchmark (device) ----
speed = [
    ("Stock llama.cpp settings", 9.4, "R25, 2026-11-04 (8.50 in R31)", BAR2),
    ("Vollie, bit-identical runtime only", 12.0, "R25, 2026-11-04", BAR2),
    ("Vollie, served config (round 16)", 20.51, "median of 10 runs, R49, 2026-11-08", BAR),
]
W, H, left, right, top = 760, 300, 250, 40, 78
scale = (W - left - right) / 24.0
body = text(24, 32, "Printed decode speed, official 32-token benchmark", 17, weight="bold")
body += text(24, 54, "tok/s, measured on the phone (Galaxy Z Flip 3). Rows are different days, each against Vollie in its own session.", 12, MUTED)
for v in (0, 5, 10, 15, 20):
    x = left + v * scale
    body += "<line x1='%s' y1='%s' x2='%s' y2='%s' stroke='%s'/>\n" % (x, top - 8, x, top + 3 * 60 - 14, GRID)
    body += text(x, top + 3 * 60 - 0, str(v), 12, MUTED, "middle")
for i, (label, val, note, color) in enumerate(speed):
    y = top + i * 60
    body += text(left - 12, y + 18, label, 13, INK, "end")
    body += "<rect x='%s' y='%s' width='%.1f' height='26' fill='%s' rx='2'/>\n" % (left, y, val * scale, color)
    body += text(left + val * scale + 8, y + 18, "%s" % val, 14, INK, "start", "bold")
    body += text(left - 12, y + 36, note, 11, MUTED, "end")
with open(os.path.join(OUT, "speed.svg"), "w", encoding="utf-8") as f:
    f.write(svg(W, H, body, "Printed decode speed: stock 9.4, bit-identical 12.0, served 20.51 tok/s",
                "Bar chart of the official 32-token benchmark on the device: stock llama.cpp settings 9.4 tok/s, "
                "Vollie bit-identical runtime 12.0, Vollie served configuration median 20.51."))

# ---- Chart 2: what each lever saved, ms per token (device in-run A/B) ----
levers = [
    ("Bit-identical runtime stack", 27.1, "27.1", "steady token 91.0 to 63.9 ms, R25", False),
    ("Expert skip, threshold 118", 7.90, "7.90", "in-run A/B at 2,112 MHz, R25 / R31", False),
    ("In-register Q3_K expert kernel", 2.67, "2.67", "estimated: scaled from 3.53 at lower clocks, R47", True),
    ("Two-stage output head", 1.70, "1.70", "head op, in-run A/B, R31", False),
    ("Fused expert op", 0.802, "0.80", "MUL_MAT_ID + GLU, R47", False),
    ("Sampler fast path", 0.77, "0.77", "outside-graph, 1.90 to 1.13 ms at 1,555 MHz, R33", False),
]
W, H, left, right, top = 820, 430, 330, 40, 92
scale = (W - left - right) / 30.0
body = text(24, 32, "What each lever saved, ms per token", 17, weight="bold")
body += text(24, 54, "Device in-run A/B results (lower is faster). Different clocks and days, so the bars do not add up to the total.", 12, MUTED)
body += text(24, 72, "Hatched bar = estimated by arithmetic, not a direct measurement. Skip 117 and the head together spend +1.91 % perplexity.", 12, MUTED)
body += ("<defs><pattern id='h' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
         "<rect width='6' height='6' fill='%s'/><rect width='3' height='6' fill='#ffffff' opacity='0.55'/></pattern></defs>\n" % BAR)
for v in (0, 10, 20, 30):
    x = left + v * scale
    body += "<line x1='%s' y1='%s' x2='%s' y2='%s' stroke='%s'/>\n" % (x, top - 8, x, top + len(levers) * 52 - 12, GRID)
    body += text(x, top + len(levers) * 52 + 6, str(v), 12, MUTED, "middle")
for i, (label, val, shown, note, est) in enumerate(levers):
    y = top + i * 52
    fill = "url(#h)" if est else BAR
    body += text(left - 12, y + 16, label, 13, INK, "end")
    body += "<rect x='%s' y='%s' width='%.1f' height='22' fill='%s' rx='2'/>\n" % (left, y, val * scale, fill)
    body += text(left + val * scale + 8, y + 16, shown, 13, INK, "start", "bold")
    body += text(left - 12, y + 34, note, 10.5, MUTED, "end")
with open(os.path.join(OUT, "levers.svg"), "w", encoding="utf-8") as f:
    f.write(svg(W, H, body, "Milliseconds per token saved by each lever",
                "Horizontal bar chart: bit-identical runtime stack 27.1 ms, expert skip 7.90, Q3_K kernel 2.67 estimated, "
                "two-stage head 1.70, fused expert op 0.80, sampler fast path 0.77."))
print("wrote", os.path.join(OUT, "speed.svg"), os.path.join(OUT, "levers.svg"))
