#!/usr/bin/env python3
"""Draw the README media.

Usage:
  python3 -I tools/readme_media.py            writes docs/media/{logo,grid,findings,score}.svg and docs/media/detector.gif
  python3 -I tools/readme_media.py --check    checks that the SVGs on disk are the ones this script draws, and that the
                                              counts printed in findings.svg equal those computed from paper/figdata
Reads only aggregate files: paper/figdata/score.csv (per definition) and, for the animation, the detector in
experiments/20261107-stall-shape-multiverse/scripts/core.py run on a synthetic key stream (no data). The animation
needs NumPy and Pillow; the SVGs need nothing beyond the standard library.
No number of the paper comes from this script. Exit code: 0 = written or checked / 1 = a mismatch.
"""
import os
import sys

sys.dont_write_bytecode = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA = os.path.join(ROOT, "docs/media")
DATA = os.path.join(ROOT, "paper/figdata/score.csv")
SURF, CARD, INK, MUTED, GRID, BAND = "#fcfcfb", "#ffffff", "#0b0b0b", "#52514e", "#e4e3df", "#eeeeea"
BLUE, ORANGE, AQUA, LINE = "#2a78d6", "#eb6834", "#1baf7a", "#d6d5d0"
FONT = 'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"'


def svg(w, h, body, label):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" %s role="img" '
            'aria-label="%s">\n%s\n</svg>\n' % (w, h, w, h, FONT, label, "\n".join(body)))


# ---------- logo ----------
def logo():
    # a key stream: ticks with one long gap; the gap is held by a bracket (the window) and marked by a dot (onset)
    xs = [0, 7, 12, 20, 26, 31, 39, 44, 50, 57, 62, 68, 74, 79, 140, 146, 151, 158, 164, 170, 177, 183]
    b = ['<rect width="460" height="96" rx="18" fill="%s"/>' % CARD]
    for i, x in enumerate(xs):
        h = 22 if i % 3 else 30
        b.append('<rect x="%d" y="%d" width="3" height="%d" rx="1.5" fill="%s"/>' % (24 + x, 48 - h // 2, h, INK))
    b.append('<rect x="106" y="26" width="58" height="44" rx="8" fill="%s" fill-opacity="0.16"/>' % BLUE)
    b.append('<path d="M106 76 v6 h58 v-6" fill="none" stroke="%s" stroke-width="2.5" stroke-linecap="round"/>' % BLUE)
    b.append('<circle cx="164" cy="20" r="6" fill="%s"/>' % ORANGE)
    b.append('<text x="226" y="45" font-size="27" font-weight="700" fill="%s">stall-shape</text>' % INK)
    b.append('<text x="226" y="73" font-size="19" fill="%s" letter-spacing="2">MULTIVERSE</text>' % BLUE)
    return svg(460, 96, b, "stall-shape multiverse")


# ---------- the 90 definitions ----------
def grid():
    axes = [("Window W", ["15 s", "30 s*", "60 s"]),
            ("Threshold θ", ["q 0.95", "q 0.99*", "q 0.995", "10 s", "30 s"]),
            ("Condition A", ["net*", "characters"]),
            ("Segmentation", ["default*", "5 min gaps", "30 min gaps"])]
    W, H = 900, 330
    b = ['<rect width="%d" height="%d" rx="16" fill="%s"/>' % (W, H, CARD),
         '<text x="32" y="44" font-size="19" font-weight="700" fill="%s">One detector, four choices, 90 definitions</text>' % INK,
         '<text x="32" y="68" font-size="13" fill="%s">Every combination is one definition; the starred levels make the '
         'baseline. Blue: writer-relative thresholds; orange: fixed seconds.</text>' % MUTED]
    cw = (W - 64 - 3 * 34) / 4
    for k, (name, levels) in enumerate(axes):
        x0 = 32 + k * (cw + 34)
        b.append('<text x="%.0f" y="104" font-size="13" font-weight="700" fill="%s">%s</text>' % (x0, MUTED, name.upper()))
        for j, lv in enumerate(levels):
            y = 116 + j * 30
            base = lv.endswith("*")
            col = BLUE if lv.startswith("q") else ORANGE if (k == 1) else INK
            b.append('<rect x="%.0f" y="%d" width="%.0f" height="24" rx="12" fill="%s" fill-opacity="%s" stroke="%s"/>'
                     % (x0, y, cw, col if base else "#ffffff", "1" if base else "1", col if k == 1 else LINE))
            b.append('<text x="%.0f" y="%d" font-size="13" fill="%s" text-anchor="middle">%s</text>'
                     % (x0 + cw / 2, y + 16, "#ffffff" if base else (col if k == 1 else INK), lv.rstrip("*")))
        if k < 3:
            b.append('<text x="%.0f" y="140" font-size="22" fill="%s" text-anchor="middle">&#215;</text>' % (x0 + cw + 17, MUTED))
    b.append('<line x1="32" x2="%d" y1="284" y2="284" stroke="%s"/>' % (W - 32, GRID))
    b.append('<text x="32" y="310" font-size="14" fill="%s"><tspan font-weight="700">3 × 5 × 2 × 3 = 90</tspan> stall '
             'definitions (baseline + 89)</text>' % INK)
    b.append('<text x="%d" y="310" font-size="14" fill="%s" text-anchor="end">plus a pause detector: interval above '
             '2, 5 or 10 s (18)</text>' % (W - 32, MUTED))
    return svg(W, H, b, "The four axes of the stall detector and their levels")


# ---------- what moves, what stays ----------
def read_score():
    lines = open(DATA, encoding="utf-8").read().split("\n")
    head = lines[0].split()
    return [dict(zip(head, map(float, x.split()))) for x in lines[1:] if x.strip()]


def counts():
    rows = [r for r in read_score() if r["b0"] != 1]
    return {"n": len(rows),
            "present": sum(1 for r in rows if r["lo"] > 0 or r["hi"] < 0),
            "positive": sum(1 for r in rows if r["lo"] > 0),
            "positive_w15": sum(1 for r in rows if r["lo"] > 0 and r["W15"] == 1),
            "adj_negative": sum(1 for r in rows if r["lhi"] < 0),
            "fixed": sum(1 for r in rows if r["fam"] == 0),
            "fixed_negative": sum(1 for r in rows if r["fam"] == 0 and r["rho"] < 0)}


def findings():
    c = counts()
    moves = [("Direction of the stall–score link", "%d of %d positive, all 15-s window + writer quantile"
              % (c["positive"], c["n"])),
             ("Reading after adjusting for length", "given length, no stall interval below zero"),
             ("Rank order of the four datasets", "differs from the baseline for 80 of 89"),
             ("Dispersion within writers vs 1.5", "depends on the definition in two CS1 cells"),
             ("Ratio of CS1 2019 to CS1 2021", "reverses for 20 of 89 (22%), near the 25% criterion")]
    stays = [("Presence of the stall–score link", "interval excludes zero for %d of %d" % (c["present"], c["n"])),
             ("Sign with fixed thresholds", "all %d fixed-second definitions negative" % c["fixed"]),
             ("Largest component of stall length", "stall-to-stall share stays largest in all 89 (CS1)")]
    W, H = 900, 380
    b = ['<rect width="%d" height="%d" rx="16" fill="%s"/>' % (W, H, CARD)]
    for k, (title, col, items) in enumerate([("MOVES WITH THE DEFINITION", ORANGE, moves),
                                              ("STAYS ACROSS DEFINITIONS", AQUA, stays)]):
        x0 = 28 + k * 436
        b.append('<rect x="%d" y="24" width="416" height="%d" rx="12" fill="%s" fill-opacity="0.07" stroke="%s" '
                 'stroke-opacity="0.45"/>' % (x0, H - 48, col, col))
        b.append('<circle cx="%d" cy="54" r="7" fill="%s"/>' % (x0 + 26, col))
        b.append('<text x="%d" y="59" font-size="14" font-weight="700" fill="%s" letter-spacing="1">%s</text>'
                 % (x0 + 42, INK, title))
        for j, (head, sub) in enumerate(items):
            y = 98 + j * 56
            b.append('<text x="%d" y="%d" font-size="15" font-weight="700" fill="%s">%s</text>' % (x0 + 26, y, INK, head))
            b.append('<text x="%d" y="%d" font-size="13" fill="%s">%s</text>' % (x0 + 26, y + 20, MUTED, sub))
    return svg(W, H, b, "What moves with the definition and what stays")


# ---------- score plot (unchanged from version 0.1.0) ----------
W_, PH, ML, MR, MT, GAP = 760, 210, 64, 16, 56, 64
Y0, Y1 = -0.35, 0.20


def panel(rows, top, keys, title):
    v, lo, hi = keys
    n = len(rows)
    sx = lambda i: ML + (i + 0.5) * (W_ - ML - MR) / n
    sy = lambda y: top + (Y1 - y) / (Y1 - Y0) * PH
    out = ['<text x="%d" y="%d" font-size="13" fill="%s">%s</text>' % (ML, top - 10, INK, title),
           '<rect x="%d" y="%.1f" width="%d" height="%.1f" fill="%s"/>' % (ML, sy(0.1), W_ - ML - MR, sy(-0.1) - sy(0.1), BAND)]
    for t in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="%s" stroke-width="%s"/>'
                   % (ML, W_ - MR, sy(t), sy(t), MUTED if t == 0 else GRID, "1" if t == 0 else "0.6"))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="%s" text-anchor="end">%s</text>'
                   % (ML - 6, sy(t) + 4, MUTED, "%+.1f" % t if t else "0"))
    for k, r in enumerate(rows):
        c = BLUE if r["fam"] == 1 else ORANGE
        x = sx(k)
        out.append('<line x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke="%s" stroke-width="2" stroke-opacity="0.45"/>'
                   % (x, x, sy(r[hi]), sy(r[lo]), c))
        if r["b0"] == 1:
            out.append('<circle cx="%.1f" cy="%.1f" r="5.5" fill="%s" stroke="%s" stroke-width="2"/>' % (x, sy(r[v]), INK, SURF))
        else:
            out.append('<circle cx="%.1f" cy="%.1f" r="3.2" fill="%s" stroke="%s" stroke-width="1"/>' % (x, sy(r[v]), c, SURF))
    return out


def score():
    rows = sorted(read_score(), key=lambda r: r["i"])
    h = MT + 2 * PH + GAP + 40
    s = ['<rect width="100%%" height="100%%" rx="16" fill="%s"/>' % CARD,
         '<text x="%d" y="26" font-size="15" font-weight="700" fill="%s">Stall rate × essay score (KLiCKe), 90 '
         'definitions</text>' % (ML, INK)]
    lx = ML
    for c, label in ((BLUE, "writer-quantile threshold (54)"), (ORANGE, "fixed threshold (36)"), (INK, "baseline")):
        s.append('<circle cx="%d" cy="42" r="4" fill="%s"/>' % (lx + 4, c))
        s.append('<text x="%d" y="46" font-size="12" fill="%s">%s</text>' % (lx + 13, MUTED, label))
        lx += 13 + 7 * len(label) + 24
    s.append('<rect x="%d" y="36" width="14" height="10" fill="%s"/>' % (lx, BAND))
    s.append('<text x="%d" y="46" font-size="12" fill="%s">within &#177;0.1</text>' % (lx + 19, MUTED))
    s += panel(rows, MT + 20, ("rho", "lo", "hi"), "(a) Unadjusted Spearman correlation, definitions ordered by it")
    s += panel(rows, MT + 20 + PH + GAP, ("ln", "llo", "lhi"), "(b) Given essay length and length of observation, same order")
    s.append('<text x="%d" y="%d" font-size="11" fill="%s">Bars: 95%% intervals from 2,000 resamples of writers. '
             'Source: paper/figdata/score.csv.</text>' % (ML, h - 12, MUTED))
    return svg(W_, h, s, "Stall rate and essay score across 90 definitions, unadjusted and given length")


# ---------- animation: one key stream, many definitions ----------
def stream():
    import random
    rnd = random.Random(7)
    ev, t = [], 0
    # (typing seconds, pause after in seconds)
    plan = [(40, 3), (25, 12), (30, 4), (20, 26), (35, 2), (30, 48), (25, 6), (15, 8), (30, 95), (35, 17), (40, 5),
            (25, 33), (30, 0)]
    for k, (dur, pause) in enumerate(plan):
        end = t + dur * 1000
        while t < end:
            kind = "d" if (k == 7 and rnd.random() < 0.8) else ("d" if rnd.random() < 0.06 else "i")
            ev.append((t, kind))
            t += int(rnd.lognormvariate(5.4, 0.45))
        t += pause * 1000
    return ev


def detector_gif(path):
    import numpy as np  # noqa: F401  (core needs it)
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, os.path.join(ROOT, "experiments/20261107-stall-shape-multiverse/scripts"))
    import core
    ev = stream()
    u = core.synth(ev)
    T = ev[-1][0]
    defs = [("Baseline: W 30 s · θ writer 0.99 quantile · net", dict(core.B0)),
            ("W 15 s · θ writer 0.99 quantile · net", dict(core.B0, W=15_000)),
            ("W 60 s · θ writer 0.99 quantile · net", dict(core.B0, W=60_000)),
            ("W 30 s · θ fixed 10 s · net", dict(core.B0, theta=("fix", 10_000))),
            ("W 30 s · θ fixed 30 s · net", dict(core.B0, theta=("fix", 30_000))),
            ("W 30 s · θ writer 0.99 quantile · characters only", dict(core.B0, rule="ins"))]
    gaps = [(a[0], b[0]) for a, b in zip(ev, ev[1:])]
    pauses = [("Pause detector: interval above %d s" % s, [(a, b) for a, b in gaps if b - a > s * 1000]) for s in (2, 5, 10)]
    S = 2
    Wd, Hd = 900 * S, 400 * S
    try:
        f_b = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 21 * S)
        f_r = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16 * S)
        f_s = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13 * S)
    except OSError:
        f_b = f_r = f_s = ImageFont.load_default()
    x0, x1 = 40 * S, (900 - 40) * S
    sx = lambda ms: x0 + (x1 - x0) * ms / T
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    def frame(title, spans, onsets, n, kind, idx, allc):
        im = Image.new("RGB", (Wd, Hd), rgb(CARD))
        d = ImageDraw.Draw(im)
        d.text((x0, 22 * S), "Same keystrokes, different definition", font=f_b, fill=rgb(INK))
        d.text((x0, 52 * S), title, font=f_r, fill=rgb(BLUE if kind == "stall" else ORANGE))
        # key presses
        ky = 150 * S
        for t, k in ev:
            c = rgb(ORANGE) if k == "d" else rgb("#8a8984")
            d.line([(sx(t), ky - 20 * S), (sx(t), ky + 20 * S)], fill=c, width=1)
        d.text((x0, ky + 32 * S), "key presses (gray: character, orange: deletion)", font=f_s, fill=rgb(MUTED))
        # labels
        ly = 238 * S
        d.rounded_rectangle([x0, ly - 18 * S, x1, ly + 18 * S], radius=8 * S, fill=rgb(BAND))
        col = rgb(BLUE) if kind == "stall" else rgb(ORANGE)
        for a, b in spans:
            d.rounded_rectangle([sx(a), ly - 14 * S, max(sx(b), sx(a) + 3 * S), ly + 14 * S], radius=5 * S, fill=col)
        for t0 in onsets:
            d.ellipse([sx(t0) - 6 * S, ly - 30 * S, sx(t0) + 6 * S, ly - 18 * S], fill=rgb(INK))
        d.text((x0, ly + 26 * S), ("stall from s to resumption, dot at the onset t0" if kind == "stall"
                                    else "pause = one long interval between key presses"), font=f_s, fill=rgb(MUTED))
        d.text((x1, 22 * S), "%d labels" % n, font=f_b, fill=rgb(INK), anchor="ra")
        d.text((x1, 52 * S), "%.0f min of synthetic typing" % (T / 60000), font=f_s, fill=rgb(MUTED), anchor="ra")
        # strip: label counts of every definition shown, current one highlighted
        by = 318 * S
        d.text((x0, by), "labels per definition", font=f_s, fill=rgb(MUTED))
        bw = (x1 - x0) / len(allc)
        for j, (cn, kd) in enumerate(allc):
            bx = x0 + j * bw
            cc = rgb(BLUE) if kd == "stall" else rgb(ORANGE)
            on = j == idx
            d.rounded_rectangle([bx + 4 * S, by + 22 * S, bx + bw - 4 * S, by + 58 * S], radius=8 * S,
                                fill=cc if on else rgb(BAND))
            d.text((bx + bw / 2, by + 40 * S), str(cn), font=f_r, fill=rgb("#ffffff") if on else rgb(MUTED), anchor="mm")
        return im.resize((Wd // S, Hd // S), Image.LANCZOS)

    items = [(t, core.detect(u, c), "stall") for t, c in defs] + [(t, sp, "pause") for t, sp in pauses]
    allc = [(len(x), k) for _, x, k in items]
    frames = []
    for i, (title, x, kind) in enumerate(items):
        if kind == "stall":
            frames.append(frame(title, [(s, r if r is not None else T) for s, t0, r, _ in x], [t0 for s, t0, r, _ in x],
                                len(x), kind, i, allc))
        else:
            frames.append(frame(title, x, [], len(x), kind, i, allc))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=1600, loop=0, optimize=True, disposal=2)


def drawn():
    return {"logo.svg": logo(), "grid.svg": grid(), "findings.svg": findings(), "score.svg": score()}


def main(argv):
    out = drawn()
    if argv == ["--check"]:
        bad = [k for k, v in out.items()
               if not os.path.exists(os.path.join(MEDIA, k)) or open(os.path.join(MEDIA, k), encoding="utf-8").read() != v]
        c = counts()
        expect = {"n": 89, "present": 82, "positive": 10, "positive_w15": 10, "adj_negative": 0, "fixed": 36,
                  "fixed_negative": 36}
        if c != expect:
            bad.append("counts %s != %s" % (c, expect))
        print("readme media: " + ("match" if not bad else "differ: " + ", ".join(bad)))
        return 1 if bad else 0
    os.makedirs(MEDIA, exist_ok=True)
    for k, v in out.items():
        open(os.path.join(MEDIA, k), "w", encoding="utf-8").write(v)
    detector_gif(os.path.join(MEDIA, "detector.gif"))
    print("wrote " + ", ".join(sorted(out) + ["detector.gif"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
