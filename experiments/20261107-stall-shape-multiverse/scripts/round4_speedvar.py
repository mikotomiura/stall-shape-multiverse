"""conditions-round4.md: 打鍵の速さの測り方を振る・課題文の中で見る（KLiCKe）。

usage: round4_speedvar.py <data_dir> <cache_dir> <out_json> <seed> [jobs]
"""
import json
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import core  # noqa: E402
import round1_extra as r1  # noqa: E402
from round2_press import partial_rank  # noqa: E402

N_BOOT = 2000
B0 = analyze.B0_ID
RES = SCORE = SPEEDS = None
VARIANTS = ("V1", "V2", "V3", "V4", "V5", "V6")


def unit_speeds(u):
    t = np.asarray(u["t"], float)
    ins = np.asarray(u["ins"], bool)
    gaps, cgaps = [], []
    for a, b in u["iv"]:
        m = (t >= a) & (t <= b)
        tt, ii = t[m], ins[m]
        if len(tt) > 1:
            g = np.diff(tt)
            gaps.append(g)
            cgaps.append(g[ii[1:] & ii[:-1]])
    g = np.concatenate(gaps) if gaps else np.array([])
    cg = np.concatenate(cgaps) if cgaps else np.array([])
    med = lambda x: float(np.median(x)) if len(x) else None
    return {"V1": med(g[g < 1000]), "V2": med(g[g < 2000]), "V3": med(g[g < 3000]),
            "V4": (float(np.mean(g[g < 2000])) if (g < 2000).any() else None),
            "V5": med(cg[cg < 2000]), "V6": med(g[(g >= 10) & (g <= 750)])}


def _init(cache, score, speeds):
    global RES, SCORE, SPEEDS
    RES = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    SCORE, SPEEDS = score, speeds


def boot_ci(f, n, rng, strata=None):
    vals = []
    for _ in range(N_BOOT):
        if strata is None:
            ix = rng.integers(0, n, n)
        else:
            ix = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        v = f(ix)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def one(arg):
    cid, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(("r4" + cid).encode())])
    ws, e, n = r1.writer_sums(RES, cid, "KLiCKe")
    out = {}
    for v in VARIANTS:
        keep = [i for i, w in enumerate(ws) if w in SCORE and SPEEDS.get(w, {}).get(v) is not None]
        rate = n[keep] / e[keep]
        sc = np.array([SCORE[ws[i]][0] for i in keep], float)
        sp = np.array([SPEEDS[ws[i]][v] for i in keep], float)
        out[v] = {"n": len(keep), "S2": partial_rank(sc, rate, sp),
                  "ci": boot_ci(lambda ix: partial_rank(sc[ix], rate[ix], sp[ix]), len(keep), rng)}
        if v == "V2":
            pr = np.array([SCORE[ws[i]][1] for i in keep])
            strata = [np.flatnonzero(pr == p) for p in np.unique(pr)]

            def within(ix):
                vals, wts = [], []
                for p in np.unique(pr[ix]):
                    jj = ix[pr[ix] == p]
                    r = partial_rank(sc[jj], rate[jj], sp[jj])
                    if r is not None:
                        vals.append(r)
                        wts.append(len(jj))
                return float(np.average(vals, weights=wts)) if vals else None
            out["V2_within_prompt"] = {"S2": within(np.arange(len(keep))), "ci": boot_ci(within, len(keep), rng, strata)}
    return cid, out


def main(data, cache, out, seed, jobs=8):
    seed = int(seed)
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    speeds = {u["writer"]: unit_speeds(u) for u in units if u["dataset"] == "KLiCKe"}
    score = r1.load_scores(data)
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score, speeds)) as ex:
        res = dict(ex.map(one, [(core.cfg_id(c), seed) for c in core.CONFIGS]))
    ref = json.load(open(os.path.join(os.path.dirname(out), "round3_speed.json")))["by_cfg"]
    for cid, v in res.items():
        if abs(v["V2"]["S2"] - ref[cid]["S2"]) > 1e-6:
            sys.exit("V2 が追加の条件 4 の S2 と違う: %s" % cid)
    others = [c for c in res if c != B0]
    summ = {"b0": res[B0], "by_variant": {}}
    for v in VARIANTS + ("V2_within_prompt",):
        vals = [res[c][v] for c in others]
        summ["by_variant"][v] = {"neg_ci": sum(x["ci"][1] < 0 for x in vals), "pos_ci": sum(x["ci"][0] > 0 for x in vals),
                                 "incl0": sum(x["ci"][0] <= 0 <= x["ci"][1] for x in vals),
                                 "min": min(x["S2"] for x in vals), "median": float(np.median([x["S2"] for x in vals])),
                                 "max": max(x["S2"] for x in vals), "abs_ge_0.1": sum(abs(x["S2"]) >= 0.1 for x in vals)}
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ["by_variant"], ensure_ascii=False, indent=1, default=conv))


if __name__ == "__main__":
    main(*sys.argv[1:])
