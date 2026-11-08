"""conditions-round2.md: 停滞の率は打鍵の量を数えているか（KLiCKe）と、データごとの打鍵の速さの記述。

usage: round2_press.py <data_dir> <cache_dir> <out_json> <seed> [jobs]
       round2_press.py --selftest
集計値だけを書く。
"""
import json
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import core  # noqa: E402
import round1_extra as r1  # noqa: E402

N_BOOT = 2000
B0 = analyze.B0_ID
RES = None
SCORE = None
PRESS = None
HOUR_MS = 3600_000


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_rank(y, x, z):
    """y と x の、z を固定した偏順位相関。"""
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    c = np.corrcoef([ry, rx, rz])
    a, b, d = c[0, 1], c[0, 2], c[1, 2]
    den = np.sqrt((1 - b * b) * (1 - d * d))
    return float((a - b * d) / den) if den > 0 else None


def unit_press_and_iki(u):
    """観測区間の中の押下の数・区間の長さ（ms）・2 秒未満の隣り合う間隔の中央値。"""
    t = np.asarray(u["t"], float)
    n, dur, gaps = 0, 0.0, []
    for a, b in u["iv"]:
        m = (t >= a) & (t <= b)
        tt = t[m]
        n += int(m.sum())
        dur += max(0.0, b - a)
        if len(tt) > 1:
            g = np.diff(tt)
            gaps.append(g[g < 2000])
    g = np.concatenate(gaps) if gaps else np.array([])
    return n, dur, (float(np.median(g)) if len(g) else None)


def _init(cache, score, press):
    global RES, SCORE, PRESS
    RES = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    SCORE, PRESS = score, press


def one(arg):
    cid, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(("r2" + cid).encode())])
    ws, e, n = r1.writer_sums(RES, cid, "KLiCKe")
    keep = [i for i, w in enumerate(ws) if w in SCORE and w in PRESS]
    rate = n[keep] / e[keep]
    sc = np.array([SCORE[ws[i]][0] for i in keep], float)
    pr = np.array([PRESS[ws[i]] for i in keep], float)
    rho = spearman(rate, sc)
    p1 = spearman(rate, pr)
    p2 = partial_rank(sc, rate, pr)
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(keep), len(keep))
        v = partial_rank(sc[ix], rate[ix], pr[ix])
        if v is not None:
            bs.append(v)
    return cid, {"n": len(keep), "rho": rho, "P1": p1, "P2": p2,
                 "P2_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def selftest():
    rng = np.random.default_rng(3)
    x = rng.random(500)
    y = x + rng.normal(0, 0.3, 500)
    assert partial_rank(y, x, x) is None, "x 自身を固定すると定まらない"
    v = partial_rank(y, x, x + rng.normal(0, 0.005, 500))
    assert abs(v) < 0.2, "ほぼ x を固定すると 0 近く (%r)" % v
    noise = rng.random(500)
    assert abs(partial_rank(y, x, noise) - spearman(y, x)) < 0.05, "無関係を固定するとほぼ元の相関"
    print("selftest ok")


def main(data, cache, out, seed, jobs=8):
    seed = int(seed)
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    press, iki = {}, {}
    for u in units:
        n, dur, med = unit_press_and_iki(u)
        if u["dataset"] == "KLiCKe" and dur > 0:
            press[u["writer"]] = press.get(u["writer"], 0) + n / (dur / HOUR_MS)
        if med is not None:
            iki.setdefault(u["dataset"], {}).setdefault(u["writer"], []).append(med)
    speed = {}
    for d, byw in iki.items():
        v = np.array([np.median(x) for x in byw.values()])
        speed[d] = {"writers": len(v), "median_ms": float(np.median(v)),
                    "q25_ms": float(np.percentile(v, 25)), "q75_ms": float(np.percentile(v, 75))}
    score = r1.load_scores(data)
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score, press)) as ex:
        res = dict(ex.map(one, [(core.cfg_id(c), seed) for c in core.CONFIGS]))
    ref = json.load(open(os.path.join(os.path.dirname(out), "round1_extra.json")))["by_cfg"]
    for cid, v in res.items():
        if abs(v["rho"] - ref[cid]["score"]["rho"]) > 1e-6:
            sys.exit("元の相関を再現しない: %s" % cid)
    # 参照: 得点と押下の率
    ws = sorted(w for w in press if w in score)
    sc = np.array([score[w][0] for w in ws])
    pr = np.array([press[w] for w in ws])
    rng = np.random.default_rng(seed)
    bs = [spearman(pr[ix], sc[ix]) for ix in (rng.integers(0, len(ws), len(ws)) for _ in range(N_BOOT))]
    others = [c for c in res if c != B0]
    fam = {"q": [c for c in others if "_q" in c], "fix": [c for c in others if "_fix" in c]}
    summ = {"score_vs_press": {"rho": spearman(pr, sc), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               "n": len(ws)},
            "b0": res[B0], "speed": speed, "by_family": {}}
    for f, cs in fam.items():
        p1 = [res[c]["P1"] for c in cs]
        summ["by_family"][f] = {
            "n": len(cs), "P1_median": float(np.median(p1)), "P1_min": min(p1), "P1_max": max(p1),
            "rho_pos_ci": sum(ref[c]["score"]["ci"][0] > 0 for c in cs),
            "rho_neg_ci": sum(ref[c]["score"]["ci"][1] < 0 for c in cs),
            "P2_pos_ci": sum(res[c]["P2_ci"][0] > 0 for c in cs),
            "P2_neg_ci": sum(res[c]["P2_ci"][1] < 0 for c in cs),
            "P2_min": min(res[c]["P2"] for c in cs), "P2_median": float(np.median([res[c]["P2"] for c in cs])),
            "P2_max": max(res[c]["P2"] for c in cs)}
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(*sys.argv[1:])
