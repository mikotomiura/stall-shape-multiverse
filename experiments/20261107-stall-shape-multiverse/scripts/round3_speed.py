"""conditions-round3.md: 停滞の時間を含まない打鍵の速さで固定し直す（KLiCKe）と、記述の値の書き出し。

usage: round3_speed.py <data_dir> <cache_dir> <out_json> <seed> [jobs]
集計値だけを書く。
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
from round2_press import partial_rank, spearman, unit_press_and_iki  # noqa: E402

N_BOOT = 2000
B0 = analyze.B0_ID
RES = SCORE = SPEED = None


def _init(cache, score, speed):
    global RES, SCORE, SPEED
    RES = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    SCORE, SPEED = score, speed


def one(arg):
    cid, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(("r3" + cid).encode())])
    ws, e, n = r1.writer_sums(RES, cid, "KLiCKe")
    keep = [i for i, w in enumerate(ws) if w in SCORE and w in SPEED]
    rate = n[keep] / e[keep]
    sc = np.array([SCORE[ws[i]][0] for i in keep], float)
    sp = np.array([SPEED[ws[i]] for i in keep], float)
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(keep), len(keep))
        v = partial_rank(sc[ix], rate[ix], sp[ix])
        if v is not None:
            bs.append(v)
    return cid, {"n": len(keep), "rho": spearman(rate, sc), "S1": spearman(rate, sp), "S2": partial_rank(sc, rate, sp),
                 "S2_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def counts(vals):
    return {"pos": sum(v[0] > 0 for v in vals), "neg": sum(v[1] < 0 for v in vals),
            "incl0": sum(v[0] <= 0 <= v[1] for v in vals)}


def main(data, cache, out, seed, jobs=8):
    seed = int(seed)
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    speed = {}
    for u in units:
        if u["dataset"] == "KLiCKe":
            _, _, med = unit_press_and_iki(u)
            if med is not None:
                speed[u["writer"]] = med
    score = r1.load_scores(data)
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score, speed)) as ex:
        res = dict(ex.map(one, [(core.cfg_id(c), seed) for c in core.CONFIGS]))
    resdir = os.path.dirname(out)
    ref = json.load(open(os.path.join(resdir, "round1_extra.json")))["by_cfg"]
    r2 = json.load(open(os.path.join(resdir, "round2_press.json")))["by_cfg"]
    for cid, v in res.items():
        if abs(v["rho"] - ref[cid]["score"]["rho"]) > 1e-6:
            sys.exit("元の相関を再現しない: %s" % cid)
    ws = sorted(w for w in speed if w in score)
    sc = np.array([score[w][0] for w in ws])
    sp = np.array([speed[w] for w in ws])
    rng = np.random.default_rng(seed)
    bs = [spearman(sp[ix], sc[ix]) for ix in (rng.integers(0, len(ws), len(ws)) for _ in range(N_BOOT))]
    others = [c for c in res if c != B0]
    groups = {"all": others, "q": [c for c in others if "_q" in c], "fix": [c for c in others if "_fix" in c],
              "ins": [c for c in others if "_ins_" in c], "net": [c for c in others if "_net_" in c]}
    summ = {"score_vs_speed": {"rho": spearman(sp, sc), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               "n": len(ws)}, "b0": res[B0], "groups": {}}
    for g, cs in groups.items():
        rho = np.array([res[c]["rho"] for c in cs])
        s1 = np.array([res[c]["S1"] for c in cs])
        p1 = np.array([r2[c]["P1"] for c in cs])
        s2 = [res[c]["S2"] for c in cs]
        summ["groups"][g] = {
            "n": len(cs),
            "pearson_rho_vs_P1": float(np.corrcoef(rho, p1)[0, 1]), "pearson_rho_vs_S1": float(np.corrcoef(rho, s1)[0, 1]),
            "S1_median": float(np.median(s1)), "S1_min": float(s1.min()), "S1_max": float(s1.max()),
            "S2_min": min(s2), "S2_median": float(np.median(s2)), "S2_max": max(s2),
            "S2_abs_gt_0.1": sum(abs(x) > 0.1 for x in s2),
            "S2_ci": counts([res[c]["S2_ci"] for c in cs]), "P2_ci": counts([r2[c]["P2_ci"] for c in cs]),
            "P2_abs_gt_rho_abs": sum(abs(r2[c]["P2"]) > abs(r2[c]["rho"]) for c in cs),
            "S2_abs_gt_rho_abs": sum(abs(res[c]["S2"]) > abs(res[c]["rho"]) for c in cs)}
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


if __name__ == "__main__":
    main(*sys.argv[1:])
