"""conditions-round5.md (c): 観測の長さ・作文の長さと、停滞・休止の率と得点（KLiCKe）。

usage: round5_length.py <data_dir> <cache_dir> <out_json> <seed> [jobs]
       round5_length.py --selftest
入力: configs.pkl（停滞の 90 通り）・pause_configs.pkl（休止の 36 通り・round5_pause.py）。
集計値だけを書く。本文（Text）は長さを数えるだけで、書き手ごとの値も出さない。
"""
import csv
import io
import json
import os
import pickle
import sys
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import core  # noqa: E402
import round1_extra as r1  # noqa: E402
import round5_pause as r5p  # noqa: E402
from round2_press import partial_rank  # noqa: E402

N_BOOT = 2000
B0 = analyze.B0_ID
STALL = PAUSE = SCORE = LOBS = LTEXT = None


def partial2(y, x, Z):
    """y と x の、Z（列のリスト）を固定した偏順位相関: 順位を Z の順位に回帰した残差の相関。"""
    ry, rx = rankdata(y), rankdata(x)
    A = np.column_stack([np.ones(len(ry))] + [rankdata(z) for z in Z])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    if np.std(ey) == 0 or np.std(ex) == 0:
        return None
    return float(np.corrcoef(ey, ex)[0, 1])


def load_text_len(data):
    """作業場の書き手名 → 本文の文字数（前後の空白を除く）。本文そのものは返さない。"""
    zp = os.path.join(data, "klicke/raw/WritingTask.zip")
    with zipfile.ZipFile(zp) as z:
        members = sorted(m for m in z.namelist() if "keystrokelogs/csv/" in m and m.endswith(".csv"))
        idx = {os.path.basename(m)[:-4]: i for i, m in enumerate(members)}
        raw = z.read("WritingTask/holistic_scores.csv").decode("utf-8-sig", errors="replace")
    out = {}
    for row in csv.DictReader(io.StringIO(raw)):
        i = idx.get(row["ID"])
        if i is not None:
            out["k:%05d" % i] = len(row["Text"].strip())
    return out


def _init(cache, score, lobs, ltext):
    global STALL, PAUSE, SCORE, LOBS, LTEXT
    STALL = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    PAUSE = pickle.load(open(os.path.join(cache, "pause_configs.pkl"), "rb"))
    SCORE, LOBS, LTEXT = score, lobs, ltext


def ci(vals):
    vals = [v for v in vals if v is not None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def one(arg):
    cid, seed = arg
    res = PAUSE if cid.startswith("P_") else STALL
    rng = np.random.default_rng([seed, zlib.crc32(("r5l" + cid).encode())])
    ws, e, n = r1.writer_sums(res, cid, "KLiCKe")
    keep = [i for i, w in enumerate(ws) if w in SCORE and w in LTEXT]
    rate = n[keep] / e[keep]
    sc = np.array([SCORE[ws[i]][0] for i in keep], float)
    lo = np.array([LOBS[ws[i]] for i in keep], float)
    lt = np.array([LTEXT[ws[i]] for i in keep], float)
    out = {"n": len(keep), "rho_raw": r1.spearman(rate, sc),
           "rho_rate_obs": r1.spearman(rate, lo), "rho_rate_text": r1.spearman(rate, lt),
           "p_text": partial2(sc, rate, [lt]), "p_obs": partial2(sc, rate, [lo]), "p_both": partial2(sc, rate, [lt, lo])}
    bt, bo, bb = [], [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(keep), len(keep))
        bt.append(partial2(sc[ix], rate[ix], [lt[ix]]))
        bo.append(partial2(sc[ix], rate[ix], [lo[ix]]))
        bb.append(partial2(sc[ix], rate[ix], [lt[ix], lo[ix]]))
    out["p_text_ci"], out["p_obs_ci"], out["p_both_ci"] = ci(bt), ci(bo), ci(bb)
    return cid, out


def counts(res, ids, key):
    vals = [res[c] for c in ids]
    return {"neg_ci": sum(v[key + "_ci"][1] < 0 for v in vals), "pos_ci": sum(v[key + "_ci"][0] > 0 for v in vals),
            "incl0": sum(v[key + "_ci"][0] <= 0 <= v[key + "_ci"][1] for v in vals),
            "min": min(v[key] for v in vals), "median": float(np.median([v[key] for v in vals])), "max": max(v[key] for v in vals)}


def selftest():
    rng = np.random.default_rng(5)
    y, x, z = rng.normal(size=(3, 500))
    a, b = partial2(y, x, [z]), partial_rank(y, x, z)
    assert abs(a - b) < 1e-6, "共変量 1 つで partial_rank と一致しない: %r %r" % (a, b)
    z2 = x + 1e-6 * rng.normal(size=500)
    assert abs(partial2(y + x, x, [z2, z])) < 0.1, "x そのものを固定しても 0 近くにならない"
    w = rng.normal(size=500)
    assert abs(partial2(y + x, x, [w, z]) - r1.spearman(y + x, x)) < 0.05, "無関係な 2 つを固定して元の相関から離れた"
    print("selftest ok")


def main(data, cache, out, seed, jobs=8):
    seed = int(seed)
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    lobs, lnet = {}, {}
    for u in units:
        if u["dataset"] != "KLiCKe":
            continue
        lobs[u["writer"]] = sum(b - a for a, b in u["iv"]) / analyze.HOUR
        lnet[u["writer"]] = int(np.sum(u["ins"])) - int(np.sum(u["dele"]))
    del units
    score = r1.load_scores(data)
    ltext = load_text_len(data)
    ws = sorted(w for w in lobs if w in score and w in ltext)
    sc = np.array([score[w][0] for w in ws])
    lo = np.array([lobs[w] for w in ws])
    lt = np.array([ltext[w] for w in ws], float)
    ln = np.array([lnet[w] for w in ws], float)
    rng = np.random.default_rng([seed, zlib.crc32(b"r5l-ref")])
    ref = {}
    for name, a, b in (("score_obs", sc, lo), ("score_text", sc, lt), ("obs_text", lo, lt), ("text_net", lt, ln)):
        bs = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(ws), len(ws))
            bs.append(r1.spearman(a[ix], b[ix]))
        ref[name] = {"rho": r1.spearman(a, b), "ci": ci(bs)}
    ref["n"] = len(ws)
    ref["text_len_quartiles"] = [float(np.percentile(lt, q)) for q in (25, 50, 75)]
    ref["obs_h_quartiles"] = [float(np.percentile(lo, q)) for q in (25, 50, 75)]
    ids = [core.cfg_id(c) for c in core.CONFIGS] + [r5p.pid(c) for c in r5p.PCONFIGS]
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score, lobs, ltext)) as ex:
        res = dict(ex.map(one, [(c, seed) for c in ids]))
    # 対照: 元の相関を round1_extra（停滞）と round5_pause（休止）から小数 6 桁で再現する
    rd = os.path.dirname(out)
    r1j = json.load(open(os.path.join(rd, "round1_extra.json")))["by_cfg"]
    r5j = json.load(open(os.path.join(rd, "round5_pause.json")))["by_cfg"]
    for c in ids:
        refrho = (r5j[c] if c.startswith("P_") else r1j[c])["score"]["rho"]
        if res[c]["n"] != (r5j[c] if c.startswith("P_") else r1j[c])["score"]["n"] or abs(res[c]["rho_raw"] - refrho) > 1e-6:
            sys.exit("元の相関を再現しない: %s" % c)
    stall = [c for c in ids if not c.startswith("P_") and c != B0]
    pause = [c for c in ids if c.startswith("P_")]
    pos10 = [c for c in stall if r1j[c]["score"]["ci"][0] > 0]
    summ = {"reference": ref, "b0": res[B0],
            "stall89": {k: counts(res, stall, k) for k in ("p_text", "p_obs", "p_both")},
            "pause36": {k: counts(res, pause, k) for k in ("p_text", "p_obs", "p_both")},
            "pause36_by_family": {fam: {k: counts(res, [c for c in pause if pred(c)], k) for k in ("p_both",)}
                                  for fam, pred in (("fixed", lambda c: "fix" in c), ("quantile", lambda c: "_q" in c))},
            "stall_pos10": {"n": len(pos10), "p_both_pos_ci": sum(res[c]["p_both_ci"][0] > 0 for c in pos10),
                            "p_both_neg_ci": sum(res[c]["p_both_ci"][1] < 0 for c in pos10),
                            "p_both_range": [min(res[c]["p_both"] for c in pos10), max(res[c]["p_both"] for c in pos10)]},
            "rate_obs_range_stall": [min(res[c]["rho_rate_obs"] for c in stall), max(res[c]["rho_rate_obs"] for c in stall)],
            "rate_text_range_stall": [min(res[c]["rho_rate_text"] for c in stall), max(res[c]["rho_rate_text"] for c in stall)],
            "rate_obs_range_pause": [min(res[c]["rho_rate_obs"] for c in pause), max(res[c]["rho_rate_obs"] for c in pause)],
            "rate_text_range_pause": [min(res[c]["rho_rate_text"] for c in pause), max(res[c]["rho_rate_text"] for c in pause)]}
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(*sys.argv[1:])
