"""conditions-round1.md の (a) 90 通りの区間 と (b) KLiCKe の得点との関連。

usage: round1_extra.py <data_dir> <cache_dir> <out_json> <seed> [jobs]
       round1_extra.py --selftest
集計値だけを書く（書き手ごとの値・ID・本文を書かない）。
"""
import csv
import io
import itertools
import json
import os
import pickle
import sys
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import core  # noqa: E402

N_BOOT = 2000
B0 = analyze.B0_ID
DS = analyze.DATASETS
PAIRS = list(itertools.combinations(DS, 2))
RES = None
SCORE = None


def main_level(d):
    return "task" if d in ("CS1 2019", "CS1 2021") else "session" if d == "CoAuthor" else "writer"


def writer_sums(res, cid, d):
    """主の水準の単位（露出 5 分以上）を書き手ごとに足した (書き手, 露出 h, 件数)。率の点は c3 と同じ。"""
    rows = analyze.block_rows(res, cid, d)
    w, e, n = analyze.level_table(rows, main_level(d))
    acc = {}
    for wi, ei, ni in zip(w, e, n):
        a = acc.setdefault(wi, [0.0, 0.0])
        a[0] += ei
        a[1] += ni
    ws = sorted(acc)
    return ws, np.array([acc[x][0] for x in ws]), np.array([acc[x][1] for x in ws])


def boot_rate(e, n, rng):
    k = len(e)
    idx = rng.integers(0, k, (N_BOOT, k))
    return n[idx].sum(1) / e[idx].sum(1)


def spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _init(cache, score):
    global RES, SCORE
    RES = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    SCORE = score


def one(arg):
    cid, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(cid.encode())])
    out = {"rate": {}, "rate_ci": {}, "phi_b": {}, "ratio": {}, "score": None}
    boots = {}
    for d in DS:
        ws, e, n = writer_sums(RES, cid, d)
        out["rate"][d] = float(n.sum() / e.sum())
        b = boot_rate(e, n, rng)
        boots[d] = b
        out["rate_ci"][d] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    for a, b in PAIRS:
        r = boots[a] / boots[b]
        out["ratio"]["%s / %s" % (a, b)] = {"point": out["rate"][a] / out["rate"][b],
                                            "ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
    for d, lev in analyze.MAIN_B:
        rows = analyze.block_rows(RES, cid, d)
        w, e, n = analyze.level_table(rows, lev)
        out["phi_b"]["%s %s" % (d, lev)] = {"phi": analyze.phi(w, e, n, True),
                                            "ci": analyze.phi_boot(w, e, n, True, rng)}
    # (b) KLiCKe の得点
    ws, e, n = writer_sums(RES, cid, "KLiCKe")
    keep = [i for i, w in enumerate(ws) if w in SCORE]
    rate = n[keep] / e[keep]
    sc = np.array([SCORE[ws[i]][0] for i in keep], float)
    pr = np.array([SCORE[ws[i]][1] for i in keep])
    rho = spearman(rate, sc)
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(keep), len(keep))
        v = spearman(rate[ix], sc[ix])
        if v is not None:
            bs.append(v)
    per_prompt = [spearman(rate[pr == p], sc[pr == p]) for p in np.unique(pr)]
    out["score"] = {"n": len(keep), "n_unmatched": len(ws) - len(keep), "rho": rho,
                    "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "rho_prompt_median": float(np.median([v for v in per_prompt if v is not None]))}
    return cid, out


def load_scores(data):
    """KLiCKe の作業場の書き手名（k:<番号>）→ (得点, 課題文の番号)。ID・本文は返り値に残さない。"""
    zp = os.path.join(data, "klicke/raw/WritingTask.zip")
    with zipfile.ZipFile(zp) as z:
        members = sorted(m for m in z.namelist() if "keystrokelogs/csv/" in m and m.endswith(".csv"))
        idx = {os.path.basename(m)[:-4]: i for i, m in enumerate(members)}
        raw = z.read("WritingTask/holistic_scores.csv").decode("utf-8-sig", errors="replace")  # 本文（Text）に cp1252 の字が混じる。ID・Score は ASCII
    prompts = {}
    out = {}
    for row in csv.DictReader(io.StringIO(raw)):
        i = idx.get(row["ID"])
        if i is None:
            continue
        p = prompts.setdefault(row["Prompt"], len(prompts))
        out["k:%05d" % i] = (float(row["Score"]), p)
    return out


def summarize(res):
    b0 = res[B0]
    others = [c for c in res if c != B0]
    s = {"n_cfg_excl_b0": len(others), "A1": {}, "A2": {}, "A3": {}, "phi_range": {}, "B": {}}
    for k, v0 in b0["ratio"].items():
        lo, hi = v0["ci"]
        if lo <= 1 <= hi:
            s["A1"][k] = {"b0_point": v0["point"], "b0_ci": v0["ci"], "count": None, "note": "基準の向きが定まらない"}
            continue
        up = v0["point"] > 1
        cnt = sum((res[c]["ratio"][k]["ci"][1] < 1) if up else (res[c]["ratio"][k]["ci"][0] > 1) for c in others)
        pt = sum((res[c]["ratio"][k]["point"] < 1) if up else (res[c]["ratio"][k]["point"] > 1) for c in others)
        s["A1"][k] = {"b0_point": v0["point"], "b0_ci": v0["ci"], "count": cnt, "point_count": pt}
    for cell, v0 in b0["phi_b"].items():
        vals = [res[c]["phi_b"][cell] for c in others]
        s["A2"][cell] = sum(v["ci"] is not None and v["ci"][0] > 1 for v in vals)
        above = v0["phi"] >= analyze.PHI_CUT
        s["A3"][cell] = sum(v["ci"] is not None and ((v["ci"][1] < analyze.PHI_CUT) if above else (v["ci"][0] >= analyze.PHI_CUT))
                            for v in vals)
        ph = [v["phi"] for v in vals if v["phi"] is not None]
        s["phi_range"][cell] = {"b0": v0["phi"], "b0_ci": v0["ci"], "min": min(ph), "median": float(np.median(ph)),
                                "max": max(ph)}
    sc0 = b0["score"]
    vals = [res[c]["score"] for c in others]
    s["B"] = {"b0": sc0, "rho_min": min(v["rho"] for v in vals), "rho_median": float(np.median([v["rho"] for v in vals])),
              "rho_max": max(v["rho"] for v in vals),
              "sign_opposite_point": sum(np.sign(v["rho"]) != np.sign(sc0["rho"]) for v in vals),
              "ci_excl0_pos": sum(v["ci"][0] > 0 for v in vals), "ci_excl0_neg": sum(v["ci"][1] < 0 for v in vals),
              "sign_opposite_ci": sum(((v["ci"][1] < 0) if sc0["rho"] > 0 else (v["ci"][0] > 0)) for v in vals),
              "prompt_median_range": [min(v["rho_prompt_median"] for v in vals), max(v["rho_prompt_median"] for v in vals)],
              "n_matched_b0": sc0["n"], "n_unmatched_b0": sc0["n_unmatched"]}
    return s


def selftest():
    rng = np.random.default_rng(1)
    x = rng.random(300)
    assert abs(spearman(x, x) - 1) < 1e-12, "陽性対照: 同じ順位で 1"
    v = spearman(x, rng.permutation(x))
    assert abs(v) < 0.2, "陰性対照: 並べ替えで 0 前後 (%r)" % v
    e = np.full(50, 1.0)
    n = np.full(50, 3.0)
    b = boot_rate(e, n, rng)
    assert np.allclose(b, 3.0), "率が一定なら区間は点"
    print("selftest ok")


def main(data, cache, out, seed, jobs=8):
    score = load_scores(data)
    seed = int(seed)
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score)) as ex:
        res = dict(ex.map(one, [(core.cfg_id(c), seed) for c in core.CONFIGS]))
    # 対照: 率の点が c3 と一致・基準の φ の区間が c2 と 0.05 以内
    resdir = os.path.dirname(out)
    c3 = json.load(open(os.path.join(resdir, "c3_multiverse.json")))
    for cid, v in res.items():
        for d in DS:
            if abs(v["rate"][d] - c3[cid][d]["rate_pooled"]) > 1e-9:
                sys.exit("率の点が c3 と違う: %s %s" % (cid, d))
    c2 = json.load(open(os.path.join(resdir, "c2_overdispersion.json")))
    for cell, v in res[B0]["phi_b"].items():
        d, lev = cell.rsplit(" ", 1)
        ref = c2[d]["%s_b" % lev]["ci"]
        if max(abs(v["ci"][0] - ref[0]), abs(v["ci"][1] - ref[1])) > 0.05:
            sys.exit("基準の φ の区間が c2 と 0.05 を超えて違う: %s %r %r" % (cell, v["ci"], ref))
    summ = summarize(res)
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(*sys.argv[1:])
