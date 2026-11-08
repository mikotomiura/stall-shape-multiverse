"""conditions-round8.md: θ を因果に推定し直した二段の区間（ポアソンの重み）。

usage: round8_causal.py inner <cache_dir> <seed> [jobs]
       round8_causal.py outer <cache_dir> <out_json> <seed> [jobs]
       round8_causal.py --selftest
中間（repo の外）: <cache_dir>/theta_causal_inner.pkl。集計値だけを repo に書く。
立ち上がりと限界（candidates）・外側の再標本（one_cfg）は round5_theta.py をそのまま使う。
"""
import json
import math
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
import round5_theta as r5  # noqa: E402

cand = core.cand
M = r5.M
QS = r5.QS
B0 = analyze.B0_ID


def weighted_theta(vals, W, ks):
    """vals: (N,) 元の間隔（時刻の順）。W: (R, N) の重み。各 k で、最初の k 個の重み付き nearest-rank 分位
    （重みの和 × q の切り上げに初めて届く値）を (R, len(ks), len(QS)) で返す。下限 100 個は元の個数で判定。"""
    R = W.shape[0]
    out = np.full((R, len(ks), len(QS)), np.nan)
    for j, k in enumerate(ks):
        if k < core.WARM or k == 0:
            continue
        order = np.argsort(vals[:k], kind="stable")
        sv = vals[:k][order]
        cum = np.cumsum(W[:, :k][:, order], axis=1)
        tot = cum[:, -1]
        for qi, q in enumerate(QS):
            target = np.maximum(np.ceil(q * tot - 1e-9), 1)
            idx = (cum >= target[:, None]).argmax(axis=1)
            th = sv[idx]
            th = np.where(tot > 0, th, np.nan)
            out[:, j, qi] = th
    return out


def unit_inner(arg):
    i, u, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(b"r8causal"), i])
    reg, _ = core.regions(u)
    starts = [r[0] for r in reg]
    nreg = len(reg)
    out = {}
    for seg in core.SEGS:
        iv = core.intervals_for(u, seg)
        pool = sorted(cand.gap_pool(u["t"].tolist(), iv))
        ptime = np.array([p[0] for p in pool], np.int64)
        pval = np.array([p[1] for p in pool], float)
        cands = {(W, rule): r5.candidates(u, W, rule, seg, ptime) for W in core.WS for rule in core.RULES}
        ks = sorted({c[3] for cs in cands.values() for c in cs})
        kpos = {k: j for j, k in enumerate(ks)}
        Wt = np.vstack([np.ones((1, len(pval))), rng.poisson(1.0, (M, len(pval)))]).astype(float)
        th = weighted_theta(pval, Wt, ks)                  # 行 0 は重み 1（元の θ）
        for (W, rule), cs in cands.items():
            if not cs:
                for q in QS:
                    out[core.cfg_id(dict(W=W, theta=("q", q), rule=rule, seg=seg))] = np.zeros((M + 1, nreg), np.int16)
                continue
            x = np.array([c[0] for c in cs], float)
            lb = np.array([c[1] for c in cs], float)
            lf = np.array([c[2] for c in cs], float)
            kj = np.array([kpos[c[3]] for c in cs])
            for qi, q in enumerate(QS):
                T = th[:, kj, qi]
                ok = (~np.isnan(T)) & (T <= lb[None, :]) & (T < lf[None, :])
                t0 = x[None, :] + np.nan_to_num(T)
                ri = np.clip(np.searchsorted(starts, t0, "right") - 1, 0, nreg - 1)
                cnt = np.zeros((M + 1, nreg), np.int64)
                rr, cc = np.nonzero(ok)
                np.add.at(cnt, (rr, ri[rr, cc]), 1)
                out[core.cfg_id(dict(W=W, theta=("q", q), rule=rule, seg=seg))] = cnt.astype(np.int16)
    return {"unit": i, "dataset": u["dataset"], "writer": u["writer"], "counts": out}


def run_inner(cache, seed, jobs):
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    stall = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    seed = int(seed)
    res = []
    with ProcessPoolExecutor(int(jobs)) as ex:
        for k, r in enumerate(ex.map(unit_inner, [(i, u, seed) for i, u in enumerate(units)], chunksize=2)):
            s = stall[r["unit"]]
            for c in r5.QCONFIGS:
                cid = core.cfg_id(c)
                if [n for _, _, n in s["cfg"][cid]["rows"]] != r["counts"][cid][0].tolist():
                    sys.exit("重み 1 の件数が configs.pkl と違う: unit %d %s" % (r["unit"], cid))
            res.append(r)
            if k % 500 == 0:
                print("%d / %d" % (k, len(units)), flush=True)
    with open(os.path.join(cache, "theta_causal_inner.pkl"), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("inner done: 重み 1 の件数は 54 通り × %d 単位で configs.pkl と一致" % len(res))


def _init(cache, score):
    r5.STALL = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    r5.INNER = pickle.load(open(os.path.join(cache, "theta_causal_inner.pkl"), "rb"))
    r5.SCORE = score


def recount(ref, new):
    """点の値と区間の両方が反対側の通りだけを逆に数える。基準の区間が境を含むセルは数えない。"""
    others = [c for c in ref if c != B0]
    ci_sc = lambda c: new[c]["score_ci"] if c in new else ref[c]["score"]["ci"]
    ci_ra = lambda c, k: new[c]["ratio"][k]["ci"] if c in new else ref[c]["ratio"][k]["ci"]
    ci_ph = lambda c, k: new[c]["phi_b"][k]["ci"] if c in new else ref[c]["phi_b"][k]["ci"]
    s = {"score": {}, "A1": {}, "A3": {}}
    rho = lambda c: ref[c]["score"]["rho"]
    s["score"] = {"neg": sum(rho(c) < 0 and ci_sc(c)[1] < 0 for c in others),
                  "pos": sum(rho(c) > 0 and ci_sc(c)[0] > 0 for c in others),
                  "b0_ci": ci_sc(B0),
                  "pos_cfgs": sorted(c for c in others if rho(c) > 0 and ci_sc(c)[0] > 0)}
    s["score"]["other"] = len(others) - s["score"]["neg"] - s["score"]["pos"]
    s["score"]["pos_W15"] = sum(c.startswith("W15") for c in s["score"]["pos_cfgs"])
    for k in ref[B0]["ratio"]:
        lo, hi = ci_ra(B0, k)
        if lo <= 1 <= hi:
            s["A1"][k] = None
            continue
        up = ref[B0]["ratio"][k]["point"] > 1
        f = lambda c: (ref[c]["ratio"][k]["point"] < 1 and ci_ra(c, k)[1] < 1) if up else \
            (ref[c]["ratio"][k]["point"] > 1 and ci_ra(c, k)[0] > 1)
        s["A1"][k] = {"count": sum(f(c) for c in others), "fixed": sum(f(c) for c in others if "fix" in c),
                      "quantile": sum(f(c) for c in others if "_q" in c)}
    for cell in ref[B0]["phi_b"]:
        lo, hi = ci_ph(B0, cell)
        if lo <= analyze.PHI_CUT <= hi:
            s["A3"][cell] = None
            continue
        above = ref[B0]["phi_b"][cell]["phi"] >= analyze.PHI_CUT
        def f(c):
            ci, p = ci_ph(c, cell), ref[c]["phi_b"][cell]["phi"]
            if ci is None or p is None:
                return False
            return (p < analyze.PHI_CUT and ci[1] < analyze.PHI_CUT) if above else (p >= analyze.PHI_CUT and ci[0] >= analyze.PHI_CUT)
        s["A3"][cell] = sum(f(c) for c in others)
    return s


def outside(ref, new):
    ids = list(new)
    sc = sum(not (new[c]["score_ci"][0] <= ref[c]["score"]["rho"] <= new[c]["score_ci"][1]) for c in ids)
    ra = sum(not (new[c]["ratio"][k]["ci"][0] <= ref[c]["ratio"][k]["point"] <= new[c]["ratio"][k]["ci"][1])
             for c in ids for k in new[c]["ratio"])
    ph = sum(new[c]["phi_b"][k]["ci"] is not None and ref[c]["phi_b"][k]["phi"] is not None and
             not (new[c]["phi_b"][k]["ci"][0] <= ref[c]["phi_b"][k]["phi"] <= new[c]["phi_b"][k]["ci"][1])
             for c in ids for k in new[c]["phi_b"])
    n_sc, n_ra, n_ph = len(ids), sum(len(new[c]["ratio"]) for c in ids), sum(len(new[c]["phi_b"]) for c in ids)
    return {"score": [sc, n_sc], "ratio": [ra, n_ra], "phi": [ph, n_ph],
            "share_max": max(sc / n_sc, ra / n_ra, ph / n_ph)}


def run_outer(cache, out, seed, jobs):
    seed = int(seed)
    data = os.path.dirname(os.path.normpath(cache))
    score = r1.load_scores(data)
    ref = json.load(open(os.path.join(os.path.dirname(out), "round1_extra.json")))["by_cfg"]
    ids = [core.cfg_id(c) for c in r5.QCONFIGS]
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score)) as ex:
        new = dict(ex.map(r5.one_cfg, [(c, seed + 8, True) for c in ids]))
    ou = outside(ref, new)
    summ = {"outside": ou, "reading": "置き換える" if ou["share_max"] <= 0.05 else "一段を主に戻す",
            "recount_causal": recount(ref, new), "recount_onestage": recount(ref, {}),
            "width_ratio_score_median": float(np.median([(new[c]["score_ci"][1] - new[c]["score_ci"][0]) /
                                                         (ref[c]["score"]["ci"][1] - ref[c]["score"]["ci"][0]) for c in ids]))}
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": new}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


def selftest():
    ok = True
    rng = np.random.default_rng(2)
    vals = rng.choice([200.0, 500.0, 1500.0, 9000.0], size=400, p=[.5, .3, .15, .05])
    ks = [100, 150, 250, 400]
    one = weighted_theta(vals, np.ones((1, 400)), ks)[0]
    for j, k in enumerate(ks):
        for qi, q in enumerate(QS):
            r = min(max(math.ceil(q * k), 1), k)
            if one[j, qi] != np.sort(vals[:k])[r - 1]:
                ok = False
                print("  - 重み 1 で nearest-rank と違う", k, q)
    W = rng.poisson(1.0, (50, 400)).astype(float)
    a = weighted_theta(vals, W, ks)
    W2 = W.copy()
    W2[:, 150:] = rng.poisson(1.0, (50, 250))
    b = weighted_theta(vals, W2, ks)
    if not np.array_equal(a[:, :2], b[:, :2]):
        ok = False
        print("  - θ が s より後の重みに依った（因果でない）")
    if np.array_equal(a[:, 3], b[:, 3]):
        ok = False
        print("  - 後の重みを替えても k = 400 の θ が動かない（重みが効いていない）")
    c = weighted_theta(np.full(400, 700.0), W, ks)
    if not np.all(c == 700.0):
        ok = False
        print("  - 一定の間隔で θ が動いた")
    # 合成の列で、重み 1 の件数が core.detect と一致する（境の列を含む）
    b1000 = [(i * 1000, "i") for i in range(200)]
    seqs = [(b1000 + [(199_000 + 90_000, "i")], [(0, 199_000 + 30_000 + 1000)]),
            ([(i * 1000, "i") for i in range(100)] + [(129_000, "o"), (219_000, "i")], None)]
    for trial in range(12):
        if trial < len(seqs):
            ev, iv = seqs[trial]
            uu = core.synth(ev, iv=iv)
        else:
            t = np.cumsum(rng.choice([200, 500, 1500, 40_000, 95_000], size=400, p=[.5, .3, .15, .04, .01]))
            kinds = rng.choice(["i", "d", "o"], size=400, p=[.8, .15, .05])
            uu = core.synth(list(zip(t.tolist(), kinds.tolist())))
        uu.update(dataset="x", writer="w", task="t")
        rr = unit_inner((trial, uu, 1))
        for cfg in r5.QCONFIGS:
            if cfg["seg"] == "replay" and len(core.detect(uu, cfg)) != int(rr["counts"][core.cfg_id(cfg)][0].sum()):
                ok = False
                print("  - 重み 1 の件数が detect と違う", trial, core.cfg_id(cfg))
    print("selftest:", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    elif a[0] == "inner":
        run_inner(*a[1:])
    elif a[0] == "outer":
        run_outer(*a[1:])
