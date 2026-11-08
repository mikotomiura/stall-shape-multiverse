"""C1・C2・C3 の集計と §J の数え（conditions.md）。出力は集計値だけ（書き手ごとの値・時刻・ID を出さない）。

usage: analyze.py <cache_dir> <out_dir> <seed> [--skip-c1]
入力: <cache_dir>/configs.pkl（run_configs.py）・<cache_dir>/c3b.json（c3b.py）
"""
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

DATASETS = ("CS1 2019", "CS1 2021", "CoAuthor", "KLiCKe")
MIN_MS = core.MIN_UNIT_OBS_MS
HOUR = 3600_000
N_BOOT = 2000
PHI_CUT = 1.5
SHARE_CUT = 0.25
B0_ID = core.cfg_id(core.B0)


# ---------- 行の組み立て ----------

def block_rows(res, cid, dataset):
    """まとまりごとの行: (書き手, 課題, 単位, 日, まとまり, 露出 ms, 件数)。"""
    out = []
    for r in res:
        if r["dataset"] != dataset:
            continue
        for k, (day, e, n) in enumerate(r["cfg"][cid]["rows"]):
            out.append((r["writer"], r["task"], r["unit"], (r["unit"], day), (r["unit"], k), e, n))
    return out


def level_table(rows, level):
    """level: writer / task / day / block（CS1 の task は単位 = 書き手 × 課題）・session（CoAuthor の単位）。"""
    idx = {"writer": 0, "task": 2, "session": 2, "day": 3, "block": 4}[level]
    agg = defaultdict(lambda: [None, 0, 0])
    for row in rows:
        a = agg[row[idx]]
        a[0] = row[0]
        a[1] += row[5]
        a[2] += row[6]
    w, e, n = zip(*[(v[0], v[1], v[2]) for v in agg.values() if v[1] >= MIN_MS]) if agg else ((), (), ())
    return list(w), np.array(e, float) / HOUR, np.array(n, float)


def short_rows(rows, level):
    """D4: 露出 5 分未満で外した単位（主の水準）の数と、そこに落ちた停滞の数。"""
    idx = {"writer": 0, "task": 2, "session": 2, "day": 3, "block": 4}[level]
    agg = defaultdict(lambda: [0, 0])
    for row in rows:
        agg[row[idx]][0] += row[5]
        agg[row[idx]][1] += row[6]
    out = [v for v in agg.values() if v[0] < MIN_MS]
    return {"units": len(out), "stalls": int(sum(v[1] for v in out))}


def phi(w, e, n, within):
    """Pearson の分散の指数。within=True なら書き手ごとの率で期待を作る（(b)）。"""
    if len(n) < 3:
        return None
    if not within:
        lam = n.sum() / e.sum()
        mu = lam * e
        k, p = len(n), 1
        keep = mu > 0
    else:
        ws = np.array(w)
        uniq, inv = np.unique(ws, return_inverse=True)
        sn = np.bincount(inv, n)
        se = np.bincount(inv, e)
        mu = (sn / se)[inv] * e
        keep = mu > 0   # 件数 0 の書き手は情報が無いので外す
        k = int(keep.sum())
        p = len(np.unique(inv[keep]))
    if k - p <= 0:
        return None
    return float((((n - mu) ** 2)[keep] / mu[keep]).sum() / (k - p))


def phi_boot(w, e, n, within, rng):
    ws = np.array(w)
    uniq, inv = np.unique(ws, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([groups[g] for g in pick])
        wb = np.repeat(np.arange(len(pick)), [len(groups[g]) for g in pick])   # 再標本の書き手を別人として扱う
        v = phi(wb, e[ix], n[ix], within)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


LEVELS = {
    "CS1 2019": [("block", True), ("day", True), ("task", True), ("block", False), ("day", False),
                 ("task", False), ("writer", False)],
    "CS1 2021": [("block", True), ("day", True), ("task", True), ("block", False), ("day", False),
                 ("task", False), ("writer", False)],
    "CoAuthor": [("session", True), ("session", False), ("writer", False)],
    "KLiCKe": [("writer", False)],
}
MAIN_B = [("CS1 2019", "block"), ("CS1 2019", "task"), ("CS1 2021", "block"), ("CS1 2021", "task"),
          ("CoAuthor", "session")]


# ---------- C3 の 1 通り ----------

def summarize_cfg(res, cid):
    out = {}
    for d in DATASETS:
        rows = block_rows(res, cid, d)
        uw, ue, un = level_table(rows, "session" if d != "CS1 2019" and d != "CS1 2021" else "task")
        ww, we, wn = level_table(rows, "writer")
        main_lev = "session" if d != "CS1 2019" and d != "CS1 2021" else "task"
        short = short_rows(rows, main_lev)
        lens = [l for r in res if r["dataset"] == d for _, l in r["cfg"][cid]["lens"]]
        done = np.array([l for l in lens if l is not None], float) / 1000
        ag = np.array([r["cfg"][cid]["agree"] for r in res if r["dataset"] == d], float).sum(0)
        o = {
            "stalls": int(un.sum()), "hours": float(ue.sum()),
            "rate_pooled": float(un.sum() / ue.sum()) if ue.sum() else None,
            "rate_writer_median": float(np.median(wn / we)) if len(we) else None,
            "writers_zero_share": float(np.mean(wn == 0)) if len(wn) else None,
            "length_median_s": float(np.median(done)) if len(done) else None,
            "length_q": [float(np.percentile(done, 25)), float(np.percentile(done, 75))] if len(done) else None,
            "unresumed": int(sum(l is None for l in lens)),
            "agree_b0_in_cfg": float(ag[0] / ag[1]) if ag[1] else None,
            "agree_cfg_in_b0": float(ag[2] / ag[3]) if ag[3] else None,
            "dropped_short": short,
            "phi": {},
        }
        for lev, within in LEVELS[d]:
            w, e, n = level_table(rows, lev)
            o["phi"]["%s_%s" % (lev, "b" if within else "a")] = phi(w, e, n, within)
        out[d] = o
    rates = {d: out[d]["rate_pooled"] or 0 for d in DATASETS}
    out["rank"] = sorted(DATASETS, key=lambda d: -rates[d])
    return out


# ---------- C1 ----------

def unit_frames(res, cid, d):
    rows = [r for r in block_rows(res, cid, d) if r[5] >= MIN_MS]
    lens = []
    for r in res:
        if r["dataset"] != d:
            continue
        rg_days = {k: day for k, (day, _, _) in enumerate(r["cfg"][cid]["rows"])}
        for k, l in r["cfg"][cid]["lens"]:
            if l is not None and l > 0:
                lens.append((r["writer"], r["task"], r["unit"], (r["unit"], rg_days[k]), (r["unit"], k), l / 1000))
    return rows, lens


def codes(vals):
    m = {}
    return np.array([m.setdefault(v, len(m)) for v in vals]), len(m)


def count_design(rows, levels):
    """levels: [(名前, 行の列番号)]。疎な exog_vc と ident を作る。"""
    mats, ident = [], []
    for j, (_, col) in enumerate(levels):
        c, k = codes([r[col] for r in rows])
        mats.append(sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), c)), shape=(len(rows), k)))
        ident += [j] * k
    return sp.hstack(mats).tocsr(), np.array(ident)


def fit_counts(rows, levels, endog=None):
    from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
    e = np.array([r[5] for r in rows], float) / HOUR
    y = np.array([r[6] for r in rows], float) if endog is None else endog
    X = np.column_stack([np.ones(len(rows)), np.log(e)])
    Z, ident = count_design(rows, levels)
    # ずれ 9: 分散成分の事前の SD を既定の 1 ではなく 2 にした（弱くした）。K5 も同じ設定で回す
    m = PoissonBayesMixedGLM(y, X, Z, ident, vcp_p=2.0, fe_p=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # ずれ 2: 既定の BFGS は乱効果の数の 2 乗の行列を持ち実データで終わらない。合成で BFGS と小数 2 桁まで一致を確かめた
        f = m.fit_vb(fit_method="L-BFGS-B", minim_opts={"maxiter": 20000})
    sd = np.exp(np.asarray(f.vcp_mean))
    ok = getattr(f, "optim_retvals", None)
    return {"sd": {n: float(s) for (n, _), s in zip(levels, sd)}, "log_exposure_slope": float(f.fe_mean[1]),
            "converged": bool(ok.success) if ok is not None else None,
            "finite": bool(np.all(np.isfinite(sd)))}


LEN_METHODS = (("default", {}), ("powell", {"method": ["powell"], "maxiter": 5000}),
               ("nm", {"method": ["nm"], "maxiter": 5000}))


def fit_lengths(lens, nested, endog=None):
    """対数の長さ。書き手を群・nested を分散成分。CS1 の課題の共通の効果は固定効果。"""
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM
    df = pd.DataFrame({"writer": [str(x[0]) for x in lens], "task": [str(x[1]) for x in lens],
                       "unit": [str(x[2]) for x in lens], "day": [str(x[3]) for x in lens],
                       "block": [str(x[4]) for x in lens]})
    df["y"] = np.log([x[5] for x in lens]) if endog is None else endog
    form = "y ~ C(task)" if df["task"].nunique() > 1 else "y ~ 1"
    vc = {n: "0 + C(%s)" % n for n in nested}
    mod = MixedLM.from_formula(form, df, groups="writer", vc_formula=vc if vc else None, re_formula="1")
    # ずれ 13: 既定の最適化は実データで REML の最大に届かず未収束のことがある。既定・Powell・Nelder–Mead で当て、
    # 収束した中で REML の対数尤度が最大のものを採る（推定量は REML のまま）
    fits = {}
    for name, kw in LEN_METHODS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fits[name] = mod.fit(reml=True, **kw)
            except (np.linalg.LinAlgError, ValueError):
                continue
    ok = {k: f for k, f in fits.items() if f.converged and np.isfinite(f.llf)}
    pool = ok or {k: f for k, f in fits.items() if np.isfinite(f.llf)}
    best = max(pool, key=lambda k: pool[k].llf)
    f = pool[best]
    out = {"writer": float(math.sqrt(max(0.0, float(f.cov_re.iloc[0, 0]))))}
    for n, v in zip(f.model.exog_vc.names, f.vcomp):
        out[n] = float(math.sqrt(max(0.0, v)))
    out["stall"] = float(math.sqrt(f.scale))
    return {"sd": out, "converged": bool(f.converged), "method": best,
            "llf": {k: float(x.llf) for k, x in fits.items()},
            "converged_by_method": {k: bool(x.converged) for k, x in fits.items()}}


def shares(sd, keys):
    v = {k: sd[k] ** 2 for k in keys}
    tot = sum(v.values())
    return {k: (v[k] / tot if tot else None) for k in keys}


COUNT_LEVELS = {
    "CS1 2019": [("writer", 0), ("task_common", 1), ("task_in_writer", 2), ("day", 3), ("block", 4)],
    "CS1 2021": [("writer", 0), ("task_common", 1), ("task_in_writer", 2), ("day", 3), ("block", 4)],
    "CoAuthor": [("writer", 0), ("session", 2)],
}
LEN_NESTED = {"CS1 2019": ["unit", "day", "block"], "CS1 2021": ["unit", "day", "block"], "CoAuthor": ["unit"]}


def k5_recovery(rows, lens, d, rng):
    """既知の分散成分から模擬して戻るか（±30%）。"""
    lv = COUNT_LEVELS[d]
    out = {}
    for tag, truth in (("all0.3", {n: 0.3 for n, _ in lv}),
                       ("writer0.5", {n: (0.5 if n == "writer" else 0.2) for n, _ in lv})):
        e = np.array([r[5] for r in rows], float) / HOUR
        eta = np.log(np.sum([r[6] for r in rows]) / e.sum()) + np.log(e)
        for n, col in lv:
            c, k = codes([r[col] for r in rows])
            eta = eta + rng.normal(0, truth[n], k)[c]
        y = rng.poisson(np.exp(eta)).astype(float)
        est = fit_counts(rows, lv, endog=y)["sd"]
        out["counts_" + tag] = {n: {"true": truth[n], "est": est[n], "ok": abs(est[n] / truth[n] - 1) <= 0.3}
                                for n, _ in lv}
        # 長さ
        nested = LEN_NESTED[d]
        names = ["writer"] + nested
        tl = {n: (truth.get(n) if n in truth else (truth["task_in_writer"] if n == "unit" and "task_in_writer" in truth
                                                       else truth.get("session", 0.3))) for n in names}
        col = {"writer": 0, "unit": 2, "day": 3, "block": 4}
        yl = rng.normal(0, 1.0, len(lens))
        for n in names:
            c, k = codes([x[col[n]] for x in lens])
            yl = yl + rng.normal(0, tl[n], k)[c]
        estl = fit_lengths(lens, nested, endog=yl)["sd"]
        out["lengths_" + tag] = {n: {"true": tl[n], "est": estl.get(n), "ok": estl.get(n) is not None and
                                     abs(estl[n] / tl[n] - 1) <= 0.3} for n in names}
    return out


def run_c1(res, rng):
    out = {}
    for cfg in core.C1_CONFIGS:
        cid = core.cfg_id(cfg)
        out[cid] = {}
        for d in COUNT_LEVELS:
            rows, lens = unit_frames(res, cid, d)
            fc = fit_counts(rows, COUNT_LEVELS[d])
            fl = fit_lengths(lens, LEN_NESTED[d])
            out[cid][d] = {"n_rows": len(rows), "n_lengths": len(lens), "counts": fc,
                           "counts_share": shares(fc["sd"], [n for n, _ in COUNT_LEVELS[d]]),
                           "lengths": fl, "lengths_share": shares(fl["sd"], list(fl["sd"]))}
            if cid == B0_ID:
                out[cid][d]["k5"] = k5_recovery(rows, lens, d, rng)
            print("C1", cid, d, flush=True)
    return out


# ---------- §J ----------

def m3_ratio(c3b):
    """60 秒以内に揃えた比 ÷ 揃えない比。比が定義できなければ None（0 は 0 として扱う）。"""
    if not c3b:
        return None
    g = c3b.get("ratio_cs1_2019_over_klicke", {})
    a, b = g.get("all"), g.get("le60")
    if a is None or b is None or a <= 0:
        return None
    return b / a


def judge(c3, c1, c3b, boot):
    ids = list(c3)
    n = len(ids)
    b0 = c3[B0_ID]
    m1 = sum(c3[i]["rank"] != b0["rank"] for i in ids) / n
    m2 = {}
    for d, lev in MAIN_B:
        key = "%s_b" % lev
        base = b0[d]["phi"][key]
        flips = sum((c3[i][d]["phi"][key] is not None and base is not None and
                     (c3[i][d]["phi"][key] >= PHI_CUT) != (base >= PHI_CUT)) for i in ids)
        m2["%s %s" % (d, lev)] = flips / n
    m3 = m3_ratio(c3b)
    m4 = {}
    for d in DATASETS:
        low = [i for i in ids if min(c3[i][d]["agree_b0_in_cfg"] or 0, c3[i][d]["agree_cfg_in_b0"] or 0) < 0.5]
        close = [i for i in low if b0[d]["rate_pooled"] and c3[i][d]["rate_pooled"] is not None and
                 abs(c3[i][d]["rate_pooled"] / b0[d]["rate_pooled"] - 1) <= 0.2]
        m4[d] = {"low_agree_share": len(low) / n, "low_agree_with_close_rate": len(close)}
    M = {"M1_rank_changed_share": m1, "M1": m1 >= SHARE_CUT,
         "M2_flip_share": m2, "M2": any(v >= SHARE_CUT for v in m2.values()),
         "M3_ratio_le60_over_all": m3, "M3": m3 is not None and m3 <= 0.5,
         "M4": m4, "M4_any": any(v["low_agree_share"] >= SHARE_CUT and v["low_agree_with_close_rate"] > 0
                               for v in m4.values())}
    M["M_holds"] = M["M1"] or M["M2"] or M["M3"] or M["M4_any"]
    V = {}
    for d in ("CS1 2019", "CS1 2021"):
        cc = c1.get(B0_ID, {}).get(d) if c1 else None
        if not cc:
            continue
        rec = cc["k5"]
        ok = {lev: all(rec[t][lev]["ok"] for t in rec if t.startswith("counts_")) for lev, _ in COUNT_LEVELS[d]}
        sh = cc["counts_share"]
        below = sum(sh[k] for k in ("task_in_writer", "day", "block") if ok[k])
        V[d] = {"recovered": ok, "below_writer_share": below,
                "writer_share": sh["writer"] if ok["writer"] else None,
                "holds": ok["writer"] and below > sh["writer"]}
    O = {}
    for d, lev in MAIN_B:
        key = "%s_b" % lev
        O["%s %s" % (d, lev)] = {"phi": b0[d]["phi"][key], "ci": boot.get("%s|%s" % (d, key))}
    o_count = sum(1 for v in O.values() if v["phi"] is not None and v["phi"] >= PHI_CUT and v["ci"] and v["ci"][0] > 1)
    res = {"M": M, "V": V, "V_holds": any(v["holds"] for v in V.values()), "O": O, "O_count": o_count,
           "O_holds": o_count >= 2}
    if res["M"]["M_holds"] and (res["V_holds"] or res["O_holds"]):
        res["recommendation_by_rule"] = "単独にする"
    elif res["V_holds"] and res["O_holds"]:
        res["recommendation_by_rule"] = "論文 1 の改訂か論文 3 の補足に回す"
    else:
        res["recommendation_by_rule"] = "見送り"
    return res


def selftest():
    """φ が数え上げ（ポアソン）で 1 前後・過分散で 1 を超えること、M3 の 0 の扱い。"""
    fails = []
    rng = np.random.default_rng(0)
    nw, per = 200, 6
    w = np.repeat(np.arange(nw), per)
    e = rng.uniform(0.5, 2.0, nw * per)
    lam = rng.gamma(4.0, 1.0, nw)[w]
    n = rng.poisson(lam * e).astype(float)
    pb, pa = phi(w, e, n, True), phi(w, e, n, False)
    if not (0.85 < pb < 1.15):
        fails.append("ポアソンで φ(b) が 1 前後にならない: %.3f" % pb)
    if not pa > 1.5:
        fails.append("書き手の率が違うのに φ(a) が 1.5 を超えない: %.3f" % pa)
    n2 = rng.negative_binomial(2, 2 / (2 + lam * e)).astype(float)
    if not phi(w, e, n2, True) > 1.5:
        fails.append("負の二項で φ(b) が 1.5 を超えない")
    if m3_ratio({"ratio_cs1_2019_over_klicke": {"all": 4.0, "le60": 0.0}}) != 0.0:
        fails.append("M3 の比 0 を欠測にした")
    if m3_ratio({"ratio_cs1_2019_over_klicke": {"all": None, "le60": 1.0}}) is not None:
        fails.append("M3 の比が定義できないのに値を返した")
    if level_table([], "writer")[0] != []:
        fails.append("空の行で落ちる")
    print("analyze selftest:", "OK" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    return not fails


def main(cache, out, seed, *flags):
    rng = np.random.default_rng(int(seed))
    res = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    os.makedirs(out, exist_ok=True)
    c3 = {core.cfg_id(c): summarize_cfg(res, core.cfg_id(c)) for c in core.CONFIGS}
    json.dump(c3, open(os.path.join(out, "c3_multiverse.json"), "w"), indent=1, ensure_ascii=False)
    print("C3 done", flush=True)
    boot = {}
    for d in DATASETS:
        rows = block_rows(res, B0_ID, d)
        for lev, within in LEVELS[d]:
            w, e, n = level_table(rows, lev)
            boot["%s|%s_%s" % (d, lev, "b" if within else "a")] = phi_boot(w, e, n, within, rng)
    c2 = {d: {k: {"phi": c3[B0_ID][d]["phi"][k], "ci": boot["%s|%s" % (d, k)]} for k in c3[B0_ID][d]["phi"]}
          for d in DATASETS}
    json.dump(c2, open(os.path.join(out, "c2_overdispersion.json"), "w"), indent=1, ensure_ascii=False)
    print("C2 done", flush=True)
    c1 = None
    if "--skip-c1" not in flags:
        c1 = run_c1(res, rng)
        json.dump(c1, open(os.path.join(out, "c1_decomposition.json"), "w"), indent=1, ensure_ascii=False)
    p = os.path.join(cache, "c3b.json")
    c3b = json.load(open(p)) if os.path.exists(p) else None
    if c3b:
        json.dump(c3b, open(os.path.join(out, "c3b_aligned_level.json"), "w"), indent=1, ensure_ascii=False)
    j = judge(c3, c1, c3b, boot)
    json.dump(j, open(os.path.join(out, "judgement.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"recommendation_by_rule": j["recommendation_by_rule"]}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    main(*sys.argv[1:])
