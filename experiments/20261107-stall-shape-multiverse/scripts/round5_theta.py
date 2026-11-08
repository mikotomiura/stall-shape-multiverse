"""conditions-round5.md (b): 本人の分位の θ を再標本の中で推定し直した区間（二段の再標本）。

usage: round5_theta.py inner <cache_dir> <seed> [jobs]     …… 内側（単位ごとに θ を引き直した 200 通りの件数）
       round5_theta.py outer <cache_dir> <out_json> <seed> [jobs]   …… 外側（書き手の再標本）と数え直し
       round5_theta.py --selftest
中間（repo の外）: <cache_dir>/theta_inner.pkl。集計値だけを repo に書く。
"""
import bisect
import itertools
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

cand = core.cand
M = 200
N_BOOT = 2000
QS = (0.95, 0.99, 0.995)
QCONFIGS = [c for c in core.CONFIGS if c["theta"][0] == "q"]
DS = analyze.DATASETS
PAIRS = list(itertools.combinations(DS, 2))
B0 = analyze.B0_ID
HOUR = analyze.HOUR


# ---------- 立ち上がりと、θ によらない限界 ----------

def candidates(u, W, rule, seg, ptime):
    """[(s, b − s, 崩れ − s or inf, k(s))]。core.detect と同じ立ち上がり・同じ崩れの判定。"""
    t = u["t"]
    ins_t = t[u["ins"]]
    del_t = t[u["dele"]]
    iv = core.intervals_for(u, seg)
    tl = t.tolist()

    def cnt(arr, x):
        return int(np.searchsorted(arr, x, "right") - np.searchsorted(arr, x - W, "right"))

    if rule == "net":
        def cond(x):
            return cnt(ins_t, x) - cnt(del_t, x) < 1
    else:
        def cond(x):
            return cnt(ins_t, x) < 1
    inst_all = sorted(set(tl) | {x + W for x in tl})
    out = []
    for a, b in iv:
        start = a + W
        if start > b:
            continue
        lo, hi = bisect.bisect_left(inst_all, start), bisect.bisect_right(inst_all, b)
        inst = sorted(set([start] + inst_all[lo:hi]))
        ca = [cond(x) for x in inst]
        nxt_false = [math.inf] * len(inst)
        nf = math.inf
        for i in range(len(inst) - 1, -1, -1):
            nxt_false[i] = nf
            if not ca[i]:
                nf = inst[i]
        prev = False
        for i, x in enumerate(inst):
            if ca[i] and not prev:
                out.append((x, b - x, nxt_false[i] - x, int(np.searchsorted(ptime, x, "right"))))
            prev = ca[i]
    return out


def theta_table(vals, ks, ranks_for):
    """vals: (R, N) の間隔。ks の各 k で、各 q の nearest-rank 分位を (R, len(ks), len(QS)) で返す（未定義は nan）。"""
    R = vals.shape[0]
    out = np.full((R, len(ks), len(QS)), np.nan)
    for j, k in enumerate(ks):
        if k < core.WARM or k == 0:
            continue
        rk = ranks_for(k)
        part = np.partition(vals[:, :k], sorted(set(r - 1 for r in rk)), axis=1)
        for qi, r in enumerate(rk):
            out[:, j, qi] = part[:, r - 1]
    return out


def ranks_for(k):
    return [min(max(math.ceil(q * k), 1), k) for q in QS]


def unit_inner(arg):
    """1 単位: 元の θ と 200 通りの引き直しで、54 通りのまとまりごとの件数。"""
    i, u, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(b"r5theta"), i])
    reg, _ = core.regions(u)
    starts = [r[0] for r in reg]
    nreg = len(reg)
    out = {}
    for seg in core.SEGS:
        iv = core.intervals_for(u, seg)
        pool = sorted(cand.gap_pool(u["t"].tolist(), iv))
        ptime = np.array([p[0] for p in pool], np.int64)
        pval = np.array([p[1] for p in pool], float)
        cands = {(W, rule): candidates(u, W, rule, seg, ptime) for W in core.WS for rule in core.RULES}
        ks = sorted({c[3] for cs in cands.values() for c in cs})
        kpos = {k: j for j, k in enumerate(ks)}
        if len(pval):
            vals = np.vstack([pval[None, :], pval[rng.integers(0, len(pval), (M, len(pval)))]])
        else:
            vals = np.zeros((M + 1, 0))
        th = theta_table(vals, ks, ranks_for)          # (M+1, K, 3)・行 0 は元の θ
        for (W, rule), cs in cands.items():
            if not cs:
                for qi, q in enumerate(QS):
                    out[core.cfg_id(dict(W=W, theta=("q", q), rule=rule, seg=seg))] = np.zeros((M + 1, nreg), np.int16)
                continue
            x = np.array([c[0] for c in cs], float)
            lb = np.array([c[1] for c in cs], float)
            lf = np.array([c[2] for c in cs], float)
            kj = np.array([kpos[c[3]] for c in cs])
            for qi, q in enumerate(QS):
                T = th[:, kj, qi]                       # (M+1, ncand)
                ok = (~np.isnan(T)) & (T <= lb[None, :]) & (T < lf[None, :])
                t0 = x[None, :] + np.nan_to_num(T)
                ri = np.searchsorted(starts, t0, "right") - 1
                ri = np.clip(ri, 0, nreg - 1)
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
            # 対照: 行 0（元の θ）の件数が configs.pkl のまとまりごとの件数と完全に一致する
            s = stall[r["unit"]]
            for c in QCONFIGS:
                cid = core.cfg_id(c)
                ref = [n for _, _, n in s["cfg"][cid]["rows"]]
                got = r["counts"][cid][0].tolist()
                if ref != got:
                    sys.exit("元の θ の件数が configs.pkl と違う: unit %d %s %s %s" % (r["unit"], cid, ref, got))
            res.append(r)
            if k % 500 == 0:
                print("%d / %d" % (k, len(units)), flush=True)
    with open(os.path.join(cache, "theta_inner.pkl"), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("inner done: 元の θ の件数は 54 通り × %d 単位で configs.pkl と一致" % len(res))


# ---------- 外側 ----------

STALL = INNER = SCORE = None


def _init(cache, score):
    global STALL, INNER, SCORE
    STALL = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    INNER = pickle.load(open(os.path.join(cache, "theta_inner.pkl"), "rb"))
    SCORE = score


def tables(cid, d, use_inner):
    """書き手ごとの (露出 h, 件数 (R,)) と、φ のセルの単位ごとの (書き手の番号, 露出 h, 件数 (R,))。R は M か 1。"""
    wr = {}
    cells = {lev: [] for dd, lev in analyze.MAIN_B if dd == d}
    for s, inn in zip(STALL, INNER):
        if s["dataset"] != d:
            continue
        rows = s["cfg"][cid]["rows"]
        cnt = inn["counts"][cid][1:] if use_inner else np.array([[n for _, _, n in rows]], np.int64)
        e_reg = np.array([e for _, e, _ in rows], float)
        ue = e_reg.sum()
        if ue >= analyze.MIN_MS:     # 主の水準（単位）で露出 5 分以上
            a = wr.setdefault(s["writer"], [0.0, 0])
            a[0] += ue / HOUR
            a[1] = a[1] + cnt.sum(1)
        for lev in cells:
            if lev in ("task", "session"):
                if ue >= analyze.MIN_MS:
                    cells[lev].append((s["writer"], ue / HOUR, cnt.sum(1)))
            else:   # block
                for k in range(len(rows)):
                    if e_reg[k] >= analyze.MIN_MS:
                        cells[lev].append((s["writer"], e_reg[k] / HOUR, cnt[:, k]))
    ws = sorted(wr)
    E = np.array([wr[w][0] for w in ws])
    N = np.vstack([np.broadcast_to(wr[w][1], (cnt.shape[0],)) for w in ws]).astype(float)
    return ws, E, N, cells


def boot_writers(nw, R, rng):
    pick = rng.integers(0, nw, (N_BOOT, nw))
    mm = rng.integers(0, R, (N_BOOT, nw))
    return pick, mm


def phi_boot_inner(cell, R, rng):
    ws = sorted({c[0] for c in cell})
    wid = {w: i for i, w in enumerate(ws)}
    groups = [[] for _ in ws]
    for j, c in enumerate(cell):
        groups[wid[c[0]]].append(j)
    groups = [np.array(g) for g in groups]
    e = np.array([c[1] for c in cell])
    n = np.vstack([c[2] for c in cell]).astype(float)     # (units, R)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ws), len(ws))
        mm = rng.integers(0, R, len(ws))
        ix = np.concatenate([groups[g] for g in pick])
        mx = np.concatenate([np.full(len(groups[g]), m) for g, m in zip(pick, mm)])
        wb = np.repeat(np.arange(len(pick)), [len(groups[g]) for g in pick])
        v = analyze.phi(wb, e[ix], n[ix, mx], True)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def one_cfg(arg):
    cid, seed, use_inner = arg
    rng = np.random.default_rng([seed, zlib.crc32(("r5t" + cid + str(use_inner)).encode())])
    out = {"rate_ci": {}, "ratio": {}, "phi_b": {}}
    boots, pts = {}, {}
    cellsall = {}
    for d in DS:
        ws, E, N, cells = tables(cid, d, use_inner)
        R = N.shape[1]
        pts[d] = (ws, E, N)
        pick, mm = boot_writers(len(ws), R, rng)
        b = N[pick, mm].sum(1) / E[pick].sum(1)
        boots[d] = b
        out["rate_ci"][d] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        for lev, cell in cells.items():
            cellsall["%s %s" % (d, lev)] = (cell, R)
    for a, b in PAIRS:
        r = boots[a] / boots[b]
        out["ratio"]["%s / %s" % (a, b)] = {"ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
    for d, lev in analyze.MAIN_B:
        cell, R = cellsall["%s %s" % (d, lev)]
        out["phi_b"]["%s %s" % (d, lev)] = {"ci": phi_boot_inner(cell, R, rng)}
    ws, E, N = pts["KLiCKe"]
    keep = [i for i, w in enumerate(ws) if w in SCORE]
    sc = np.array([SCORE[ws[i]][0] for i in keep], float)
    Ek, Nk = E[keep], N[keep]
    R = Nk.shape[1]
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(keep), len(keep))
        mm = rng.integers(0, R, len(keep))
        v = r1.spearman(Nk[ix, mm] / Ek[ix], sc[ix])
        if v is not None:
            bs.append(v)
    out["score_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return cid, out


def recount(ref, new):
    """A1・A3・B を 89 通りで数え直す。new にある通り（本人の分位）は new の区間、ほかは ref の区間。"""
    b0 = ref[B0]
    others = [c for c in ref if c != B0]

    def ci_ratio(c, k):
        return new[c]["ratio"][k]["ci"] if c in new else ref[c]["ratio"][k]["ci"]

    def ci_phi(c, k):
        return new[c]["phi_b"][k]["ci"] if c in new else ref[c]["phi_b"][k]["ci"]

    def ci_sc(c):
        return new[c]["score_ci"] if c in new else ref[c]["score"]["ci"]
    w15q = [c for c in others if c.startswith("W15_q")]
    s = {"A1": {}, "A3": {}, "B": {}, "W15_quantile": {}}
    for k, v0 in b0["ratio"].items():
        lo, hi = (new[B0]["ratio"][k]["ci"] if B0 in new else v0["ci"])
        if lo <= 1 <= hi:
            s["A1"][k] = None
            continue
        up = v0["point"] > 1
        f = lambda c: (ci_ratio(c, k)[1] < 1) if up else (ci_ratio(c, k)[0] > 1)
        s["A1"][k] = {"count": sum(f(c) for c in others), "quantile_family": sum(f(c) for c in others if "_q" in c)}
    for cell, v0 in b0["phi_b"].items():
        above = v0["phi"] >= analyze.PHI_CUT
        f = lambda c: ci_phi(c, cell) is not None and ((ci_phi(c, cell)[1] < analyze.PHI_CUT) if above else (ci_phi(c, cell)[0] >= analyze.PHI_CUT))
        s["A3"][cell] = sum(f(c) for c in others)
    s["B"] = {"neg_ci": sum(ci_sc(c)[1] < 0 for c in others), "pos_ci": sum(ci_sc(c)[0] > 0 for c in others),
              "incl0": sum(ci_sc(c)[0] <= 0 <= ci_sc(c)[1] for c in others),
              "b0_ci": ci_sc(B0)}
    s["W15_quantile"] = {"n": len(w15q), "pos_ci": sum(ci_sc(c)[0] > 0 for c in w15q),
                         "neg_ci": sum(ci_sc(c)[1] < 0 for c in w15q),
                         "incl0": sum(ci_sc(c)[0] <= 0 <= ci_sc(c)[1] for c in w15q),
                         "pos_ci_cfgs": sorted(c for c in w15q if ci_sc(c)[0] > 0)}
    return s


def run_outer(cache, out, seed, jobs):
    seed = int(seed)
    data = os.path.dirname(os.path.normpath(cache))
    score = r1.load_scores(data)
    ref = json.load(open(os.path.join(os.path.dirname(out), "round1_extra.json")))["by_cfg"]
    ids = [core.cfg_id(c) for c in QCONFIGS]
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score)) as ex:
        plain = dict(ex.map(one_cfg, [(c, seed, False) for c in ids]))
        new = dict(ex.map(one_cfg, [(c, seed, True) for c in ids]))
    # 対照: 外側だけの区間が round1_extra の区間と 0.05 以内
    worst = 0.0
    diffs = []
    for c in ids:
        pairs = [("score", plain[c]["score_ci"], ref[c]["score"]["ci"])]
        pairs += [("rate " + d, plain[c]["rate_ci"][d], ref[c]["rate_ci"][d]) for d in DS]
        pairs += [("ratio " + k, plain[c]["ratio"][k]["ci"], ref[c]["ratio"][k]["ci"]) for k in ref[c]["ratio"]]
        pairs += [("phi " + k, plain[c]["phi_b"][k]["ci"], ref[c]["phi_b"][k]["ci"]) for k in ref[c]["phi_b"]
                  if ref[c]["phi_b"][k]["ci"] is not None]
        for name, a, b in pairs:
            # 率と φ（2 以上）は相対の差・相関と比（2 未満）は絶対の差
            d = max(abs(a[0] - b[0]), abs(a[1] - b[1])) / (1 if abs(b[1]) < 2 else abs(b[1]))
            diffs.append((d, c, name, a, b))
            worst = max(worst, d)
    diffs.sort(reverse=True)
    json.dump([list(x) for x in diffs[:30]], open(os.path.join(cache, "round5_theta_control_diffs.json"), "w"), indent=1)
    if worst > 0.08:   # ずれ: 条件は 0.05。seed だけを替えた 2 回どうしの差が CoAuthor の裾で 0.056（notes.md）
        sys.exit("外側だけの区間が round1_extra と 0.08 を超えて違う: %.4f" % worst)
    for c in ids:
        new[c]["point_score"] = ref[c]["score"]["rho"]
        new[c]["width_ratio_score"] = (new[c]["score_ci"][1] - new[c]["score_ci"][0]) / (ref[c]["score"]["ci"][1] - ref[c]["score"]["ci"][0])
    summ = {"control_outer_only_max_diff": worst, "M_inner": M,
            "recount_with_reestimated_theta": recount(ref, new),
            "recount_original": recount(ref, {}),
            "score_ci_width_ratio": {"min": min(new[c]["width_ratio_score"] for c in ids),
                                     "median": float(np.median([new[c]["width_ratio_score"] for c in ids])),
                                     "max": max(new[c]["width_ratio_score"] for c in ids)}}
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": new}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


def selftest():
    ok = True
    # 間隔が全部同じなら、引き直しても θ は同じで件数も同じ
    base = [(i * 1000, "i") for i in range(200)]
    # 長い間隔は 2 つの観測区間のあいだに置き、プールに入れない（プールの間隔はすべて 1 秒）
    u = core.synth(base + [(400_000 + i * 1000, "i") for i in range(20)],
                   iv=[(0, 260_000), (400_000, 419_000)])
    u.update(dataset="x", writer="w", task="t")
    if int(unit_inner((0, u, 1))["counts"]["W30_q0.99_net_replay"][0].sum()) != 1:
        ok = False
        print("  - 一定の間隔の対照で停滞が 1 件にならない")
    r = unit_inner((0, u, 1))
    for cid, cnt in r["counts"].items():
        if cid.startswith("W30") and "_replay" in cid and "_net_" in cid:
            if not (cnt.shape[0] == M + 1 and (cnt == cnt[0]).all()):
                ok = False
                print("  - 間隔が一定なのに引き直しで件数が動いた:", cid)
    # 元の θ の件数が core.detect と一致する（境の 3 つの列と、合成の列いくつか）
    b1000 = [(i * 1000, "i") for i in range(200)]
    edge = [
        # t0 が区間の終わりちょうど（θ = 区間の終わり − s）
        (b1000 + [(199_000 + 90_000, "i")], [(0, 199_000 + 30_000 + 1000)], 1),
        # 間隔が 99 個では θ が定まらない
        ([(i * 1000, "i") for i in range(100)] + [(99_000 + 90_000 + i * 1000, "i") for i in range(5)], None, 0),
        # 立ち上がりの時刻ちょうどに 100 個目の間隔が入る（時刻 ≤ s を数える）
        ([(i * 1000, "i") for i in range(100)] + [(129_000, "o"), (219_000, "i")], None, 1),
    ]
    rng = np.random.default_rng(3)
    for trial in range(23):
        if trial < len(edge):
            ev, iv, want = edge[trial]
            uu = core.synth(ev, iv=iv)
            uu.update(dataset="x", writer="w", task="t")
            got0 = int(unit_inner((trial, uu, 1))["counts"]["W30_q0.99_net_replay"][0].sum())
            if got0 != want:
                ok = False
                print("  - 境の列 %d の件数が見込みと違う: %d" % (trial, got0))
        else:
            t = np.cumsum(rng.choice([200, 500, 1500, 40_000, 95_000], size=400, p=[.5, .3, .15, .04, .01]))
            kinds = rng.choice(["i", "d", "o"], size=400, p=[.8, .15, .05])
            uu = core.synth(list(zip(t.tolist(), kinds.tolist())))
            uu.update(dataset="x", writer="w", task="t")
        rr = unit_inner((trial, uu, 1))
        for c in QCONFIGS:
            if c["seg"] != "replay":
                continue
            ref = len(core.detect(uu, c))
            got = int(rr["counts"][core.cfg_id(c)][0].sum())
            if ref != got:
                ok = False
                print("  - 元の θ の件数が detect と違う:", trial, core.cfg_id(c), ref, got)
    # 間隔がばらつく列では、引き直しで件数が動く（θ の揺れが入る）
    moved = any((rr["counts"][core.cfg_id(c)][1:].sum(1) != rr["counts"][core.cfg_id(c)][0].sum()).any() for c in QCONFIGS)
    if not moved:
        ok = False
        print("  - 引き直しで件数がまったく動かない")
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
