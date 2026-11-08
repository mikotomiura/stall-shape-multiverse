"""conditions-round9.md: 観測区間の始めの 60 秒と 100 個目の間隔より前を外した共通の基準（感度）と、始めの W 秒の割合。

usage: round9_wstart.py prep <cache_dir> [jobs]          …… <cache_dir>/wstart_configs.pkl と <cache_dir>/wstart_shares.json
       round9_wstart.py analyze <data_dir> <cache_dir> <out_json> <seed> [jobs]
       round9_wstart.py --selftest
集計値だけを repo に書く。
"""
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import core  # noqa: E402
import round1_extra as r1  # noqa: E402
import round7_warm as r7  # noqa: E402
import round8_causal as r8  # noqa: E402

CUT = 60_000
STALL = None


def cut_iv(iv, w, cut=CUT):
    out = []
    for a, b in iv:
        x = max(a + cut, w)
        if x < b:
            out.append((x, b))
    return out


def rows_cut(u, ivc, stalls):
    """切った区間 ivc に重ねた露出と、t0 が ivc の中にある件数のまとまりごとの行。"""
    reg, days = core.regions(u)
    expo = [core.overlap(ivc, a, b) for a, b in reg]
    n = [0] * len(reg)
    for s, t0, r, _ in stalls:
        if any(x <= t0 <= b for x, b in ivc):
            n[core.region_index(reg, t0)] += 1
    return list(zip(days, expo, n))


def wshare(iv, W):
    """各区間の始めの W（区間が短ければ全体）の合計と、区間の合計。"""
    return sum(min(W, b - a) for a, b in iv), sum(b - a for a, b in iv)


def _init_prep(cache):
    global STALL
    STALL = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))


def one_unit(arg):
    i, u = arg
    s = STALL[i]
    assert s["unit"] == i
    ivs = {seg: core.intervals_for(u, seg) for seg in core.SEGS}
    ws = {seg: r7.warm_point(u, seg) for seg in core.SEGS}
    out = {}
    for c in core.CONFIGS:
        cid = core.cfg_id(c)
        ref = s["cfg"][cid]["rows"]
        st = core.detect(u, c)
        iv = ivs[c["seg"]]
        if rows_cut(u, cut_iv(iv, -math.inf, 0), st) != [tuple(x) for x in ref]:
            raise SystemExit("切らない検出し直しが configs.pkl と違う: unit %d %s" % (i, cid))
        out[cid] = {"rows": rows_cut(u, cut_iv(iv, ws[c["seg"]]), st)}
    sh = {seg: {W: wshare(ivs[seg], W) for W in core.WS} for seg in core.SEGS}
    return {"dataset": u["dataset"], "writer": u["writer"], "task": u["task"], "unit": i, "cfg": out, "share": sh}


def run_prep(cache, jobs=6):
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    res = []
    with ProcessPoolExecutor(int(jobs), initializer=_init_prep, initargs=(cache,)) as ex:
        for k, r in enumerate(ex.map(one_unit, [(i, u) for i, u in enumerate(units)], chunksize=4)):
            res.append(r)
            if k % 500 == 0:
                print("%d / %d" % (k, len(units)), flush=True)
    shares = {}
    for d in analyze.DATASETS:
        shares[d] = {}
        for seg in core.SEGS:
            shares[d][seg] = {}
            for W in core.WS:
                a = sum(r["share"][seg][W][0] for r in res if r["dataset"] == d)
                b = sum(r["share"][seg][W][1] for r in res if r["dataset"] == d)
                shares[d][seg][str(W // 1000)] = a / b if b else None
    for r in res:
        del r["share"]
    with open(os.path.join(cache, "wstart_configs.pkl"), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(shares, open(os.path.join(cache, "wstart_shares.json"), "w"), indent=1)
    print("prep done: 切らない検出し直しは 90 通り × %d 単位で configs.pkl と一致" % len(res))
    print(json.dumps(shares, indent=1))


def _init_an(cache, score):
    r1.RES = pickle.load(open(os.path.join(cache, "wstart_configs.pkl"), "rb"))
    r1.SCORE = score


def run_analyze(data, cache, out, seed, jobs=4):
    seed = int(seed)
    r1.RES = pickle.load(open(os.path.join(cache, "wstart_configs.pkl"), "rb"))
    score = r1.load_scores(data)
    with ProcessPoolExecutor(int(jobs), initializer=_init_an, initargs=(cache, score)) as ex:
        res = dict(ex.map(r1.one, [(core.cfg_id(c), seed) for c in core.CONFIGS]))
    for v in res.values():
        v["rank"] = sorted(analyze.DATASETS, key=lambda d: -v["rate"][d])
    o = json.load(open(os.path.join(os.path.dirname(out), "round1_extra.json")))["by_cfg"]
    for v in o.values():
        v["rank"] = sorted(analyze.DATASETS, key=lambda d: -v["rate"][d])
    b0 = analyze.B0_ID
    others = [c for c in res if c != b0]

    def sgn(ci):
        return -1 if ci[1] < 0 else (1 if ci[0] > 0 else 0)
    shares = json.load(open(os.path.join(cache, "wstart_shares.json")))
    summ = {"shares": shares,
            "recount_cut": r8.recount(res, {}), "recount_orig": r8.recount(o, {}),
            "M1": {"cut": sum(res[c]["rank"] != res[b0]["rank"] for c in others),
                   "orig": sum(o[c]["rank"] != o[b0]["rank"] for c in others)},
            "b0_cut": {"rate": res[b0]["rate"], "score": res[b0]["score"], "rank": res[b0]["rank"]},
            "score_ci_sign_flips": sum(sgn(res[c]["score"]["ci"]) * sgn(o[c]["score"]["ci"]) == -1 for c in others),
            "score_ci_sign_changes": sum(sgn(res[c]["score"]["ci"]) != sgn(o[c]["score"]["ci"]) for c in others)}
    conv = lambda x: x.item() if hasattr(x, "item") else str(x)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps({k: summ[k] for k in ("M1", "score_ci_sign_flips", "score_ci_sign_changes", "recount_cut",
                                           "recount_orig")}, ensure_ascii=False, indent=1, default=conv))


def selftest():
    ok = True
    if cut_iv([(0, 100_000), (200_000, 230_000)], -math.inf) != [(60_000, 100_000)]:
        ok = False
        print("  - 60 秒の切り方が見込みと違う")
    if cut_iv([(0, 300_000)], 100_000) != [(100_000, 300_000)]:
        ok = False
        print("  - w が 60 秒より後のとき w で切らない")
    if cut_iv([(0, 50_000)], -math.inf, 0) != [(0, 50_000)]:
        ok = False
        print("  - 切らないときに区間が変わる")
    if wshare([(0, 100_000), (0, 10_000)], 30_000) != (40_000, 110_000):
        ok = False
        print("  - 割合の計算（短い区間）が見込みと違う")
    # 1 秒ごとの押下の後に 90 秒の空白: W 15 秒・固定 10 秒で s = 押下の 15 秒後、t0 = 25 秒後
    ev = [(i * 1000, "i") for i in range(30)] + [(29_000 + 90_000, "i")]
    u = core.synth(ev)
    c = dict(core.B0, W=15_000, theta=("fix", 10_000))
    st = core.detect(u, c)
    iv = core.intervals_for(u, "replay")
    n_all = sum(n for _, _, n in rows_cut(u, cut_iv(iv, -math.inf, 0), st))
    t0 = st[0][1] if st else None
    # t0 = 29,000 + 25,000 = 54,000 < 60,000 → 共通の基準では落ちる
    n_cut = sum(n for _, _, n in rows_cut(u, cut_iv(iv, -math.inf), st))
    if not (len(st) == 1 and t0 == 54_000 and n_all == 1 and n_cut == 0):
        ok = False
        print("  - 60 秒より前の停滞の数え分けが見込みと違う:", st, n_all, n_cut)
    # 同じ形で始まりを 40 秒ずらすと t0 = 94,000 ≥ 60,000 → 残る
    ev2 = [(i * 1000, "i") for i in range(70)] + [(69_000 + 90_000, "i")]
    u2 = core.synth(ev2)
    st2 = core.detect(u2, c)
    n2 = sum(n for _, _, n in rows_cut(u2, cut_iv(core.intervals_for(u2, "replay"), -math.inf), st2))
    if not (len(st2) == 1 and st2[0][1] == 94_000 and n2 == 1):
        ok = False
        print("  - 60 秒より後の停滞が残らない:", st2, n2)
    print("selftest:", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    elif a[0] == "prep":
        run_prep(*a[1:])
    elif a[0] == "analyze":
        run_analyze(*a[1:])
