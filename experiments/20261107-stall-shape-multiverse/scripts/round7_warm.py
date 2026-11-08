"""conditions-round7.md: 検出できる時間（100 個目の間隔より後）に分母をそろえた 90 通り。

usage: round7_warm.py prep <cache_dir> [jobs]          …… <cache_dir>/warm_configs.pkl（configs.pkl と同じ形・rows だけ）
       round7_warm.py analyze <data_dir> <cache_dir> <out_json> <seed> [jobs]
       round7_warm.py --selftest
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

cand = core.cand
STALL = None


def warm_point(u, seg):
    pool = sorted(cand.gap_pool(u["t"].tolist(), core.intervals_for(u, seg)))
    return pool[core.WARM - 1][0] if len(pool) >= core.WARM else math.inf


def clip(iv, w):
    return [(max(a, w), b) for a, b in iv if b > w and max(a, w) <= b]


def rows_after(u, c, w, stalls):
    """[w, ∞) に重ねた露出と、t0 ≥ w の件数のまとまりごとの行。"""
    iv = clip(core.intervals_for(u, c["seg"]), w)
    reg, days = core.regions(u)
    expo = [core.overlap(iv, a, b) for a, b in reg]
    n = [0] * len(reg)
    for s, t0, r, _ in stalls:
        if t0 >= w:
            n[core.region_index(reg, t0)] += 1
    return list(zip(days, expo, n))


def _init_prep(cache):
    global STALL
    STALL = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))


def one_unit(arg):
    i, u, check_q = arg
    s = STALL[i]
    assert s["unit"] == i
    ws = {seg: warm_point(u, seg) for seg in core.SEGS}
    out = {}
    for c in core.CONFIGS:
        cid = core.cfg_id(c)
        ref = s["cfg"][cid]["rows"]
        w = ws[c["seg"]]
        if c["theta"][0] == "fix":
            st = core.detect(u, c)
            # 確かめ: w = −∞ なら configs.pkl と一致
            if rows_after(u, c, -math.inf, st) != [tuple(x) for x in ref]:
                raise SystemExit("固定 θ の検出し直しが configs.pkl と違う: unit %d %s" % (i, cid))
            out[cid] = {"rows": rows_after(u, c, w, st)}
        else:
            if check_q:
                st = core.detect(u, c)
                if any(t0 < w for _, t0, _, _ in st):
                    raise SystemExit("本人の分位で w より前に停滞が立った: unit %d %s" % (i, cid))
            # 件数は元のまま（t0 ≥ w が規則で保証される）・露出だけ [w, ∞) に
            reg, _ = core.regions(u)
            iv = clip(core.intervals_for(u, c["seg"]), w)
            out[cid] = {"rows": [(d, core.overlap(iv, a, b), n) for (d, _, n), (a, b) in zip(ref, reg)]}
    return {"dataset": u["dataset"], "writer": u["writer"], "task": u["task"], "unit": i, "cfg": out,
            "warm_inf": {seg: math.isinf(ws[seg]) for seg in core.SEGS}}


def run_prep(cache, jobs=5):
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    rng = np.random.default_rng(7)
    check = set(rng.choice(len(units), 300, replace=False).tolist())
    res = []
    with ProcessPoolExecutor(int(jobs), initializer=_init_prep, initargs=(cache,)) as ex:
        for k, r in enumerate(ex.map(one_unit, [(i, u, i in check) for i, u in enumerate(units)], chunksize=4)):
            res.append(r)
            if k % 500 == 0:
                print("%d / %d" % (k, len(units)), flush=True)
    with open(os.path.join(cache, "warm_configs.pkl"), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("prep done: 固定 θ 36 通りは w = −∞ で configs.pkl と一致・本人の分位は 300 単位で t0 ≥ w を確かめた")


def _init_an(cache, score):
    r1.RES = pickle.load(open(os.path.join(cache, "warm_configs.pkl"), "rb"))
    r1.SCORE = score


def run_analyze(data, cache, out, seed, jobs=4):
    seed = int(seed)
    r1.RES = pickle.load(open(os.path.join(cache, "warm_configs.pkl"), "rb"))
    score = r1.load_scores(data)
    with ProcessPoolExecutor(int(jobs), initializer=_init_an, initargs=(cache, score)) as ex:
        res = dict(ex.map(r1.one, [(core.cfg_id(c), seed) for c in core.CONFIGS]))
    for v in res.values():
        v["rank"] = sorted(analyze.DATASETS, key=lambda d: -v["rate"][d])
    orig = json.load(open(os.path.join(os.path.dirname(out), "round1_extra.json")))
    o = orig["by_cfg"]
    for v in o.values():
        v["rank"] = sorted(analyze.DATASETS, key=lambda d: -v["rate"][d])
    b0 = analyze.B0_ID
    others = [c for c in res if c != b0]
    fixed = [c for c in others if "fix" in c]
    pair = "CS1 2019 / CS1 2021"

    def a1_fixed(R):
        v0 = R[b0]["ratio"][pair]
        up = v0["point"] > 1
        return sum(((R[c]["ratio"][pair]["ci"][1] < 1) if up else (R[c]["ratio"][pair]["ci"][0] > 1)) for c in fixed)

    def sgn(ci):
        return -1 if ci[1] < 0 else (1 if ci[0] > 0 else 0)
    summ = {"warm": r1.summarize(res), "orig": orig["summary"],
            "M1": {"warm": sum(res[c]["rank"] != res[b0]["rank"] for c in others),
                   "orig": sum(o[c]["rank"] != o[b0]["rank"] for c in others)},
            "A1_cs1_fixed": {"warm": a1_fixed(res), "orig": a1_fixed(o)},
            "b0_warm": {"rate": res[b0]["rate"], "score": res[b0]["score"], "rank": res[b0]["rank"]},
            "score_ci_sign_flips": sum(sgn(res[c]["score"]["ci"]) * sgn(o[c]["score"]["ci"]) == -1 for c in others),
            "score_ci_sign_changes": sum(sgn(res[c]["score"]["ci"]) != sgn(o[c]["score"]["ci"]) for c in others),
            "units_never_warm": {d: {seg: sum(r["warm_inf"][seg] for r in r1.RES if r["dataset"] == d) for seg in core.SEGS}
                                 for d in analyze.DATASETS}}
    conv = lambda x: x.item() if hasattr(x, "item") else str(x)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps({k: summ[k] for k in ("M1", "A1_cs1_fixed", "score_ci_sign_flips", "score_ci_sign_changes")},
                     ensure_ascii=False, indent=1, default=conv))
    print(json.dumps({"A1": summ["warm"]["A1"], "A3": summ["warm"]["A3"], "B": summ["warm"]["B"]},
                     ensure_ascii=False, indent=1, default=conv))


def selftest():
    ok = True
    # 1 秒ごと 200 回 → 100 個目の間隔の時刻は 100,000 ms
    base = [(i * 1000, "i") for i in range(200)]
    u = core.synth(base)
    if warm_point(u, "replay") != 100_000:
        ok = False
        print("  - w が 100 個目の間隔の時刻でない:", warm_point(u, "replay"))
    if warm_point(core.synth(base[:100]), "replay") != math.inf:
        ok = False
        print("  - 間隔 99 個で w が有限")
    if clip([(0, 50), (80, 200)], 100) != [(100, 200)] or clip([(0, 50)], 100) != []:
        ok = False
        print("  - clip が見込みと違う")
    # 固定 θ: w より前の停滞は落ちる・後は残る
    ev = [(i * 1000, "i") for i in range(20)] + [(19_000 + 60_000 + i * 1000, "i") for i in range(200)] + \
        [(278_000 + 60_000, "i")]
    uu = core.synth(ev)
    c = dict(core.B0, theta=("fix", 10_000))
    st = core.detect(uu, c)
    w = warm_point(uu, "replay")
    n_all = sum(n for _, _, n in rows_after(uu, c, -math.inf, st))
    n_w = sum(n for _, _, n in rows_after(uu, c, w, st))
    if not (len(st) == 2 and n_all == 2 and n_w == 1):
        ok = False
        print("  - w の前後の数え分けが見込みと違う:", len(st), n_all, n_w, w)
    # w の直後（25 秒後）に t0 が来る停滞は残る: 101 回の押下（w = 100,000）→ W 15 秒・固定 10 秒で t0 = 125,000
    ev2 = [(i * 1000, "i") for i in range(101)] + [(100_000 + 90_000, "i")]
    u2 = core.synth(ev2)
    c2 = dict(core.B0, W=15_000, theta=("fix", 10_000))
    st2 = core.detect(u2, c2)
    w2 = warm_point(u2, "replay")
    if not (w2 == 100_000 and len(st2) == 1 and sum(n for _, _, n in rows_after(u2, c2, w2, st2)) == 1):
        ok = False
        print("  - w の直後の停滞が残らない:", w2, st2)
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
