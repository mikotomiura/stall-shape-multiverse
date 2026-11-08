"""conditions-round5.md (a): 休止の系統（隣り合う押下の間隔が閾値を超える）を 36 通りで振る。

usage: round5_pause.py <data_dir> <cache_dir> <out_json> <seed> [jobs]
       round5_pause.py --selftest
中間（repo の外）: <cache_dir>/pause_configs.pkl …… configs.pkl と同じ形（rows だけ）。
集計値だけを書く（書き手ごとの値・ID・本文を書かない）。
"""
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
from round2_press import partial_rank  # noqa: E402
from round4_speedvar import unit_speeds  # noqa: E402

N_BOOT = 2000
MIN_GAPS = 100
KINDS = ("all", "char")
THRS = (("fix", 2000), ("fix", 5000), ("fix", 10_000), ("q", 0.95), ("q", 0.99), ("q", 0.995))
PCONFIGS = [dict(kind=k, thr=t, seg=s) for k, t, s in itertools.product(KINDS, THRS, core.SEGS)]
DS = analyze.DATASETS
PAIRS = list(itertools.combinations(DS, 2))
STALL_B0 = analyze.B0_ID
RES = SCORE = SPEED = None


def pid(c):
    th = "fix%ds" % (c["thr"][1] // 1000) if c["thr"][0] == "fix" else "q%g" % c["thr"][1]
    return "P_%s_%s_%s" % (c["kind"], th, c["seg"])


def gaps_in(u, c):
    """同じ観測区間の中の、その種類の隣り合う押下の (始まりの時刻, 間隔) の配列。"""
    t = np.asarray(u["t"], np.int64)
    if c["kind"] == "char":
        t = t[np.asarray(u["ins"], bool)]
    st, gp = [], []
    for a, b in core.intervals_for(u, c["seg"]):
        lo, hi = np.searchsorted(t, a, "left"), np.searchsorted(t, b, "right")
        x = t[lo:hi]
        if len(x) > 1:
            st.append(x[:-1])
            gp.append(np.diff(x))
    if not st:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(st), np.concatenate(gp)


def threshold(gaps, thr):
    if thr[0] == "fix":
        return thr[1]
    n = len(gaps)
    if n < MIN_GAPS:
        return None
    rank = min(max(math.ceil(thr[1] * n), 1), n)
    return int(np.partition(gaps, rank - 1)[rank - 1])


def detect_pauses(u, c):
    """戻り値: 休止の始まりの時刻の列（閾値が定まらなければ None）。"""
    st, gp = gaps_in(u, c)
    th = threshold(gp, c["thr"])
    if th is None:
        return None
    return st[gp > th]


def one_unit(arg):
    i, u = arg
    assert np.all(np.diff(u["t"]) >= 0), "押下が時刻の順でない"
    out = {}
    for c in PCONFIGS:
        p = detect_pauses(u, c)
        if p is None:
            out[pid(c)] = {"rows": [], "undefined": True}
            continue
        rows, _ = core.unit_rows(u, [(int(x), int(x), None, None) for x in p], c)
        out[pid(c)] = {"rows": rows, "undefined": False}
    return {"dataset": u["dataset"], "writer": u["writer"], "task": u["task"], "unit": i, "cfg": out}


def _init(cache, score, speed):
    global RES, SCORE, SPEED
    RES = pickle.load(open(os.path.join(cache, "pause_configs.pkl"), "rb"))
    SCORE, SPEED = score, speed


def score_block(ws, e, n, rng):
    keep = [i for i, w in enumerate(ws) if w in SCORE]
    rate = n[keep] / e[keep]
    sc = np.array([SCORE[ws[i]][0] for i in keep], float)
    rho = r1.spearman(rate, sc)
    keep2 = [j for j, i in enumerate(keep) if SPEED.get(ws[i]) is not None]
    sp = np.array([SPEED[ws[keep[j]]] for j in keep2], float)
    r2, s2 = rate[keep2], sc[keep2]
    pr = partial_rank(s2, r2, sp)
    bs, bp = [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(keep), len(keep))
        v = r1.spearman(rate[ix], sc[ix])
        if v is not None:
            bs.append(v)
        jx = rng.integers(0, len(keep2), len(keep2))
        v = partial_rank(s2[jx], r2[jx], sp[jx])
        if v is not None:
            bp.append(v)
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n": len(keep), "rho": rho, "ci": pc(bs), "n_speed": len(keep2), "partial_speed": pr, "partial_ci": pc(bp),
            "zero_share": float(np.mean(n[keep] == 0))}


def one_cfg(arg):
    cid, seed = arg
    rng = np.random.default_rng([seed, zlib.crc32(("r5p" + cid).encode())])
    out = {"rate": {}, "rate_ci": {}, "ratio": {}, "phi_b": {}, "undefined_units": {}}
    boots = {}
    for d in DS:
        out["undefined_units"][d] = sum(r["cfg"][cid]["undefined"] for r in RES if r["dataset"] == d)
        ws, e, n = r1.writer_sums(RES, cid, d)
        out["rate"][d] = float(n.sum() / e.sum())
        b = r1.boot_rate(e, n, rng)
        boots[d] = b
        out["rate_ci"][d] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out["rank"] = sorted(DS, key=lambda d: -out["rate"][d])
    for a, b in PAIRS:
        r = boots[a] / boots[b]
        out["ratio"]["%s / %s" % (a, b)] = {"point": out["rate"][a] / out["rate"][b],
                                            "ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
    for d, lev in analyze.MAIN_B:
        rows = analyze.block_rows(RES, cid, d)
        w, e, n = analyze.level_table(rows, lev)
        out["phi_b"]["%s %s" % (d, lev)] = {"phi": analyze.phi(w, e, n, True),
                                            "ci": analyze.phi_boot(w, e, n, True, rng)}
    ws, e, n = r1.writer_sums(RES, cid, "KLiCKe")
    out["score"] = score_block(ws, e, n, rng)
    return cid, out


def summarize(res, b0):
    ids = list(res)
    s = {"n_defs": len(ids), "stall_b0_rank": b0["rank"], "PA_rank_differs": sum(res[c]["rank"] != b0["rank"] for c in ids),
         "PB": {}, "PC": {}, "PD": {}}
    for k, v0 in b0["ratio"].items():
        lo, hi = v0["ci"]
        if lo <= 1 <= hi:
            s["PB"][k] = {"count": None, "note": "停滞の基準の向きが定まらない"}
            continue
        up = v0["point"] > 1
        s["PB"][k] = {"count": sum((res[c]["ratio"][k]["ci"][1] < 1) if up else (res[c]["ratio"][k]["ci"][0] > 1) for c in ids),
                      "point_count": sum((res[c]["ratio"][k]["point"] < 1) if up else (res[c]["ratio"][k]["point"] > 1) for c in ids),
                      "range": [min(res[c]["ratio"][k]["point"] for c in ids), max(res[c]["ratio"][k]["point"] for c in ids)]}
    for cell, v0 in b0["phi_b"].items():
        above = v0["phi"] >= analyze.PHI_CUT
        vals = [res[c]["phi_b"][cell] for c in ids]
        s["PC"][cell] = {"count": sum(v["ci"] is not None and ((v["ci"][1] < analyze.PHI_CUT) if above else (v["ci"][0] >= analyze.PHI_CUT))
                                      for v in vals),
                         "ci_above1": sum(v["ci"] is not None and v["ci"][0] > 1 for v in vals),
                         "range": [min(v["phi"] for v in vals if v["phi"] is not None), max(v["phi"] for v in vals if v["phi"] is not None)]}
    for key, ck in (("raw", ("rho", "ci")), ("speed", ("partial_speed", "partial_ci"))):
        vals = [res[c]["score"] for c in ids]
        s["PD"][key] = {"neg_ci": sum(v[ck[1]][1] < 0 for v in vals), "pos_ci": sum(v[ck[1]][0] > 0 for v in vals),
                        "incl0": sum(v[ck[1]][0] <= 0 <= v[ck[1]][1] for v in vals),
                        "min": min(v[ck[0]] for v in vals), "median": float(np.median([v[ck[0]] for v in vals])),
                        "max": max(v[ck[0]] for v in vals)}
        by = {}
        for fam, pred in (("fixed", lambda c: "fix" in c), ("quantile", lambda c: "_q" in c),
                          ("all", lambda c: c.startswith("P_all")), ("char", lambda c: c.startswith("P_char"))):
            vv = [res[c]["score"] for c in ids if pred(c)]
            by[fam] = {"n": len(vv), "neg_ci": sum(v[ck[1]][1] < 0 for v in vv), "pos_ci": sum(v[ck[1]][0] > 0 for v in vv),
                       "min": min(v[ck[0]] for v in vv), "max": max(v[ck[0]] for v in vv)}
        s["PD"][key]["by_family"] = by
    s["zero_share_range"] = [min(res[c]["score"]["zero_share"] for c in ids), max(res[c]["score"]["zero_share"] for c in ids)]
    s["undefined_units"] = {c: res[c]["undefined_units"] for c in ids if any(res[c]["undefined_units"].values())}
    s["rates"] = {d: [min(res[c]["rate"][d] for c in ids), max(res[c]["rate"][d] for c in ids)] for d in DS}
    return s


def selftest():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("  -", msg)
    # 1 秒ごとに 150 回 → 2 秒ちょうどの間隔 → 3 秒の間隔
    ev = [(i * 1000, "i") for i in range(150)]
    ev += [(149_000 + 2000, "i"), (149_000 + 2000 + 3000, "i")]
    u = core.synth(ev)
    p = detect_pauses(u, dict(kind="all", thr=("fix", 2000), seg="replay"))
    chk(list(p) == [151_000], "閾値ちょうど（2 秒）を数えた、または 3 秒を数えない: %s" % (p,))
    # 区間をまたぐ間隔は数えない
    u2 = core.synth([(i * 1000, "i") for i in range(10)] + [(100_000 + i * 1000, "i") for i in range(10)],
                    iv=[(0, 9000), (100_000, 109_000)])
    chk(len(detect_pauses(u2, dict(kind="all", thr=("fix", 2000), seg="replay"))) == 0, "区間をまたぐ間隔を数えた")
    # 文字だけの種類: 文字 → 削除 1.5 秒 → 文字 1.5 秒 は、全押下では休止なし・文字だけでは 3 秒の 1 件
    u3 = core.synth([(0, "i"), (1500, "d"), (3000, "i")])
    chk(len(detect_pauses(u3, dict(kind="all", thr=("fix", 2000), seg="replay"))) == 0, "全押下で削除をはさむ間隔を休止にした")
    chk(len(detect_pauses(u3, dict(kind="char", thr=("fix", 2000), seg="replay"))) == 1, "文字だけで削除をはさむ間隔が 1 つにならない")
    # 本人の分位: 99 個では定まらない・100 個以上で定まる
    u4 = core.synth([(i * 1000, "i") for i in range(100)])
    chk(detect_pauses(u4, dict(kind="all", thr=("q", 0.99), seg="replay")) is None, "100 個未満で分位が定まった")
    ev5 = [(i * 1000, "i") for i in range(101)] + [(100_000 + 5000, "i")]
    p5 = detect_pauses(core.synth(ev5), dict(kind="all", thr=("q", 0.99), seg="replay"))
    chk(p5 is not None and len(p5) == 1, "分位の閾値（1 秒）を超える 5 秒の間隔を 1 件にしない: %s" % (p5,))
    # 分位の順位: 10・20・…・2000 ms の 200 個と 5 秒の 1 個 → 0.99 分位は 199 番目の 1990 ms → 2000 と 5000 の 2 件
    gl = list(range(10, 2001, 10)) + [5000]
    t6 = np.concatenate(([0], np.cumsum(gl)))
    p6 = detect_pauses(core.synth([(int(x), "i") for x in t6]), dict(kind="all", thr=("q", 0.99), seg="replay"))
    chk(p6 is not None and len(p6) == 2, "分位の順位が見込みと違う: %s" % (None if p6 is None else len(p6),))
    # 偏相関の対照は round2_press の selftest が受け持つ
    print("selftest:", "OK" if ok else "FAIL")
    return ok


def main(data, cache, out, seed, jobs=8):
    seed = int(seed)
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    with ProcessPoolExecutor(int(jobs)) as ex:
        pres = list(ex.map(one_unit, enumerate(units), chunksize=16))
    # 対照: 露出が停滞の同じ切り方の露出と一致する（閾値が定まらない単位を除く）
    stall = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))
    for r, s in zip(pres, stall):
        assert r["unit"] == s["unit"]
        for c in PCONFIGS:
            pr = r["cfg"][pid(c)]
            if pr["undefined"]:
                continue
            ref = s["cfg"]["W30_q0.99_net_%s" % c["seg"]]["rows"]
            if [(d, e) for d, e, _ in pr["rows"]] != [(d, e) for d, e, _ in ref]:
                sys.exit("露出が停滞と違う: unit %d %s" % (r["unit"], pid(c)))
    del stall
    with open(os.path.join(cache, "pause_configs.pkl"), "wb") as f:
        pickle.dump(pres, f, protocol=pickle.HIGHEST_PROTOCOL)
    del pres
    speed = {u["writer"]: unit_speeds(u)["V2"] for u in units if u["dataset"] == "KLiCKe"}
    score = r1.load_scores(data)
    del units
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, score, speed)) as ex:
        res = dict(ex.map(one_cfg, [(pid(c), seed) for c in PCONFIGS]))
    b0 = json.load(open(os.path.join(os.path.dirname(out), "round1_extra.json")))["by_cfg"][STALL_B0]
    b0 = dict(b0, rank=sorted(DS, key=lambda d: -b0["rate"][d]))
    summ = summarize(res, b0)
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False, default=conv)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=conv))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    main(*sys.argv[1:])
