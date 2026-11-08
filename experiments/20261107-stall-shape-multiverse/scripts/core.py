"""停滞の検出を設定で振る（conditions.md の B・C3・K2・K3）。

検出の規則は `20261103-stall-candidates-c1c2c6/candidates.py` の detect（stall_onset.rs の写し・replay と一致を照合済み）
と同じで、窓 W・θ・条件 A・観測区間を引数にした。
usage: core.py --selftest
"""
import bisect
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../20261103-stall-candidates-c1c2c6"))
sys.path.insert(0, os.path.join(HERE, "../../20261101-klicke-early-stall-trigger/scripts"))
import candidates as cand  # noqa: E402  Acc（因果に累積する nearest-rank 分位）と gap_pool

WARM = 100
DAY_GAP_MS = 6 * 3600_000   # D1
MIN_UNIT_OBS_MS = 5 * 60_000  # D4

WS = (15_000, 30_000, 60_000)
THETAS = (("q", 0.95), ("q", 0.99), ("q", 0.995), ("fix", 10_000), ("fix", 30_000))
RULES = ("net", "ins")
SEGS = ("replay", "g5", "g30")
CONFIGS = [dict(W=w, theta=th, rule=r, seg=sg) for w, th, r, sg in itertools.product(WS, THETAS, RULES, SEGS)]
B0 = dict(W=30_000, theta=("q", 0.99), rule="net", seg="replay")
C1_CONFIGS = [dict(W=w, theta=th, rule="net", seg="replay") for w in WS for th in THETAS if th[0] == "q"]


def cfg_id(c):
    th = "q%g" % c["theta"][1] if c["theta"][0] == "q" else "fix%ds" % (c["theta"][1] // 1000)
    return "W%d_%s_%s_%s" % (c["W"] // 1000, th, c["rule"], c["seg"])


def key_segments(t, gap_ms):
    """押下の空白 gap_ms 以上で切った [最初の押下, 最後の押下] の列。"""
    if len(t) == 0:
        return []
    cut = np.flatnonzero(np.diff(t) >= gap_ms)
    starts = np.concatenate(([0], cut + 1))
    ends = np.concatenate((cut, [len(t) - 1]))
    return [(int(t[a]), int(t[b])) for a, b in zip(starts, ends) if t[b] > t[a]]


def intervals_for(u, seg):
    if seg == "replay":
        return u["iv"]
    return key_segments(u["t"], {"g5": 5 * 60_000, "g30": 30 * 60_000}[seg])


class FixTheta:
    def __init__(self, ms):
        self.ms = ms

    def at(self, t):
        return (self.ms, None)


def detect(u, c):
    """戻り値: [(s, t0, resume or None, 区間の終わり)]。resume は t0 より後で同じ観測区間の中の最初の文字の押下
    （区間の外なら None・`candidates.py` の `Day.resume` と同じ）。"""
    W = c["W"]
    t = u["t"]
    ins_t = t[u["ins"]]
    del_t = t[u["dele"]]
    iv = intervals_for(u, c["seg"])
    tl = t.tolist()
    if c["theta"][0] == "q":
        theta = cand.Acc(cand.gap_pool(tl, iv), c["theta"][1], c.get("warm", WARM))
    else:
        theta = FixTheta(c["theta"][1])

    def cnt(arr, x):
        return int(np.searchsorted(arr, x, "right") - np.searchsorted(arr, x - W, "right"))

    if c["rule"] == "net":
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
        prev = False
        for i, x in enumerate(inst):
            if ca[i] and not prev:
                th = theta.at(x)
                if th is not None and x + th[0] <= b:
                    t0 = x + th[0]
                    j, ok = i + 1, True
                    while j < len(inst) and inst[j] <= t0:
                        if not ca[j]:
                            ok = False
                            break
                        j += 1
                    if ok:
                        k = int(np.searchsorted(ins_t, t0, "right"))
                        r = int(ins_t[k]) if k < len(ins_t) and ins_t[k] <= b else None
                        out.append((x, t0, r, b))
            prev = ca[i]
    return out


# ---------- 単位の下の区切り（CS1 のまとまり・日） ----------

def regions(u):
    """まとまりの開始で単位の時間を区切る [(開始, 次の開始), ...] と日の番号。まとまりが無ければ 1 区間。"""
    bl = sorted(u["blocks"])
    if not bl:
        return [(-math.inf, math.inf)], [0]
    edges = [b[0] for b in bl]
    reg = [(-math.inf if i == 0 else edges[i], edges[i + 1] if i + 1 < len(edges) else math.inf)
           for i in range(len(edges))]
    day, days = 0, [0]
    for i in range(1, len(bl)):
        if bl[i][0] - bl[i - 1][1] >= DAY_GAP_MS:
            day += 1
        days.append(day)
    return reg, days


def overlap(iv, a, b):
    return sum(max(0, min(y, b) - max(x, a)) for x, y in iv)


def region_index(reg, x):
    starts = [r[0] for r in reg]
    return max(0, bisect.bisect_right(starts, x) - 1)


def unit_rows(u, stalls, c):
    """まとまりごとの (日, 露出 ms, 件数) と、停滞ごとの (まとまり, 長さ ms or None)。"""
    iv = intervals_for(u, c["seg"])
    reg, days = regions(u)
    expo = [overlap(iv, a, b) for a, b in reg]
    n = [0] * len(reg)
    lens = []
    for s, t0, r, _ in stalls:
        k = region_index(reg, t0)
        n[k] += 1
        lens.append((k, None if r is None else r - t0))
    return list(zip(days, expo, n)), lens


# ---------- ラベルの一致 ----------

def inside_share(points, episodes):
    """points（t0 の列）のうち、episodes の [s, 終わり] のどれかに入る数。"""
    if not points or not episodes:
        return 0
    ep = sorted(episodes)
    starts = [e[0] for e in ep]
    runmax = list(itertools.accumulate((e[1] for e in ep), max))
    hit = 0
    for p in points:
        i = bisect.bisect_right(starts, p) - 1
        if i >= 0 and runmax[i] >= p:
            hit += 1
    return hit


def episodes(stalls):
    """[s, 再開]。再開しなかった停滞は、その観測区間の終わりまで。"""
    return [(s, r if r is not None else b) for s, t0, r, b in stalls]


# ---------- K2・K3 ----------

def synth(events, iv=None):
    """events: [(ms, 'i'|'d'|'o')]（文字・削除・その他）。"""
    events = sorted(events)
    t = np.array([e[0] for e in events], dtype=np.int64)
    u = {"t": t, "ins": np.array([e[1] == "i" for e in events]),
         "dele": np.array([e[1] == "d" for e in events]),
         "iv": iv or [(int(t[0]), int(t[-1]))], "blocks": [], "replay": []}
    return u


def selftest():
    fails = []
    # 1 秒ごとに 200 回打つ（θ の 0.99 分位は 1 秒）→ 90 秒止まる → 再開
    base = [(i * 1000, "i") for i in range(200)]
    stop = base + [(199_000 + 90_000 + i * 1000, "i") for i in range(20)]
    u = synth(stop)
    st = detect(u, B0)
    if len(st) != 1:
        fails.append("止まりが 1 件にならない: %d" % len(st))
    else:
        s, t0, r, _ = st[0]
        if not (s == 199_000 + 30_000 and t0 == s + 1000 and r == 289_000):
            fails.append("s・t0・再開が見込みと違う: %s" % (st[0],))
    if detect(synth(base + [(199_000 + 20_000 + i * 1000, "i") for i in range(20)]), B0):
        fails.append("30 秒未満の止まりで立った")
    # 削除の連打: 文字が続いていても正味 < 1 なら立つ（net）・ins では立たない
    churn = base + [(199_000 + 500 + i * 500, "i" if i % 2 == 0 else "d") for i in range(200)] + \
        [(199_000 + 500 + 200 * 500 + 1000 * i, "i") for i in range(5)]
    churn = [e for e in churn if not (e[1] == "i" and 199_000 < e[0] < 299_500 and (e[0] // 500) % 4 == 2)]
    n_net = len(detect(synth(churn), B0))
    n_ins = len(detect(synth(churn), dict(B0, rule="ins")))
    if not (n_net >= 1 and n_ins == 0):
        fails.append("削除の連打の扱いが見込みと違う: net %d ins %d" % (n_net, n_ins))
    # θ の分位が下限 100 個未満なら立たない（50 回だけ打つ）・固定 θ なら立つ
    short = [(i * 1000, "i") for i in range(50)] + [(49_000 + 90_000, "i")]
    if detect(synth(short), B0):
        fails.append("100 個未満で θ が定義された")
    if len(detect(synth(short), dict(B0, theta=("fix", 10_000)))) != 1:
        fails.append("固定 θ で立たない")
    # W = 60 秒なら 50 秒の止まりでは立たない・15 秒なら立つ
    mid = base + [(199_000 + 50_000 + i * 1000, "i") for i in range(20)]
    if detect(synth(mid), dict(B0, W=60_000)) or len(detect(synth(mid), dict(B0, W=15_000))) != 1:
        fails.append("W の効きが見込みと違う")
    # 押下の空白で切る区間: 6 分の空白（g5 では区間の外・g30 では中）
    gap = base + [(199_000 + 360_000 + i * 1000, "i") for i in range(200)]
    if detect(synth(gap), dict(B0, seg="g5")) or len(detect(synth(gap), dict(B0, seg="g30"))) != 1:
        fails.append("区間の切り方の効きが見込みと違う")
    # まとまりと日
    u2 = synth(base)
    u2["blocks"] = [(0, 10_000), (50_000, 60_000), (60_000 + DAY_GAP_MS, 70_000 + DAY_GAP_MS)]
    reg, days = regions(u2)
    if days != [0, 0, 1] or region_index(reg, 55_000) != 1 or region_index(reg, -5) != 0:
        fails.append("まとまり・日の区切りが見込みと違う")
    # θ の分位が効く: 200 個の間隔のうち 4 個だけ 5 秒（0.95 分位 = 1 秒・0.99 分位 = 5 秒）
    var = [(i * 1000 + 4000 * min(i // 50, 4), "i") for i in range(201)]
    end = var[-1][0]
    var2 = var + [(end + 90_000, "i")]
    d95 = detect(synth(var2), dict(B0, theta=("q", 0.95)))
    d99 = detect(synth(var2), B0)
    if not (len(d95) == 1 and d95[0][1] - d95[0][0] == 1000 and len(d99) == 1 and d99[0][1] - d99[0][0] == 5000):
        fails.append("θ の分位の効きが見込みと違う: %s %s" % (d95, d99))
    # 固定 θ の値が t0 − s になる
    d10 = detect(synth(var2), dict(B0, theta=("fix", 10_000)))
    if not (len(d10) == 1 and d10[0][1] - d10[0][0] == 10_000):
        fails.append("固定 θ の値が効いていない: %s" % (d10,))
    # 区間の終わりちょうどに t0 が来る止まりは立つ（s + θ ≤ 区間の終わり）
    u3 = synth(base + [(199_000 + 90_000, "i")], iv=[(0, 199_000 + 30_000 + 1000)])
    if len(detect(u3, B0)) != 1:
        fails.append("区間の終わりちょうどの t0 で立たない")
    # ラベルの一致（入れ子の区間: 長い区間の中の点）
    if inside_share([5, 15, 30], [(0, 10), (12, 20)]) != 2 or inside_share([50], [(0, 100), (10, 20)]) != 1:
        fails.append("inside_share が見込みと違う")
    # 再開は同じ観測区間の中だけ（区間の外の文字は再開に数えない）・再開しなければ区間の終わりまで
    u4 = synth(base + [(199_000 + 90_000, "i"), (199_000 + 3 * 3600_000, "i")],
               iv=[(0, 199_000 + 100_000), (199_000 + 3 * 3600_000 - 10, 199_000 + 3 * 3600_000 + 10)])
    d4 = detect(u4, B0)
    if not (len(d4) == 1 and d4[0][2] == 289_000):
        fails.append("区間の中の再開が見込みと違う: %s" % (d4,))
    u5 = synth(base + [(199_000 + 3 * 3600_000, "i")],
               iv=[(0, 199_000 + 100_000), (199_000 + 3 * 3600_000 - 10, 199_000 + 3 * 3600_000 + 10)])
    d5 = detect(u5, B0)
    if not (len(d5) == 1 and d5[0][2] is None and episodes(d5) == [(d5[0][0], 199_000 + 100_000)]):
        fails.append("区間の外の文字を再開に数えた: %s" % (d5,))
    # --- 2 系統レビューで生き残った変異を捕まえる対照 ---
    # 空白がちょうど gap_ms なら切る（≥）
    if key_segments(np.array([0, 1000, 301_000]), 300_000) != [(0, 1000)]:
        fails.append("空白ちょうど 5 分で切れない")
    # 区間の始まりから W 経つまでは立たない（区間の前の押下は見えないので）
    late = synth([(20_000 + i * 1000, "i") for i in range(280)], iv=[(0, 299_000)])
    if detect(late, dict(B0, theta=("fix", 10_000))):
        fails.append("区間の始まりから W 経つ前に立った")
    # θ の間に条件が崩れたら立たない（継続の確かめ）。崩れた後の立ち上がりで 1 件だけ立つ
    brk = base + [(234_000, "i")] + [(234_000 + 90_000 + i * 1000, "i") for i in range(5)]
    db = detect(synth(brk), dict(B0, theta=("fix", 10_000)))
    if not (len(db) == 1 and db[0][0] == 264_000):
        fails.append("θ の間に崩れた止まりの扱いが見込みと違う: %s" % (db,))
    # 止まりの中で条件が真のまま次の瞬間が来ても、立ち上がりでなければ立たない
    dl = base + [(239_000, "d")] + [(289_000 + i * 1000, "i") for i in range(5)]
    dd = detect(synth(dl), dict(B0, theta=("fix", 10_000)))
    if len(dd) != 1:
        fails.append("立ち上がりでない瞬間で立った: %s" % (dd,))
    # θ は s までの間隔だけで決まる（因果）。止まりの後に長い間隔が続いても最初の止まりの θ は 1 秒
    fut = base + [(289_000 + i * 5000, "i") for i in range(300)]
    df = detect(synth(fut), B0)
    if not (df and df[0][1] - df[0][0] == 1000):
        fails.append("θ が後の間隔を見た: %s" % (df[:1],))
    # 露出の重なり・停滞の長さ（再開 − t0）・まとまりの開始ちょうどの割り当て
    if overlap([(0, 10), (20, 30)], 5, 25) != 10:
        fails.append("overlap が見込みと違う")
    rows, lens = unit_rows(u, st, B0)
    if not (lens and lens[0][1] == 289_000 - (199_000 + 31_000) and rows[0][1] == u["iv"][0][1] - u["iv"][0][0]):
        fails.append("unit_rows の長さか露出が見込みと違う: %s %s" % (rows, lens))
    if region_index(reg, 50_000) != 1:
        fails.append("まとまりの開始ちょうどが前のまとまりに入った")
    print("selftest:", "OK" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    return not fails


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
