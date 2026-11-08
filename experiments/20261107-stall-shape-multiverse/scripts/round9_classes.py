#!/usr/bin/env python3
"""区間の類による読み（結果を見た後の提案・原稿の考察）を、既にある集計の区間から数える。新しい計算はしない。

類: 区間が −0.1 より下 / +0.1 より上 / ±0.1 の内（無視できる）/ どれでもない（精度不足）。
読み: 精度不足が群の半分以上なら「精度不足」。そうでなければ、分類できた定義（精度不足を除く）が 1 つの類だけなら、
その類。2 つ以上の類にあり、1 つの類が分類できた定義の 90% 以上ならその類（例外の数を添える）。それ以外は「定義に依る」。
主の類の割合（分類できた定義のうち）も出す。

  python3 round9_classes.py <results_dir> <out_json>
"""
import json
import sys
from collections import Counter
from pathlib import Path

BAND, SHARE = 0.1, 0.9
B0 = "W30_q0.99_net_replay"


def cls(ci):
    lo, hi = ci
    if hi < -BAND:
        return "below"
    if lo > BAND:
        return "above"
    if lo > -BAND and hi < BAND:
        return "negligible"
    return "imprecise"


def reading(cis):
    c = Counter(cls(ci) for ci in cis)
    classified = {k: v for k, v in c.items() if k != "imprecise"}
    n = sum(classified.values())
    top_share = None
    if n:
        top, m = max(classified.items(), key=lambda kv: kv[1])
        top_share = m / n
    if c.get("imprecise", 0) >= 0.5 * len(cis):
        r = "imprecise"
    else:
        r = top if (len(classified) == 1 or m >= SHARE * n) else "depends"
    return {"n": len(cis), "counts": dict(c), "reading": r, "top_share": top_share}


def selftest():
    assert cls([-0.3, -0.2]) == "below" and cls([0.2, 0.3]) == "above"
    assert cls([-0.05, 0.05]) == "negligible" and cls([-0.15, 0.0]) == "imprecise"
    assert reading([[-0.05, 0.05]] * 9 + [[-0.15, 0.0]])["reading"] == "negligible"
    assert reading([[-0.05, 0.05]] * 5 + [[-0.3, -0.2]] * 5)["reading"] == "depends"
    assert reading([[-0.05, 0.05]] * 9 + [[-0.3, -0.2]])["reading"] == "negligible"
    assert reading([[-0.15, 0.0]] * 3)["reading"] == "imprecise"
    assert reading([[-0.05, 0.05]] * 5 + [[-0.15, 0.0]] * 5)["reading"] == "imprecise"
    assert reading([[-0.05, 0.05]] * 6 + [[-0.15, 0.0]] * 4)["reading"] == "negligible"
    print("selftest: OK")


def main(res, out):
    r1 = json.load(open(res / "round1_extra.json"))["by_cfg"]
    r3 = json.load(open(res / "round3_speed.json"))
    ln = json.load(open(res / "round5_length.json"))
    pz = json.load(open(res / "round5_pause.json"))["by_cfg"]
    raw = lambda k: r1[k]["score"]["ci"]
    spd = lambda k: r3["summary"]["b0"]["S2_ci"] if k == B0 else r3["by_cfg"][k]["S2_ci"]
    lng = lambda k: ln["summary"]["b0"]["p_both_ci"] if k == B0 else ln["by_cfg"][k]["p_both_ci"]
    alt = [k for k in r1 if k != B0]
    prec = [k for k in r1 if k.startswith("W30_") and k.split("_")[2] == "net"
            and k.split("_")[1] in ("q0.95", "q0.99", "q0.995", "fix10s")]
    grids = {"stall_all89": alt, "stall_fixed36": [k for k in alt if "fix" in k],
             "stall_quantile53": [k for k in alt if "_q" in k],
             "stall_default_precedent12_incl_b0": prec,
             "stall_default_precedent_fixed3": [k for k in prec if "fix" in k],
             "stall_default_precedent_quantile9_incl_b0": [k for k in prec if "_q" in k]}
    out_d = {"band": BAND, "share": SHARE, "stall": {}, "pause_fixed18": {}}
    for g, ks in grids.items():
        out_d["stall"][g] = {"raw": reading([raw(k) for k in ks]), "length": reading([lng(k) for k in ks]),
                             "speed": reading([spd(k) for k in ks])}
    fx = [k for k in pz if "fix" in k]
    lnb = ln["by_cfg"]
    out_d["pause_fixed18"] = {"raw": reading([pz[k]["score"]["ci"] for k in fx]),
                              "length": reading([lnb[k]["p_both_ci"] for k in fx]),
                              "speed": reading([pz[k]["score"]["partial_ci"] for k in fx])}
    json.dump(out_d, open(out, "w"), indent=1)
    print(json.dumps(out_d, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(Path(sys.argv[1]), sys.argv[2])
