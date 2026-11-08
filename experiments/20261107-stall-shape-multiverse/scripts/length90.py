"""停滞の長さの分解を 90 通りすべてで回す（conditions-length90.md）。

usage: length90.py <cache_dir> <out_json> [jobs]
入力: <cache_dir>/configs.pkl。集計値（分け前・収束）だけを書く。
"""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import core  # noqa: E402

RES = None


def _init(cache):
    global RES
    RES = pickle.load(open(os.path.join(cache, "configs.pkl"), "rb"))


def one(arg):
    cid, d = arg
    _, lens = analyze.unit_frames(RES, cid, d)
    fl = analyze.fit_lengths(lens, analyze.LEN_NESTED[d])
    sh = analyze.shares(fl["sd"], list(fl["sd"]))
    others = [v for k, v in sh.items() if k != "stall" and v is not None]
    return cid, d, {"n_lengths": len(lens), "shares": sh, "converged": fl["converged"], "method": fl["method"],
                    "stall_minus_max_other": (sh["stall"] - max(others)) if others else None}


def main(cache, out, jobs=8):
    tasks = [(core.cfg_id(c), d) for c in core.CONFIGS for d in analyze.LEN_NESTED]
    res = {}
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache,)) as ex:
        for cid, d, v in ex.map(one, tasks):
            res.setdefault(cid, {})[d] = v
            print("L90", cid, d, flush=True)
    # 対照: 既存の 9 通りと小数 3 桁で一致
    ref = json.load(open(os.path.join(os.path.dirname(out), "c1_decomposition.json")))
    bad = []
    for cid, byd in ref.items():
        for d, v in byd.items():
            for k, s in v["lengths_share"].items():
                if s is not None and abs(res[cid][d]["shares"][k] - s) >= 5e-4:
                    bad.append((cid, d, k, s, res[cid][d]["shares"][k]))
    if bad:
        sys.exit("既存の 9 通りと一致しない: %r" % bad[:5])
    summ = {}
    for d in analyze.LEN_NESTED:
        vals = [res[cid][d] for cid in res]
        ok = [v["converged"] and v["stall_minus_max_other"] is not None and v["stall_minus_max_other"] > 0 for v in vals]
        summ[d] = {"n_cfg": len(vals), "R1_holds_all": all(ok), "n_fail": len(ok) - sum(ok),
                   "fail_cfgs": [cid for cid, o in zip(res, ok) if not o],
                   "n_unconverged": sum(not v["converged"] for v in vals),
                   "stall_share_range": [min(v["shares"]["stall"] for v in vals), max(v["shares"]["stall"] for v in vals)],
                   "writer_share_range": [min(v["shares"]["writer"] for v in vals), max(v["shares"]["writer"] for v in vals)],
                   "min_margin": min(v["stall_minus_max_other"] for v in vals if v["stall_minus_max_other"] is not None)}
    json.dump({"summary": summ, "by_cfg": res}, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
