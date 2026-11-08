"""K1: B0 の始まり (s, t0) が全単位で replay と一致するか。CS1 2019 の見た側は変種の作業場とも比べる。

usage: k1_match.py <data_dir> <cache_dir> <out_json>
出力は一致・不一致の数だけ。
"""
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402
import prep  # noqa: E402

VARIANTS = {"work2019_w15": dict(core.B0, W=15_000), "work2019_w60": dict(core.B0, W=60_000),
            "work2019_q995": dict(core.B0, theta=("q", 0.995)),
            # 論文 1 の付録の変種は下限 200 個（supplement の表 "warm-up 200"）
            "work2019_q995_warm200": dict(core.B0, theta=("q", 0.995), warm=200)}


def one_cache(u):
    mine = {(s, t0) for s, t0, *_ in core.detect(u, core.B0)}
    return u["dataset"], mine == set(u["replay"])


def one_file(args):
    path, cfg = args
    u = prep.unit_record(json.load(open(path)))
    mine = {(s, t0) for s, t0, *_ in core.detect(u, cfg)}
    return mine == set(u["replay"]), len(mine), len(u["replay"])


def main(data, cache, out):
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    res = {"B0": {}}
    with ProcessPoolExecutor(8) as ex:
        for d, ok in ex.map(one_cache, units, chunksize=20):
            r = res["B0"].setdefault(d, {"match": 0, "mismatch": 0})
            r["match" if ok else "mismatch"] += 1
        for v, cfg in VARIANTS.items():
            files = sorted(glob.glob(os.path.join(data, "cs1", v.replace("_warm200", ""), "u[0-9]*.json")))
            tally = {"match": 0, "mismatch": 0, "onsets_mine": 0, "onsets_replay": 0}
            for ok, a, b in ex.map(one_file, [(f, cfg) for f in files], chunksize=10):
                tally["match" if ok else "mismatch"] += 1
                tally["onsets_mine"] += a
                tally["onsets_replay"] += b
            res[v] = tally
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))
    if any(r["mismatch"] for r in res["B0"].values()):
        sys.exit("K1: B0 が replay と一致しない単位がある")


if __name__ == "__main__":
    main(*sys.argv[1:4])
