"""C3b: 推定の水準を切り方で揃える（conditions.md）。

1 秒ごとの `feat` の行のうち、最後の押下（どの押下でも）からの経過が 30・60・180 秒以内の行だけで
`p_stuck ≥ 0.60` の時間の割合を書き手ごとに出し、書き手の中央値を全部の行の値と並べる。
usage: c3b.py <data_dir> <cache_dir>   → <cache_dir>/c3b.json（集計値だけ）
"""
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prep  # noqa: E402

GS = {"le30": 30_000, "le60": 60_000, "le180": 180_000, "all": None}
CUT = 0.60


def one(arg):
    path, writer = arg
    j = json.load(open(path))
    t = np.array([k[0] for k in j["keys"]], dtype=np.int64)
    f = np.array([(r[0], r[1]) for r in j["feat"]], dtype=float).reshape(-1, 2)
    out = {}
    if len(f) == 0 or len(t) == 0:
        return writer, {g: (0, 0) for g in GS}
    k = np.searchsorted(t, f[:, 0], "right") - 1
    since = np.where(k >= 0, f[:, 0] - t[np.maximum(k, 0)], np.inf)
    hi = f[:, 1] >= CUT
    for g, lim in GS.items():
        m = np.ones(len(f), bool) if lim is None else since <= lim
        out[g] = (int(m.sum()), int((hi & m).sum()))
    return writer, out


def sources(data):
    src = []
    for name, (wd, _, _) in prep.CS1_BUILDS.items():
        units_w = json.load(open(os.path.join(data, wd, "units.json")))
        year = prep.CS1_BUILDS[name][2]
        src += [(os.path.join(data, wd, "u%s.json" % k), "CS1 " + year, "%s:%d" % (name, w)) for k, w in units_w.items()]
    for p in sorted(glob.glob(os.path.join(data, "klicke/work/w[0-9]*.json"))):
        src.append((p, "KLiCKe", "k:" + os.path.basename(p)[1:-5]))
    sess = json.load(open(os.path.join(data, "landscape/coauthor_work/sessions.json")))
    src += [(os.path.join(data, "landscape/coauthor_work/s%s.json" % k), "CoAuthor", "c:%d" % w) for k, w in sess.items()]
    return src


def main(data, cache):
    keep = {u["writer"]: u["dataset"] for u in pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))}
    src = [s for s in sources(data) if s[2] in keep]
    acc = defaultdict(lambda: {g: [0, 0] for g in GS})
    with ProcessPoolExecutor(8) as ex:
        for w, o in ex.map(one, [(p, w) for p, _, w in src], chunksize=10):
            for g, (a, b) in o.items():
                acc[w][g][0] += a
                acc[w][g][1] += b
    res = {}
    for d in ("CS1 2019", "CS1 2021", "CoAuthor", "KLiCKe"):
        ws = [w for w, dd in keep.items() if dd == d]
        res[d] = {}
        for g in GS:
            vals = [acc[w][g][1] / acc[w][g][0] for w in ws if acc[w][g][0] > 0]
            rows = sum(acc[w][g][0] for w in ws)
            res[d][g] = {"writer_median": float(np.median(vals)) if vals else None, "writers": len(vals),
                         "row_share": rows / max(1, sum(acc[w]["all"][0] for w in ws))}
    res["ratio_cs1_2019_over_klicke"] = {
        g: (res["CS1 2019"][g]["writer_median"] / res["KLiCKe"][g]["writer_median"]
            if res["KLiCKe"][g]["writer_median"] is not None and res["KLiCKe"][g]["writer_median"] > 0 else None) for g in GS}
    json.dump(res, open(os.path.join(cache, "c3b.json"), "w"), indent=1, ensure_ascii=False)
    print("c3b done")


if __name__ == "__main__":
    main(*sys.argv[1:3])
