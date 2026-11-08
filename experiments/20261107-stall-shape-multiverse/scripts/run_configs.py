"""C3 の 90 通りを全単位で回し、集計の材料を repo の外に置く。

usage: run_configs.py <cache_dir> [jobs]
出力: <cache_dir>/configs.pkl …… [{dataset, writer, task, unit, cfg: {id: {rows, lens, agree}}}]
  rows  = [(日, 露出 ms, 件数)]（まとまりごと。CS1 以外は 1 行）
  lens  = [(まとまりの番号, 長さ ms or None)]
  agree = (B0 の t0 のうちこの通りの [s, 再開] に入る数, B0 の数, この通りの t0 のうち B0 の [s, 再開] に入る数, この通りの数)
"""
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402


def one(arg):
    i, u = arg
    b0 = core.detect(u, core.B0)
    b0_t0 = [st[1] for st in b0]
    b0_ep = core.episodes(b0)
    out = {}
    for c in core.CONFIGS:
        st = b0 if c == core.B0 else core.detect(u, c)
        rows, lens = core.unit_rows(u, st, c)
        t0s = [x[1] for x in st]
        agree = (core.inside_share(b0_t0, core.episodes(st)), len(b0_t0),
                 core.inside_share(t0s, b0_ep), len(t0s))
        out[core.cfg_id(c)] = {"rows": rows, "lens": lens, "agree": agree}
    return {"dataset": u["dataset"], "writer": u["writer"], "task": u["task"], "unit": i, "cfg": out}


def main(cache, jobs=8):
    units = pickle.load(open(os.path.join(cache, "units.pkl"), "rb"))
    t = time.time()
    res = []
    with ProcessPoolExecutor(int(jobs)) as ex:
        for k, r in enumerate(ex.map(one, enumerate(units), chunksize=4)):
            res.append(r)
            if k % 500 == 0:
                print("%d / %d  %.0f s" % (k, len(units), time.time() - t), flush=True)
    with open(os.path.join(cache, "configs.pkl"), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("done %.0f s" % (time.time() - t))


if __name__ == "__main__":
    main(*sys.argv[1:3])
