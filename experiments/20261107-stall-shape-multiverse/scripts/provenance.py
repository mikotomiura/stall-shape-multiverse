"""来歴: 中間（repo の外）と scripts の SHA-256・環境・並列の数・所要時間。

usage: provenance.py <cache_dir> <out_json> <jobs> <seed> <run_configs の秒>
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy
import pandas
import scipy
import statsmodels

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.normpath(os.path.join(HERE, "../.."))


def imported_files():
    """scripts が他の実験から読み込む .py を、import をたどってすべて集める（芋づるの先も含む）。"""
    sys.path.insert(0, HERE)
    import core  # noqa: F401
    import prep  # noqa: F401  build_cs1 の場所を sys.path に足す
    import build_cs1  # noqa: F401  prep の中で遅れて import されるもの
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and f.endswith(".py"):
            f = os.path.realpath(f)
            if f.startswith(EXP_ROOT + os.sep) and os.path.dirname(f) != HERE:
                out.add(f)
    return sorted(out)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(cache, out, jobs, seed, secs):
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", HERE], capture_output=True, text=True).stdout.strip()
    prov = {
        "seed": int(seed),
        "git_head": head, "scripts_dirty": bool(dirty),
        "python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
        "pandas": pandas.__version__, "statsmodels": statsmodels.__version__,
        "cache_sha256": {n: sha(os.path.join(cache, n)) for n in ("units.pkl", "configs.pkl", "c3b.json", "prep_summary.json")},
        "scripts_sha256": {n: sha(os.path.join(HERE, n)) for n in sorted(os.listdir(HERE)) if n.endswith(".py")},
        # 他の実験から import するもの（芋づるの先も含む）
        "imported_sha256": {os.path.relpath(q, EXP_ROOT): sha(q) for q in imported_files()},
        "jobs": int(jobs), "run_configs_seconds": int(secs),
        "note": "結果は並列の数によらない。中間は repo の外にあり、ここには SHA-256 だけを置く",
    }
    json.dump(prov, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:6])
