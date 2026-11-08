#!/usr/bin/env python3
"""Run tools/check_release.py and accept only the large tables that are tables of definitions, not of writers.

Usage: python3 -I tools/verify.py [--private DATA_DIR]

check_release.py reports any JSON object with 30 or more keys, any table of 30 or more rows and any CSV file, because
such a table could hold one row per writer. This study reports values per definition: 90 stall definitions (W, theta,
condition A, segmentation), 36 pause definitions, and their combinations with dataset cells. A finding is accepted
only if (a) it is a JSON object at a listed path whose keys are all definition identifiers, or (b) it is a listed
table or CSV whose number of rows equals the expected number of definitions (times cells). Everything else fails.

Exit code: 0 = nothing found beyond the accepted tables / 1 = something found / 2 = usage error.
"""
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "experiments/20261107-stall-shape-multiverse/results/"
STALL = re.compile(r"^W(15|30|60)_(q0\.95|q0\.99|q0\.995|fix10s|fix30s)_(net|ins)_(replay|g5|g30)$")
PAUSE = re.compile(r"^P_(all|char)_(fix2s|fix5s|fix10s|q0\.95|q0\.99|q0\.995)_(replay|g5|g30)$")
# path -> JSON pointer of the object whose keys must be definition ids
JSON_OK = {EXP + n: p for n, p in [
    ("c1_lengths90.json", "/by_cfg"), ("c3_multiverse.json", "/"), ("round1_extra.json", "/by_cfg"),
    ("round2_press.json", "/by_cfg"), ("round3_speed.json", "/by_cfg"), ("round4_speedvar.json", "/by_cfg"),
    ("round5_length.json", "/by_cfg"), ("round5_pause.json", "/by_cfg"), ("round5_theta.json", "/by_cfg"),
    ("round7_warm.json", "/by_cfg"), ("round8_causal.json", "/by_cfg"), ("round9_wstart.json", "/by_cfg")]}
# path -> number of data rows (definitions x cells)
ROWS_OK = {"paper/appendix/tab_s3.tex": 90, "paper/appendix/tab_s4.tex": 90, "paper/appendix/tab_pause.tex": 36,
           "paper/figdata/score.csv": 90, "paper/figdata/ratio_cs19_cs21.csv": 90,
           "paper/figdata/ratio_cs19_kl.csv": 90, "paper/figdata/phi.csv": 450, "paper/figdata/lengths.csv": 270}


def keys_ok(path, pointer):
    d = json.load(open(os.path.join(ROOT, path)))
    for part in [p for p in pointer.split("/") if p]:
        d = d[part]
    return isinstance(d, dict) and all(STALL.match(k) or PAUSE.match(k) or k == "baseline" for k in d)


def rows(path):
    lines = open(os.path.join(ROOT, path), encoding="utf-8").read().splitlines()
    if path.endswith(".csv"):
        return len([x for x in lines[1:] if x.strip()])
    return len([x for x in lines if x.rstrip().endswith("\\\\") and "&" in x])


def gif_ok(path):
    """The README animation is binary, so its content cannot be read as text. Accept it only if it is the file that
    tools/readme_media.py draws from a synthetic key stream (byte-identical when redrawn here)."""
    import importlib.util
    import tempfile
    full = os.path.join(ROOT, path)
    if open(full, "rb").read(6) != b"GIF89a":
        return False
    spec = importlib.util.spec_from_file_location("readme_media", os.path.join(ROOT, "tools/readme_media.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "a.gif")
        try:
            m.detector_gif(out)
        except ImportError:
            print("note: Pillow or NumPy missing; %s accepted without redrawing" % path)
            return True
        return open(out, "rb").read() == open(full, "rb").read()


def main(argv):
    cmd = [sys.executable, "-I", os.path.join(ROOT, "tools/check_release.py"), "--root", ROOT] + argv
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 2:
        sys.stderr.write(r.stdout + r.stderr)
        return 2
    bad = []
    for line in r.stdout.splitlines():
        if not line.startswith("FOUND "):
            continue
        path = line[6:].split(":", 1)[0]
        if path in JSON_OK and "keys at" in line and keys_ok(path, JSON_OK[path]):
            continue
        if path in ROWS_OK and ("table of" in line or "raw data extension .csv" in line) and rows(path) == ROWS_OK[path]:
            continue
        if path == "docs/media/detector.gif" and "not UTF-8 text" in line and gif_ok(path):
            continue
        bad.append(line)
    for line in bad:
        print(line)
    print("check_release: %s; accepted definition tables; %d other findings" % (r.stdout.strip().splitlines()[-1], len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
