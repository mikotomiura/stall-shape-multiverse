"""作業場の中間を 1 つのキャッシュにまとめる（conditions.md の D・K4）。

usage: prep.py <data_dir> <cache_dir>
出力（repo の外）: <cache_dir>/units.pkl …… 単位ごとの押下・観測区間・まとまり・replay の始まり
                    <cache_dir>/prep_summary.json …… 数だけ（ID・時刻を含まない）
CS1 の課題の識別（D2）は生の CSV の 2 列から並びを作り直して照合する。取り置きの照合（D3）が合わなければ止める。
"""
import csv
import glob
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../20261104-cs1-landscape/scripts"))
sys.path.insert(0, os.path.join(HERE, "../../20261031-coauthor-pre-request-signal/scripts"))

TEXT = ("alphabetic", "numeric", "punctuation", "sentence_punctuation", "space", "enter", "tab")
DELETE = ("backspace", "delete")
MIN_WRITER_OBS_MS = 10 * 60_000   # D の表: 観測 10 分以上の書き手

# D3: 作業場 → 作るときに読み飛ばした一覧のファイル（その SHA-256 が build_summary にあるはず）
CS1_BUILDS = {
    "cs1_2019_seen": ("cs1/work2019", "cs1/holdout.json", "2019"),
    "cs1_2019_opened": ("cs1/work2019_holdout", "cs1/skip_seen_2019.json", "2019"),
    "cs1_2021_seen": ("cs1/work2021_seen", "cs1/skip_holdout_2021.json", "2021"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def key_arrays(keys):
    t = np.array([k[0] for k in keys], dtype=np.int64)
    cat = [k[1] for k in keys]
    ins = np.array([c in TEXT for c in cat], dtype=bool)
    dele = np.array([c in DELETE for c in cat], dtype=bool)
    return t, ins, dele


def obs_ms(intervals):
    return sum(b - a for a, b in intervals)


def unit_record(j):
    t, ins, dele = key_arrays(j["keys"])
    return {
        "t": t, "ins": ins, "dele": dele,
        "iv": [tuple(x) for x in j["observed_intervals"]],
        "blocks": [tuple(x) for x in j.get("blocks", [])],
        "replay": [(o[0], o[1]) for o in j["onsets"]],
    }


def cs1_pairs(year, raw, hold):
    """build_cs1.read_units と同じ規則で、単位を作る (SubjectID, AssignmentID) の組を集める（行は持たない）。"""
    import build_cs1 as b
    fn = b.map_2019 if year == "2019" else b.map_2021
    csv.field_size_limit(10**9)
    pairs = set()
    with open(raw, newline="") as f:
        for row in csv.DictReader(f):
            if row["SubjectID"] in hold:
                continue
            kind, _ = fn(row)
            if kind == "count":
                continue
            pairs.add((row["SubjectID"], row["AssignmentID"]))
    return sorted(pairs)


def load_cs1(data, name, summ):
    wd, hold_p, year = CS1_BUILDS[name]
    wd, hold_p = os.path.join(data, wd), os.path.join(data, hold_p)
    bs = json.load(open(os.path.join(wd, "build_summary.json")))
    if bs["holdout_sha256"] != sha256_file(hold_p):
        sys.exit("D3: %s の取り置きの一覧が作業場と合わない" % name)
    hold = set(json.load(open(hold_p))[year])
    pairs = cs1_pairs(year, os.path.join(data, "cs1/raw", year, "keystrokes.csv"), hold)
    if len(pairs) != bs["units"]:
        sys.exit("D2: %s の単位の数が合わない（%d と %d）" % (name, len(pairs), bs["units"]))
    subj = sorted({s for s, _ in pairs})
    s_idx = {s: i for i, s in enumerate(subj)}
    wmap = json.load(open(os.path.join(wd, "writers.json")))
    for s, i in s_idx.items():
        if wmap[str(i)] != hashlib.sha256(s.encode()).hexdigest():
            sys.exit("D2: %s の書き手の照合が合わない" % name)
    units_w = json.load(open(os.path.join(wd, "units.json")))
    # 課題の通し番号は年版の中で共通（書き手をまたぐ効果のため）。ID は残さない
    aids = sorted({a for _, a in pairs})
    # 同じ年版の作業場（見た側と開けた側）で課題の集合が同じでなければ通し番号が別の課題を指す → 止める
    h = hashlib.sha256("\n".join(map(str, aids)).encode()).hexdigest()
    if TASK_SETS.setdefault(year, h) != h:
        sys.exit("D2: %s の課題の集合が同じ年版の別の作業場と違う" % name)
    a_idx = {a: i for i, a in enumerate(aids)}
    out = []
    for k, w in units_w.items():
        sid, aid = pairs[int(k)]
        if s_idx[sid] != w:
            sys.exit("D2: %s の単位 %s の書き手が合わない" % (name, k))
        r = unit_record(json.load(open(os.path.join(wd, "u%s.json" % k))))
        r.update(dataset="CS1 " + year, writer="%s:%d" % (name, w), task="%s:%d" % (year, a_idx[aid]))
        out.append(r)
    summ[name] = {"units": len(out), "writers": len({r["writer"] for r in out}), "tasks": len(aids)}
    return out


TASK_SETS = {}


def load_klicke(data, summ):
    wd = os.path.join(data, "klicke/work")
    out = []
    for f in sorted(glob.glob(os.path.join(wd, "w[0-9]*.json"))):
        r = unit_record(json.load(open(f)))
        r.update(dataset="KLiCKe", writer="k:" + os.path.basename(f)[1:-5], task=None)
        out.append(r)
    summ["klicke"] = {"units": len(out)}
    return out


def load_coauthor(data, summ):
    wd = os.path.join(data, "landscape/coauthor_work")
    bs = json.load(open(os.path.join(wd, "build_summary.json")))
    if bs.get("held_out_sessions") != 288:
        sys.exit("D3: CoAuthor の取り置きのセッションの数が 288 でない")
    sess = json.load(open(os.path.join(wd, "sessions.json")))
    out = []
    for k, w in sess.items():
        r = unit_record(json.load(open(os.path.join(wd, "s%s.json" % k))))
        r.update(dataset="CoAuthor", writer="c:%d" % w, task=None)
        out.append(r)
    summ["coauthor"] = {"units": len(out), "writers": len({r["writer"] for r in out})}
    return out


def main(data, cache):
    os.makedirs(cache, exist_ok=True)
    summ = {}
    units = []
    for name in CS1_BUILDS:
        units += load_cs1(data, name, summ)
    units += load_klicke(data, summ)
    units += load_coauthor(data, summ)
    # 観測 10 分以上の書き手だけ（D の表）
    tot = {}
    for u in units:
        tot[u["writer"]] = tot.get(u["writer"], 0) + obs_ms(u["iv"])
    keep = {w for w, v in tot.items() if v >= MIN_WRITER_OBS_MS}
    dropped = {}
    for u in units:
        if u["writer"] not in keep:
            dropped[u["dataset"]] = dropped.get(u["dataset"], 0) + 1
    units = [u for u in units if u["writer"] in keep]
    summ["dropped_units_short_writers"] = dropped
    summ["by_dataset"] = {}
    for d in sorted({u["dataset"] for u in units}):
        us = [u for u in units if u["dataset"] == d]
        summ["by_dataset"][d] = {"units": len(us), "writers": len({u["writer"] for u in us}),
                                 "replay_onsets": sum(len(u["replay"]) for u in us)}
    with open(os.path.join(cache, "units.pkl"), "wb") as f:
        pickle.dump(units, f, protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(summ, open(os.path.join(cache, "prep_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
