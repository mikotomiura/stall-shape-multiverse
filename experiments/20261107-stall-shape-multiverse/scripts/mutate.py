"""K3: core.py を 1 か所ずつ壊して selftest が赤になるかを見る。改変なしの対照は緑でなければならない。

usage: mutate.py <python>   （一時フォルダで回し、repo の core.py は書き換えない）
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

MUTANTS = {
    "control_none": None,
    "cond_le": ("return cnt(ins_t, x) - cnt(del_t, x) < 1", "return cnt(ins_t, x) - cnt(del_t, x) <= 1"),
    "ins_rule_counts_del": ("return cnt(ins_t, x) < 1", "return cnt(ins_t, x) - cnt(del_t, x) < 1"),
    "window_W": ('np.searchsorted(arr, x - W, "right")', 'np.searchsorted(arr, x - 30_000, "right")'),
    "quantile_ignored": ('cand.gap_pool(tl, iv), c["theta"][1], c.get("warm", WARM)',
                         'cand.gap_pool(tl, iv), 0.5, c.get("warm", WARM)'),
    "warm_ignored": ('c["theta"][1], c.get("warm", WARM))', 'c["theta"][1], 1)'),
    "fixtheta_ignored": ("return (self.ms, None)", "return (1000, None)"),
    "theta_strict": ("if th is not None and x + th[0] <= b:", "if th is not None and x + th[0] < b:"),
    "seg_ignored": ('if seg == "replay":', "if True:"),
    "resume_wrong": ('k = int(np.searchsorted(ins_t, t0, "right"))', 'k = int(np.searchsorted(ins_t, t0, "left") - 1)'),
    "resume_unbounded": ("if k < len(ins_t) and ins_t[k] <= b else None", "if k < len(ins_t) else None"),
    "day_gap": ("if bl[i][0] - bl[i - 1][1] >= DAY_GAP_MS:", "if bl[i][0] - bl[i - 1][1] >= 1:"),
    "runmax": ("if i >= 0 and runmax[i] >= p:", "if i >= 0 and ep[i][1] >= p:"),
    # 2 系統レビュー（2026-11-07）で生き残った変異
    "gap_strict": ("np.diff(t) >= gap_ms", "np.diff(t) > gap_ms"),
    "no_continuation": ("if not ca[j]:", "if False:"),
    "start_at_a": ("start = a + W", "start = a"),
    "no_rise": ("if ca[i] and not prev:", "if ca[i]:"),
    "theta_future": ("th = theta.at(x)", "th = theta.at(b)"),
    "overlap_plus1": ("return sum(max(0, min(y, b) - max(x, a)) for x, y in iv)",
                      "return sum(max(0, min(y, b) - max(x, a)) + 1 for x, y in iv)"),
    "len_from_s": ("None if r is None else r - t0", "None if r is None else r - s"),
    "region_left": ("bisect.bisect_right(starts, x) - 1)", "bisect.bisect_left(starts, x) - 1)"),
}


def main(py):
    src = open(os.path.join(HERE, "core.py")).read()
    # 写しは一時フォルダに置くので、candidates.py の参照先を元の場所に固定する（固定しないと import の失敗で全部が赤に見える）
    fixed = src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", "HERE = %r" % HERE, 1)
    assert fixed != src
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, m in MUTANTS.items():
            code = fixed
            if m is not None:
                if code.count(m[0]) != 1:
                    sys.exit("変異 %s の当て先が 1 か所でない（当たらない変異は素通りに見える）" % name)
                code = code.replace(m[0], m[1])
            d = os.path.join(tmp, name)
            os.makedirs(d)
            open(os.path.join(d, "core.py"), "w").write(code)
            r = subprocess.run([py, "-I", "-B", os.path.join(d, "core.py"), "--selftest"], capture_output=True, text=True)
            red = r.returncode != 0
            ok_line = "selftest: OK" in r.stdout
            want_red = m is not None
            # 赤は「selftest: FAIL」で落ちたときだけ数える（import の失敗などは測定器の故障）
            if want_red and not (red and "selftest: FAIL" in r.stdout):
                bad.append(name)
            if not want_red and not (not red and ok_line):
                bad.append(name)
            print("%-22s %s" % (name, "赤" if red else "緑"))
    if bad:
        sys.exit("K3: 見込みと違う: %s" % ", ".join(bad))
    print("K3: 変異 %d 個すべて赤・対照緑" % (len(MUTANTS) - 1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.executable)
