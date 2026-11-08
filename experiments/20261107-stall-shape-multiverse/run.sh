#!/usr/bin/env bash
# 停滞の形を濃くする 3 つの計算（conditions.md）。データと中間は repo の外。results には集計値だけを書く。
# Step 0（capabilities_gate.py）は当たらない（GSE のログ用・俯瞰と同じ逸脱）。
# 手元の Mac（8 並列）で run_configs が約 1 時間 50 分、analyze が約 10 分。結果は並列の数によらない。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_DIR:=$HOME/gse-public-data}"
: "${CACHE:=$DATA_DIR/paper2_work}"
: "${PY:=$DATA_DIR/venv-paper2/bin/python}"
: "${JOBS:=8}"
SC="$EXP_DIR/scripts"
OUT="$EXP_DIR/results"
SEED="$(cat "$EXP_DIR/SEED")"
avail_kb="$(df -k "$DATA_DIR" | awk 'NR==2{print $4}')"
[ "$avail_kb" -gt $((2 * 1024 * 1024)) ] || { echo "空きが 2 GB を切っている（中間は約 350 MB）" >&2; exit 1; }
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
# 中間（書き手の識別を含む units.pkl など）を repo の中に置かない
python3 -I -c 'import os,sys;a,b=(os.path.realpath(x) for x in sys.argv[1:]);sys.exit(1 if a==b or a.startswith(b+os.sep) else 0)' "$CACHE" "$REPO_ROOT" \
  || { echo "CACHE が repo の中を指している" >&2; exit 1; }
mkdir -p "$CACHE" "$OUT"
"$PY" -I -B "$SC/core.py" --selftest                             # K2
"$PY" -I -B "$SC/analyze.py" --selftest                          # K2（φ・M3）
"$PY" -I -B "$SC/mutate.py" "$PY"                                    # K3
"$PY" -I -B "$SC/prep.py" "$DATA_DIR" "$CACHE"                       # D・K4（照合が合わなければ止まる）
"$PY" -I -B "$SC/k1_match.py" "$DATA_DIR" "$CACHE" "$OUT/k1_match.json"   # K1（B0 が replay と違えば止まる）
t0=$(date +%s)
"$PY" -I -B "$SC/run_configs.py" "$CACHE" "$JOBS"                    # C3 の 90 通り
secs=$(( $(date +%s) - t0 ))
"$PY" -I -B "$SC/c3b.py" "$DATA_DIR" "$CACHE"                        # C3b
"$PY" -I -B "$SC/analyze.py" "$CACHE" "$OUT" "$SEED"                 # C1・C2・C3・§J
cp "$CACHE/prep_summary.json" "$OUT/prep_summary.json"
"$PY" -I -B "$SC/provenance.py" "$CACHE" "$OUT/provenance.json" "$JOBS" "$SEED" "$secs"
