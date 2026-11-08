"""Excerpt of candidates.py: the two definitions the stall-shape scripts use, copied verbatim.

The original file also imports analyses of a study that is not published, so only these two definitions
are released. SHA-256 of the original file: 03b033183493d8cd6f60391aa1bb030d89750dab1406df0101ca51c93c4199e8
"""
import bisect
import math

class Acc:
    """(時刻, 値) を時刻の昇順に持ち、at(t) で時刻 t 以下の値の nearest-rank 分位を返す。t は単調非減少で呼ぶ。"""

    def __init__(self, pooled, q, warm):
        self.pooled = sorted(pooled)
        self.q, self.warm = q, warm
        self.vals = sorted({v for _, v in self.pooled})
        self.tree = [0] * (len(self.vals) + 1)
        self.pos = self.n = 0
        self.last_t = None

    def _add(self, v):
        i = bisect.bisect_left(self.vals, v) + 1
        while i < len(self.tree):
            self.tree[i] += 1
            i += i & -i
        self.n += 1

    def _kth(self, k):
        pos, rem = 0, k
        pw = 1 << max(0, len(self.vals).bit_length() - 1)
        while pw:
            nx = pos + pw
            if nx < len(self.tree) and self.tree[nx] < rem:
                pos, rem = nx, rem - self.tree[nx]
            pw >>= 1
        return self.vals[pos]

    def at(self, t):
        if self.last_t is not None and t < self.last_t:
            raise ValueError("Acc は時刻の昇順で引く")
        self.last_t = t
        while self.pos < len(self.pooled) and self.pooled[self.pos][0] <= t:
            self._add(self.pooled[self.pos][1])
            self.pos += 1
        if self.n == 0 or self.n < self.warm:
            return None
        rank = min(max(math.ceil(self.q * self.n), 1), self.n)
        return self._kth(rank), self.n


def gap_pool(ts, intervals):
    """同じ観測区間に入る隣り合う 2 点の (2 点目の時刻, 間隔)。全区間からプール（stall_onset.rs の pooled_theta_gaps）。"""
    out = []
    for a, b in intervals:
        x = ts[bisect.bisect_left(ts, a):bisect.bisect_right(ts, b)]
        out += [(y, y - p) for p, y in zip(x, x[1:])]
    return out
