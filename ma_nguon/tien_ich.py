# -*- coding: utf-8 -*-
"""Ham tien ich thuan, khong I/O - dung chung cho cac module dac trung."""
import math


def entropy(counter):
    """Do phan tan. 0 = tap trung mot cho, cang cao cang deu."""
    tot = sum(counter.values())
    if tot <= 1:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values())


def top1_share(counter):
    """Ti le cua gia tri pho bien nhat. Cao = don dieu."""
    tot = sum(counter.values())
    return max(counter.values()) / tot if tot else 0.0


def max_streak(bools):
    """Chuoi True dai nhat lien tiep."""
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def rolling_max(sorted_ts, window_ms):
    """So diem nhieu nhat trong mot cua so truot window_ms (ts da sort tang dan)."""
    n = len(sorted_ts)
    best, j = 0, 0
    for i in range(n):
        while sorted_ts[i] - sorted_ts[j] > window_ms:
            j += 1
        best = max(best, i - j + 1)
    return best
