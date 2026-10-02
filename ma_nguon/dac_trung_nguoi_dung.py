# -*- coding: utf-8 -*-
"""user_features(): toan bo dac trung hanh vi cho 1 nguoi dung.
Ke thua nguyen 56 dac trung tu D:\\Dataset\\code\\1_extract_full.py,
them 8 dac trung "re" va cac khoa nang cao (dev_pointintime, als bias, v.v)."""
from collections import Counter

import numpy as np

from tien_ich import entropy, top1_share, max_streak, rolling_max
from dac_trung_nang_cao import runs_pvalue, fft_nang_luong_chu_ky, do_doc_lech_theo_thoigian

MS_DAY = 86_400_000
MS_HOUR = 3_600_000
MIN_PROD_REVIEWS = 10  # nguong pop/rare (do lech dung co du_moc tu Pass A, cung nguong)
EARLY_N = 20  # nguong "con moi" trong lich su rieng cua san pham


def _streak_gia_tri_giong_nhau(gia_tri):
    best = cur = 0
    truoc = object()
    for v in gia_tri:
        cur = cur + 1 if v == truoc else 1
        best = max(best, cur)
        truoc = v
    return best


def user_features(rows, prod, gmean, als_bias=None):
    als_bias = als_bias or {}
    n = len(rows)

    ts = np.array([r["timestamp"] for r in rows], dtype=np.int64)
    order = np.argsort(ts)
    ts = ts[order]
    rows = [rows[i] for i in order]

    rt = np.array([r["rating"] for r in rows], dtype=np.float32)
    hv = np.array([r["helpful_vote"] for r in rows], dtype=np.float32)
    ver = np.array([bool(r["verified_purchase"]) for r in rows])

    days = ts // MS_DAY
    hours = (ts % MS_DAY) // MS_HOUR
    minutes_in_hour = (ts % MS_HOUR) // 60_000
    dow = (days + 4) % 7
    day_cnt = Counter(days.tolist())
    hour_cnt = Counter(hours.tolist())
    minute_cnt = Counter(minutes_in_hour.tolist())
    dow_cnt = Counter(dow.tolist())

    span = float((ts[-1] - ts[0]) / MS_DAY)
    gaps = np.diff(ts) / MS_DAY if n > 1 else np.array([0.0])
    gap_mean = float(gaps.mean())

    # R23: giong v1 - chi review cua san pham da co >= 10 review (tinh TRUOC thoi diem
    # do, co du_moc tu Pass A) moi vao mang do lech; khong co review nao -> [0.0] nhu v1.
    du_moc = np.array([bool(r["du_moc"]) for r in rows], dtype=bool)
    devs_tat_ca = np.array([r["dev_pointintime"] for r in rows], dtype=np.float64)
    signed_tat_ca = np.array([r["signed_pointintime"] for r in rows], dtype=np.float64)
    devs = devs_tat_ca[du_moc] if du_moc.any() else np.array([0.0])
    signed = signed_tat_ca[du_moc] if du_moc.any() else np.array([0.0])
    dev_trend_slope = do_doc_lech_theo_thoigian(
        signed_tat_ca[du_moc].tolist(), (ts[du_moc] / MS_DAY).tolist())
    vi_tri_sp = np.array([r["vi_tri_sp"] for r in rows], dtype=np.int64)

    # R19: Pass A chi giu truong dan xuat thay cho title/text/images
    text_len = np.array([r["text_len"] for r in rows], dtype=np.int64)
    title_hash = {r["title_hash"] for r in rows}
    has_image = np.array([bool(r["has_image"]) for r in rows], dtype=bool)

    pop, rare = [], 0
    for r in rows:
        pm = prod.get(r["parent_asin"])
        if pm and pm[1] >= MIN_PROD_REVIEWS:
            pop.append(pm[1])
        else:
            rare += 1
    pop = np.array(pop) if pop else np.array([0.0])

    burst_day, burst_n = day_cnt.most_common(1)[0]
    in_burst = days == burst_day
    burst_prods = len({rows[i]["parent_asin"] for i in range(n) if in_burst[i]})

    parent = [r["parent_asin"] for r in rows]
    n_uniq = len(set(parent))

    ngay_ban_ron = [d for d, c in day_cnt.items() if c >= 3]
    if ngay_ban_ron:
        ti_le = []
        for d in ngay_ban_ron:
            sp_trong_ngay = {parent[i] for i in range(n) if days[i] == d}
            ti_le.append(len(sp_trong_ngay) / int(day_cnt[d]))
        busy_day_prod_div_avg = float(np.mean(ti_le))
    else:
        busy_day_prod_div_avg = 1.0

    burst_2day = int(rolling_max(ts, 2 * MS_DAY))
    burst_7day = int(rolling_max(ts, 7 * MS_DAY))

    return {
        "user_id": rows[0]["user_id"],
        "n_reviews": n,

        "A_max_burst": int(burst_n),
        "A_unverified_rate": float(1 - ver.mean()),
        "A_avg_rating_dev": float(devs.mean()),

        "B_burst_2day": burst_2day,
        "B_burst_7day": burst_7day,
        "B_burst_2day_ti_le": burst_2day / n,
        "B_burst_7day_ti_le": burst_7day / n,
        "B_burst_share": float(burst_n / n),
        "B_n_busy_days": int(sum(1 for v in day_cnt.values() if v >= 3)),
        "B_n_active_days": len(day_cnt),
        "B_rev_per_actday": float(n / len(day_cnt)),
        "B_active_days": span,
        "B_density": float(n / (span + 1)),
        "B_longest_gap": float(gaps.max()),
        "B_active_span_ratio": float(len(day_cnt) / (span + 1)),

        "B_gap_mean": gap_mean,
        "B_gap_std": float(gaps.std()),
        "B_gap_min": float(gaps.min()),
        "B_gap_median": float(np.median(gaps)),
        "B_gap_cv": float(gaps.std() / (gap_mean + 1e-9)),
        "B_gap_lt1h_rate": float(np.mean(gaps < 1 / 24)),
        "B_gap_lt1d_rate": float(np.mean(gaps < 1)),
        "B_gap_periodicity_power": fft_nang_luong_chu_ky(gaps.tolist()),

        "B_hour_entropy": float(entropy(hour_cnt)),
        "B_hour_top1_share": float(top1_share(hour_cnt)),
        "B_night_rate": float(np.mean((hours >= 0) & (hours < 6))),
        "B_workhour_rate": float(np.mean((hours >= 9) & (hours < 18))),
        "B_dow_entropy": float(entropy(dow_cnt)),
        "B_dow_top1_share": float(top1_share(dow_cnt)),
        "B_weekend_rate": float(np.mean((dow == 0) | (dow == 6))),
        "B_minute_top1_share": float(top1_share(minute_cnt)),

        "B_rating_mean": float(rt.mean()),
        "B_rating_std": float(rt.std()),
        "B_rating_entropy": float(entropy(Counter(rt.tolist()))),
        "B_rating_top1_share": float(top1_share(Counter(rt.tolist()))),
        "B_extreme_rate": float(np.mean((rt == 1) | (rt == 5))),
        "B_rate_5star": float(np.mean(rt == 5)),
        "B_rate_1star": float(np.mean(rt == 1)),
        "B_rate_mid": float(np.mean((rt >= 2) & (rt <= 4))),
        "B_rating_streak_same": int(_streak_gia_tri_giong_nhau(rt.tolist())),
        "B_rating_runs_pvalue": runs_pvalue(rt.tolist()),

        "B_dev_std": float(devs.std()),
        "B_dev_max": float(devs.max()),
        "B_dev_median": float(np.median(devs)),
        "B_pos_dev_rate": float(np.mean(signed > 0.5)),
        "B_neg_dev_rate": float(np.mean(signed < -0.5)),
        "B_dev_sign_bias": float(abs(np.mean(np.sign(signed)))),
        "B_signed_dev_mean": float(signed.mean()),
        "B_promoter_dev": float(np.mean(np.maximum(0, signed))),
        "B_demoter_dev": float(np.mean(np.maximum(0, -signed))),
        "B_dev_raw_mean": float(np.mean(np.abs(signed))),
        "B_dev_streak_pos": int(max_streak((signed > 0.5).tolist())),
        "B_dev_streak_neg": int(max_streak((signed < -0.5).tolist())),
        "B_dev_trend_slope": dev_trend_slope,
        "B_early_review_rate": float(np.mean(vi_tri_sp < EARLY_N)),
        "B_user_bias_als": float(als_bias.get(rows[0]["user_id"], 0.0)),

        "B_prod_pop_mean": float(pop.mean()),
        "B_prod_pop_median": float(np.median(pop)),
        "B_prod_pop_std": float(pop.std()),
        "B_rare_prod_rate": float(rare / n),
        "B_n_uniq_prod": int(n_uniq),
        "B_variant_rate": float((n - n_uniq) / n),
        "B_burst_prod_div": float(burst_prods / burst_n),
        # R18: huong nghi ngo = IT da dang san pham trong ngay dot bien
        "B_burst_prod_lap": float(1 - burst_prods / burst_n),
        "B_busy_day_prod_div_avg": busy_day_prod_div_avg,

        "B_unver_streak": int(max_streak((~ver).tolist())),
        "B_avg_helpful": float(hv.mean()),
        "B_zero_helpful_rate": float(np.mean(hv == 0)),
        "B_helpful_max": float(hv.max()),
        "B_helpful_std": float(hv.std()),
        "B_zero_helpful_streak": int(max_streak((hv == 0).tolist())),

        "B_text_len_mean": float(np.mean(text_len)),
        "B_text_len_min": float(text_len.min()),
        "B_dup_title_rate": float(1 - len(title_hash) / n),
        "B_has_image_rate": float(np.mean(has_image)),
    }
