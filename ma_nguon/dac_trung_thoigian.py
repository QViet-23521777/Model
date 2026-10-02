# -*- coding: utf-8 -*-
"""Sua loi ro ri thoi gian: prod_mean chi duoc tinh tu review co timestamp
truoc thoi diem dang xet (khong duoc nhin thay tuong lai)."""
import numpy as np
import pandas as pd

# Giong v1 (MIN_PROD_REVIEWS trong D:\Dataset\code\1_extract_full.py): san pham it hon
# 10 review (tinh TRUOC thoi diem dang xet) khong du tin cay de lam moc do lech.
MIN_PROD_REVIEWS = 10


def tinh_lech_pointintime_mang(parent_asin, rating, gmean, shrink_v=10.0,
                               min_prod_reviews=MIN_PROD_REVIEWS):
    """Ban vector hoa (R19). Dau vao da sap xep theo thoi gian (on dinh).

    Tra ve dict mang cung thu tu dau vao: dev_pointintime, signed_pointintime,
    vi_tri_sp (= so review truoc do cua san pham), du_moc (vi_tri_sp >= 10).
    Cong thuc giong het vong lap goc: pmean = shrinkage(gmean, trung binh cac
    review TRUOC) - tong truoc do cong tuan tu theo thoi gian trong tung san pham.
    """
    # Tiet kiem RAM (Electronics ~41 trieu dong): tinh tai cho, giai phong mang tam
    # ngay khi dung xong. parent_asin la ma so nguyen (Pass A) thi bo qua factorize.
    r = np.asarray(rating, dtype=np.float64)
    n = len(r)
    ma = np.asarray(parent_asin)
    if not np.issubdtype(ma.dtype, np.integer):
        ma = pd.factorize(pd.Series(parent_asin))[0]

    thu_tu = np.argsort(ma, kind="stable")  # gom nhom SP, giu thu tu thoi gian trong nhom
    ma_s = ma[thu_tu]
    bat_dau = np.ones(n, dtype=bool)
    if n > 1:
        np.not_equal(ma_s[1:], ma_s[:-1], out=bat_dau[1:])
    del ma_s, ma
    n_truoc_s = np.arange(n, dtype=np.int64)
    dau_nhom = np.where(bat_dau, n_truoc_s, 0)
    del bat_dau
    np.maximum.accumulate(dau_nhom, out=dau_nhom)
    n_truoc_s -= dau_nhom

    # tong rating TRUOC do trong cung san pham = cumsum loai tru (exclusive) tru di
    # cumsum loai tru tai dau nhom. Rating Amazon la boi so 0.5 nen moi tong deu chinh
    # xac trong float64 -> gia tri trung het vong lap goc.
    r_s = r[thu_tu]
    tong_s = np.cumsum(r_s)
    tong_s -= r_s
    tong_s -= tong_s[dau_nhom]
    del dau_nhom

    n_f = n_truoc_s.astype(np.float64)
    tb = tong_s / np.maximum(n_f, 1.0)
    del tong_s
    pmean_s = n_f * tb
    del tb
    pmean_s += shrink_v * gmean
    n_f += shrink_v
    pmean_s /= n_f
    del n_f
    pmean_s[n_truoc_s == 0] = gmean

    d_s = r_s - pmean_s
    del r_s
    dev_s = np.zeros(n, dtype=np.float64)
    m = d_s > 0
    dev_s[m] = d_s[m] / ((5.0 - pmean_s[m]) + 0.1)
    m = d_s < 0
    dev_s[m] = -d_s[m] / ((pmean_s[m] - 1.0) + 0.1)
    del m, pmean_s

    def ve_thu_tu_goc(x):
        ra = np.empty_like(x)
        ra[thu_tu] = x
        return ra

    n_truoc = ve_thu_tu_goc(n_truoc_s)
    del n_truoc_s
    return {
        "dev_pointintime": ve_thu_tu_goc(dev_s),
        "signed_pointintime": ve_thu_tu_goc(d_s),
        "vi_tri_sp": n_truoc,
        "du_moc": n_truoc >= min_prod_reviews,
    }


def tinh_lech_pointintime(reviews_theo_thoigian, gmean, shrink_v=10.0):
    """Ban dict (giu cho tuong thich) - goi ban mang de dam bao cung gia tri."""
    reviews = list(reviews_theo_thoigian)
    if not reviews:
        return []
    kq = tinh_lech_pointintime_mang(
        [r["parent_asin"] for r in reviews], [r["rating"] for r in reviews], gmean, shrink_v)
    return [
        {
            **r,
            "dev_pointintime": float(kq["dev_pointintime"][i]),
            "signed_pointintime": float(kq["signed_pointintime"][i]),
            "vi_tri_sp": int(kq["vi_tri_sp"][i]),
            "du_moc": bool(kq["du_moc"][i]),
        }
        for i, r in enumerate(reviews)
    ]
