# -*- coding: utf-8 -*-
"""Dac trung nang cao: runs test, chu ky FFT, do doc lech theo thoi gian, ALS bias."""
import numpy as np
import pandas as pd
from scipy import stats


def runs_pvalue(gia_tri):
    x = np.asarray(gia_tri, dtype=float)
    if len(x) < 2:
        return 1.0
    trung_vi = np.median(x)
    con_lai = x[x != trung_vi]
    if len(con_lai) < 2:
        return 1.0
    dau = con_lai > trung_vi
    n1 = int(dau.sum())
    n2 = len(dau) - n1
    if n1 == 0 or n2 == 0:
        return 1.0
    n = n1 + n2
    so_runs = 1 + int(np.sum(dau[1:] != dau[:-1]))
    ky_vong = 2 * n1 * n2 / n + 1
    phuong_sai = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))
    if phuong_sai <= 0:
        return 1.0
    z = (so_runs - ky_vong) / np.sqrt(phuong_sai)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def fft_nang_luong_chu_ky(gaps):
    g = np.asarray(gaps, dtype=float)
    if len(g) < 4:
        return 0.0
    g = g - g.mean()
    if np.allclose(g, 0):
        return 0.0
    pho = np.abs(np.fft.rfft(g)) ** 2
    pho = pho[1:]  # bo thanh phan DC (tan so 0)
    if len(pho) == 0 or pho.sum() <= 0:
        return 0.0
    return float(pho.max() / pho.sum())


def do_doc_lech_theo_thoigian(signed_devs, ts_ngay):
    x = np.asarray(ts_ngay, dtype=float)
    y = np.asarray(signed_devs, dtype=float)
    # R27.1: np.ptp thay np.allclose - allclose (rtol 1e-5) tren ~19,000 ngay coi
    # cac review cach nhau vai gio la "trung nhau" -> do doc 0 cho bot.
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    # tru trung binh x truoc khi fit: cung he so goc, on dinh so hoc hon
    he_so = np.polyfit(x - x.mean(), y, 1)
    return float(he_so[0])


def fit_als_bias_ma(ma_u, so_u, ma_p, so_p, r, so_vong=5, reg=5.0):
    """Loi ALS tren ma so nguyen (0..so_u-1, 0..so_p-1) -> (user_bias, prod_bias, gmean)
    dang mang. Pass B goi truc tiep de khong phai dung chuoi/dict cho moi user."""
    r = np.asarray(r, dtype=np.float64)
    gmean = float(r.sum()) / len(r)
    mau_u = reg + np.bincount(ma_u, minlength=so_u)
    mau_p = reg + np.bincount(ma_p, minlength=so_p)

    user_bias = np.zeros(so_u)
    prod_bias = np.zeros(so_p)
    for _ in range(so_vong):
        user_bias = np.bincount(ma_u, weights=r - gmean - prod_bias[ma_p], minlength=so_u) / mau_u
        prod_bias = np.bincount(ma_p, weights=r - gmean - user_bias[ma_u], minlength=so_p) / mau_p
    return user_bias, prod_bias, gmean


def fit_als_bias_mang(users, prods, ratings, so_vong=5, reg=5.0, chi_user=None):
    """ALS baseline vector hoa (R24): pd.factorize + np.bincount.

    Cung cong thuc voi ban vong lap cu: moi vong cap nhat toan bo user_bias
    (dung prod_bias vong truoc) roi toan bo prod_bias (dung user_bias vua cap nhat),
    bias = tong(r - gmean - bias_ben_kia) / (reg + so_rating).

    chi_user: neu truyen, chi tra dict bias cho cac user nay (va prod_bias = None) -
    tranh dung dict Python cho moi user (Electronics ~18 trieu user, phan lon bi loai).
    """
    r = np.asarray(ratings, dtype=np.float64)
    ma_u, ds_u = pd.factorize(pd.Series(users))
    ma_p, ds_p = pd.factorize(pd.Series(prods))
    user_bias, prod_bias, gmean = fit_als_bias_ma(
        ma_u, len(ds_u), ma_p, len(ds_p), r, so_vong=so_vong, reg=reg)
    if chi_user is not None:
        chi_so_u = pd.Index(ds_u)
        giu = chi_so_u.isin(chi_user)
        return dict(zip(chi_so_u[giu].tolist(), user_bias[giu].tolist())), None, gmean
    return (dict(zip(ds_u.tolist(), user_bias.tolist())),
            dict(zip(ds_p.tolist(), prod_bias.tolist())),
            gmean)


def fit_als_bias(dong, so_vong=5, reg=5.0):
    """Giu giao dien cu: dong = danh sach (user, product, rating)."""
    users, prods, ratings = zip(*dong)
    return fit_als_bias_mang(list(users), list(prods), np.asarray(ratings, dtype=np.float64),
                             so_vong=so_vong, reg=reg)
