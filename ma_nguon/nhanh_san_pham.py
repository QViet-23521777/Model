# -*- coding: utf-8 -*-
"""Nhanh san pham: diem trung binh co trong so tin cay (review cua tai khoan dang ngo
duoc tinh it) + bung no tai khoan moi. Tin hieu chinh: do_chenh = diem_tho - diem_trong_so
(duong = bi bom diem, am = bi dim diem)."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from khoi_dau_lanh import doc_ma_hoa, dem_review_moi_user
from pass_b_gop_user import MIN_USER_REVIEWS

MIN_REVIEW_SP = 20          # it hon -> chua du du lieu de ket luan
NGUONG_CHENH = 0.5          # sao
SHRINK_M = 10.0             # giong shrink_v cua dev_pointintime
TRONG_SO_NGUOI_MOI = 0.5    # user khong co diem hanh vi (< 5 review)
TRONG_SO_SAN = 0.05         # tai khoan dang ngo nhat van con dong gop chut it
CUA_SO_NGAY = 7
NGUONG_BUNG_NO = 5          # >= 5 review nguoi moi trong 7 ngay
NGUONG_TAP_TRUNG = 0.5      # va cua so do chua >= 50% so review nguoi moi cua SP
                            # (khong co dieu kien nay, SP ban chay deu dan cung bi gan co)


def trong_so_tin_cay(diem_toi_da):
    d = np.asarray(diem_toi_da, dtype=np.float64)
    w = np.clip(1.0 - np.nan_to_num(d, nan=0.0), TRONG_SO_SAN, 1.0)
    return np.where(np.isnan(d), TRONG_SO_NGUOI_MOI, w)


def diem_san_pham(ma_sp, rating, trong_so, gmean):
    K = int(ma_sp.max()) + 1
    n = np.bincount(ma_sp, minlength=K)
    tong = np.bincount(ma_sp, weights=rating, minlength=K)
    w_tong = np.bincount(ma_sp, weights=trong_so, minlength=K)
    wr_tong = np.bincount(ma_sp, weights=trong_so * rating, minlength=K)
    # co ve gmean bang SHRINK_M review ao. Ben co trong so, review ao mang trong so trung binh
    # cua SP: trong so deu -> do_chenh = 0 (truoc day review ao co trong so 1 nen SP diem thap
    # bi gan 'bi_dim' chi vi co ve). trong so >= TRONG_SO_SAN > 0 nen w_tb > 0, khong chia 0
    w_tb = w_tong / np.maximum(n, 1)
    diem_tho = (tong + SHRINK_M * gmean) / (n + SHRINK_M)
    diem_ts = (wr_tong + SHRINK_M * w_tb * gmean) / (w_tong + SHRINK_M * w_tb)
    chenh = diem_tho - diem_ts
    nhan = np.where(n < MIN_REVIEW_SP, "chua_du_du_lieu",
                    np.where(chenh >= NGUONG_CHENH, "bi_bom",
                             np.where(chenh <= -NGUONG_CHENH, "bi_dim", "binh_thuong")))
    return pd.DataFrame({"ma_sp": np.arange(K), "n_review": n, "diem_tho": diem_tho,
                         "diem_trong_so": diem_ts, "do_chenh": chenh, "nhan": nhan})


def bung_no_tai_khoan_moi(ma_sp, timestamp_ms, la_nguoi_moi, K):
    """Moi SP: so review cua nguoi moi nhieu nhat trong 1 cua so CUA_SO_NGAY ngay."""
    ket_qua = np.zeros(K, dtype=np.int64)
    s = ma_sp[la_nguoi_moi].astype(np.int64)
    if len(s) == 0:
        return ket_qua
    t = timestamp_ms[la_nguoi_moi] // 1000          # giay: tranh tran int64 khi ghep khoa
    t = t - t.min()
    cua_so = CUA_SO_NGAY * 86_400
    span = int(t.max()) + cua_so + 1                # > cua so -> khong lan sang SP khac
    khoa = np.sort(s * span + t)
    s = khoa // span
    trai = np.searchsorted(khoa, khoa - cua_so, side="left")
    dem = np.arange(len(khoa)) - trai + 1
    np.maximum.at(ket_qua, s, dem)
    return ket_qua


def co_bung_no(max_7ngay, so_nguoi_moi):
    """Bung no = du nhieu review nguoi moi trong 7 ngay VA phan lon review nguoi moi don vao do."""
    tap_trung = max_7ngay / np.maximum(so_nguoi_moi, 1)
    return (max_7ngay >= NGUONG_BUNG_NO) & (tap_trung >= NGUONG_TAP_TRUNG)


def _doc(duong_dan_pass_a, cot_them=()):
    # Cot chuoi doc thang dang tu dien: Home_and_Kitchen ~67 trieu dong, nap ca cot chuoi
    # ton vai GB.
    ma_u, tu_u = doc_ma_hoa(duong_dan_pass_a, "user_id")
    ma_sp, tu_sp = doc_ma_hoa(duong_dan_pass_a, "parent_asin")
    t = pq.read_table(duong_dan_pass_a, columns=["rating", "timestamp", *cot_them])
    them = {c: t.column(c).to_numpy() for c in cot_them}
    rating = t.column("rating").to_numpy()
    ts = t.column("timestamp").to_numpy()
    del t
    return ma_u, tu_u, ma_sp, tu_sp, rating, ts, them


def _tra_theo_tu_dien(tu_dien, khoa, gia_tri):
    """Mang theo ma (0..K-1 cua tu_dien): gia_tri cua khoa trung, NaN neu khong co.
    Tra bang pyarrow, khong doi hang chuc trieu chuoi sang Python."""
    vi_tri = pc.index_in(tu_dien, value_set=pa.array(khoa).cast(tu_dien.type))
    co = vi_tri.is_valid().to_numpy(zero_copy_only=False)
    ra = np.full(len(tu_dien), np.nan)
    ra[co] = np.asarray(gia_tri, dtype=np.float64)[vi_tri.drop_null().to_numpy()]
    return ra


def _diem_theo_ma_user(tu_u, ket_qua_df, cot):
    """Mang theo ma user: gia tri `cot` trong ket_qua_df, NaN neu user khong co."""
    return _tra_theo_tu_dien(tu_u, ket_qua_df["user_id"].astype(str).to_numpy(),
                             ket_qua_df[cot].astype(np.float64).to_numpy())


def tinh_nhanh_san_pham(duong_dan_pass_a, ket_qua_df):
    ma_u, tu_u, ma_sp, tu_sp, rating, ts, _ = _doc(duong_dan_pass_a)
    diem_user = _diem_theo_ma_user(tu_u, ket_qua_df, "diem_toi_da")
    diem = diem_user[ma_u]
    la_moi = np.isnan(diem) | (dem_review_moi_user(ma_u) < MIN_USER_REVIEWS)
    w = trong_so_tin_cay(diem)
    sp = diem_san_pham(ma_sp, rating, w, gmean=float(rating.mean()))
    K = len(sp)
    sp["so_nguoi_moi"] = np.bincount(ma_sp, weights=la_moi, minlength=K).astype(np.int64)
    sp["ti_le_nguoi_moi"] = sp["so_nguoi_moi"] / sp["n_review"]
    sp["max_nguoi_moi_7ngay"] = bung_no_tai_khoan_moi(ma_sp, ts, la_moi, K)
    sp["bung_no_tk_moi"] = co_bung_no(sp["max_nguoi_moi_7ngay"].to_numpy(),
                                      sp["so_nguoi_moi"].to_numpy())
    sp.insert(0, "parent_asin", tu_sp.to_pylist())
    return sp.drop(columns="ma_sp")


def phan_tich_demoter(duong_dan_pass_a, ket_qua_df, sp_df):
    """Review cham thap hon diem SP (signed < 0) cua user bi co T_demoter: bao nhieu phan tram
    roi vao SP bi bom? Cao hon han nhom con lai -> dau hieu bao nham nguoi that."""
    ma_u, tu_u, ma_sp, tu_sp, _, _, them = _doc(duong_dan_pass_a, ["signed_pointintime"])
    co = _diem_theo_ma_user(tu_u, ket_qua_df, "T_demoter_co_can_xem")[ma_u] == 1.0
    bi_bom = _tra_theo_tu_dien(tu_sp, sp_df["parent_asin"].astype(str).to_numpy(),
                               sp_df["nhan"].eq("bi_bom").to_numpy(np.float64))[ma_sp] == 1.0
    am = them["signed_pointintime"] < 0

    def _ti_le(mask):
        return float(bi_bom[mask].mean()) if mask.any() else float("nan")

    return {"so_review_am_cua_demoter": int((am & co).sum()),
            "ti_le_tren_sp_bi_bom_demoter": _ti_le(am & co),
            "ti_le_tren_sp_bi_bom_con_lai": _ti_le(am & ~co)}
