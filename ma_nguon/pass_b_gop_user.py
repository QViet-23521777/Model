# -*- coding: utf-8 -*-
"""Pass B: gop review (da qua Pass A) theo user_id, tinh toan bo dac trung,
loc user duoi 5 review (5-core, giong v1).

Doc file 2 luot, khong nap ca file vao RAM (Home_and_Kitchen ~67 trieu review):
  luot 1 - chi 3 cot user_id/parent_asin/rating: thong ke san pham, gmean, dem review
           moi user, ALS bias;
  luot 2 - doc theo lo; file Pass A da sap theo user_id nen review cua moi user nam
           lien nhau -> xu ly tung user, ghi dac trung ra parquet theo lo."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dac_trung_nguoi_dung import user_features
from dac_trung_nang_cao import fit_als_bias_ma
from pass_a_thoigian import dung_bo_cap_phat_he_thong

MIN_USER_REVIEWS = 5
# Ghi dac trung ra parquet theo lo: moi user ~75 dac trung (~5 KB dict Python), giu
# het 1,9 trieu user Electronics trong RAM se ton ~10 GB.
KICH_THUOC_LO = 50_000
KICH_THUOC_DOC = 500_000  # so dong moi lo doc o luot 2
COT_CHUOI = ["user_id", "parent_asin"]  # luot 1 doc dang ma hoa (Categorical)
COT_THONG_KE = ["user_id", "parent_asin", "rating"]

COT_CAN_PASS_B = [
    "user_id", "parent_asin", "timestamp", "rating", "helpful_vote", "verified_purchase",
    "text_len", "has_image", "title_hash",
    "dev_pointintime", "signed_pointintime", "vi_tri_sp", "du_moc",
]


def tinh_thong_ke_san_pham(df):
    g = df.groupby("parent_asin", observed=True)["rating"].agg(["mean", "count"])
    return dict(zip(g.index.astype(str).tolist(),
                    zip(g["mean"].astype(float).tolist(), g["count"].astype(int).tolist())))


def _thong_ke_luot_1(duong_dan):
    """Luot 1 doc theo lo, chi pyarrow + numpy. File Pass A da sap theo user_id nen ma
    user = so thu tu doan lien tiep (khong can tu dien ~18 trieu chuoi); ma san pham qua
    1 dict {parent_asin: ma}. Truoc day doc ca bang + to_pandas -> dinh RAM 8.15 GB tren
    Electronics (pandas giai ma lai ~44 trieu chuoi de hop nhat Categorical)."""
    ma_u_ph, ma_p_ph, r_ph, ten_doan_ph = [], [], [], []
    ma_sp = {}
    so_doan, user_cuoi = 0, None
    for lo in pq.ParquetFile(duong_dan).iter_batches(batch_size=KICH_THUOC_DOC,
                                                     columns=COT_THONG_KE):
        u = lo.column("user_id").to_numpy(zero_copy_only=False)
        if len(u) > 1 and (u[1:] < u[:-1]).any() or (user_cuoi is not None and u[0] < user_cuoi):
            raise ValueError("File Pass A chua sap theo user_id - chay lai Pass A ban moi")
        moi = np.empty(len(u), dtype=bool)
        moi[0] = u[0] != user_cuoi
        moi[1:] = u[1:] != u[:-1]
        ma_u_ph.append((so_doan - 1 + np.cumsum(moi)).astype(np.int64))
        ten_doan_ph.append(pa.array(u[moi], type=pa.large_string()))
        so_doan += int(moi.sum())
        user_cuoi = u[-1]
        del u, moi

        sp = pc.dictionary_encode(lo.column("parent_asin"))
        anh_xa = np.fromiter((ma_sp.setdefault(s, len(ma_sp)) for s in sp.dictionary.to_pylist()),
                             dtype=np.int64)
        ma_p_ph.append(anh_xa[sp.indices.to_numpy(zero_copy_only=False)])
        r_ph.append(lo.column("rating").to_numpy(zero_copy_only=False).astype(np.float64))

    ma_u = np.concatenate(ma_u_ph) if ma_u_ph else np.empty(0, dtype=np.int64)
    ma_p = np.concatenate(ma_p_ph) if ma_p_ph else np.empty(0, dtype=np.int64)
    r = np.concatenate(r_ph) if r_ph else np.empty(0, dtype=np.float64)
    del ma_u_ph, ma_p_ph, r_ph

    dem_p = np.bincount(ma_p, minlength=len(ma_sp))
    tong_p = np.bincount(ma_p, weights=r, minlength=len(ma_sp))
    prod = dict(zip(ma_sp.keys(), zip((tong_p / dem_p).tolist(), dem_p.tolist())))
    del ma_sp

    dem_u = np.bincount(ma_u, minlength=so_doan)
    so_bi_loai = int((dem_u < MIN_USER_REVIEWS).sum())
    chi_so_giu = np.flatnonzero(dem_u >= MIN_USER_REVIEWS)
    ten_giu = pa.chunked_array(ten_doan_ph, type=pa.large_string()).take(
        pa.array(chi_so_giu)).to_pylist()
    del ten_doan_ph

    if len(r) == 0:
        return prod, float("nan"), {}, set(), so_bi_loai

    # R24: ALS tren TOAN BO review (ke ca user bi loai), chi dung dict cho user giu lai
    user_bias, _, gmean = fit_als_bias_ma(ma_u, so_doan, ma_p, len(dem_p), r,
                                          so_vong=5, reg=5.0)
    als_bias = dict(zip(ten_giu, user_bias[chi_so_giu].tolist()))
    return prod, gmean, als_bias, set(ten_giu), so_bi_loai


class _GhiTheoLo:
    def __init__(self, duong_dan):
        self.duong_dan, self.lo, self.ghi, self.luoc_do = duong_dan, [], None, None

    def them(self, dong):
        self.lo.append(dong)
        if len(self.lo) >= KICH_THUOC_LO:
            self.xa()

    def xa(self):
        if not self.lo:
            return
        bang = pa.Table.from_pandas(pd.DataFrame(self.lo), preserve_index=False)
        self.lo = []
        if self.ghi is None:
            self.luoc_do = bang.schema
            self.ghi = pq.ParquetWriter(self.duong_dan, self.luoc_do, compression="zstd")
        self.ghi.write_table(bang.cast(self.luoc_do))

    def dong(self):
        self.xa()
        if self.ghi is None:  # khong co user nao du 5 review
            pd.DataFrame().to_parquet(self.duong_dan)
        else:
            self.ghi.close()


def _dong_cua_doan(cot, dau, cuoi):
    return [dict(zip(COT_CAN_PASS_B, t)) for t in zip(*(cot[c][dau:cuoi] for c in COT_CAN_PASS_B))]


def _duyet_theo_user(duong_dan, giu):
    """Sinh danh sach dong (dict) cua tung user trong `giu`, theo thu tu file.
    File phai sap theo user_id (Pass A ghi nhu vay); neu khong -> ValueError.
    Moi lo chuyen sang list Python 1 lan theo cot roi cat lat theo user: to_dict
    ("records") cua pandas tren tung nhom chiem 62% thoi gian Pass B."""
    du = None  # cac dong cua user cuoi lo truoc (co the con tiep o lo sau)
    for lo in pq.ParquetFile(duong_dan).iter_batches(batch_size=KICH_THUOC_DOC,
                                                     columns=COT_CAN_PASS_B):
        cot = {c: lo.column(c).to_pylist() for c in COT_CAN_PASS_B}
        if du is not None:
            cot = {c: du[c] + cot[c] for c in COT_CAN_PASS_B}
        u = cot["user_id"]
        mang_u = np.asarray(u, dtype=object)
        if len(u) > 1 and (mang_u[1:] < mang_u[:-1]).any():
            raise ValueError("File Pass A chua sap theo user_id - chay lai Pass A ban moi")
        dau = [0] + (np.flatnonzero(mang_u[1:] != mang_u[:-1]) + 1).tolist()
        del mang_u
        for s, e in zip(dau[:-1], dau[1:]):
            if u[s] in giu:
                yield _dong_cua_doan(cot, s, e)
        du = {c: cot[c][dau[-1]:] for c in COT_CAN_PASS_B}
    if du is not None and du["user_id"] and du["user_id"][0] in giu:
        yield _dong_cua_doan(du, 0, len(du["user_id"]))


def chay_pass_b(duong_dan_augmented_parquet, duong_dan_ra_parquet):
    dung_bo_cap_phat_he_thong()

    # luot 1: thong ke toan cuc tu 3 cot (R24: bo user < 5 review, ALS tren toan bo)
    prod, gmean, als_bias, giu, so_bi_loai = _thong_ke_luot_1(duong_dan_augmented_parquet)

    # luot 2: doc theo lo, xu ly tung user lien tiep
    ghi = _GhiTheoLo(duong_dan_ra_parquet)
    so_user = 0
    for rows in _duyet_theo_user(duong_dan_augmented_parquet, giu):
        ghi.them(user_features(rows, prod, gmean, als_bias))
        so_user += 1
    ghi.dong()
    return {"so_user_giu": so_user, "so_user_bi_loai": so_bi_loai}
