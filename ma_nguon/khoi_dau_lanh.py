# -*- coding: utf-8 -*-
"""Khoi dau lanh: user < MIN_USER_REVIEWS bi Pass B loai nen khong co diem hanh vi.
Dem xem bao nhieu user / review roi vao nhom nay (so lieu cho bao cao)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pass_b_gop_user import MIN_USER_REVIEWS


def ma_hoa(cot):
    """Chuoi -> ma so nguyen 0..K-1 (tiet kiem RAM hon giu chuoi trong pandas)."""
    d = cot.combine_chunks().dictionary_encode()
    return d.indices.to_numpy(zero_copy_only=False).astype(np.int32), d.dictionary


def doc_ma_hoa(duong_dan_parquet, cot):
    """Nhu ma_hoa(read(cot)) nhung parquet tra thang dang tu dien: chi giu chuoi DUY NHAT,
    khong nap ca cot chuoi (Electronics: ~7 GB -> vai tram MB cho cot user_id)."""
    t = pq.read_table(duong_dan_parquet, columns=[cot], read_dictionary=[cot]).unify_dictionaries()
    c = t.column(0)
    if c.num_chunks == 0:
        return np.zeros(0, dtype=np.int32), pa.array([], type=pa.string())
    ma = np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in c.chunks])
    return ma.astype(np.int32), c.chunk(0).dictionary


def dem_review_moi_user(ma_user):
    """So review cua user, gan lai cho tung dong."""
    return np.bincount(ma_user)[ma_user]


def thong_ke_khoi_dau_lanh(duong_dan_pass_a, nguong=MIN_USER_REVIEWS):
    ma, _ = doc_ma_hoa(duong_dan_pass_a, "user_id")
    t = pq.read_table(duong_dan_pass_a, columns=["rating", "verified_purchase"])
    dem_user = np.bincount(ma)
    it = dem_review_moi_user(ma) < nguong
    rating = t.column("rating").to_numpy()
    ver = t.column("verified_purchase").to_numpy(zero_copy_only=False).astype(bool)
    del t

    def _ti_le(x):
        return float(x.mean()) if len(x) else float("nan")

    return {
        "nguong": int(nguong),
        "tong_user": int(len(dem_user)),
        "user_it_review": int((dem_user < nguong).sum()),
        "ti_le_user_it_review": float((dem_user < nguong).mean()),
        "tong_review": int(len(ma)),
        "ti_le_review_tu_user_it": float(it.mean()),
        "phan_bo_so_review": {str(k): int((dem_user == k).sum()) for k in range(1, nguong)},
        "ti_le_verified_user_it": _ti_le(ver[it]),
        "ti_le_verified_user_du": _ti_le(ver[~it]),
        "ti_le_5sao_user_it": _ti_le(rating[it] == 5),
        "ti_le_5sao_user_du": _ti_le(rating[~it] == 5),
        "ti_le_1sao_user_it": _ti_le(rating[it] == 1),
        "ti_le_1sao_user_du": _ti_le(rating[~it] == 1),
    }
