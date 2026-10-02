# -*- coding: utf-8 -*-
"""Pass A: doc file jsonl dang streaming, chi giu tuple toi thieu (SPEC muc 6),
sap xep toan cuc theo thoi gian, tinh do lech point-in-time, dung chi muc
san pham->user cho T_ring.

R19: khong giu dict JSON day du (title/text/images) trong RAM - moi dong chi giu
timestamp, user_id, parent_asin, rating, helpful_vote, verified_purchase va 3 truong
dan xuat text_len, has_image, title_hash (du de Pass B tinh dac trung van ban v1).
Du lieu gom theo cot (mang numpy / cot pandas), khong phai list cac dict."""
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dac_trung_thoigian import tinh_lech_pointintime_mang

KICH_THUOC_KHOI = 1_000_000

# Giu du lieu duoi dang bang pyarrow (khong qua pandas) de tiet kiem RAM: ghep khoi
# khong sao chep, sap xep 1 lan, ghi parquet truc tiep. large_string: offset 64-bit,
# tranh tran khi 1 cot chuoi vuot 2GB (Electronics ~41 trieu dong).
KIEU_COT = {
    "timestamp": pa.int64(),
    "user_id": pa.large_string(),
    "parent_asin": pa.large_string(),
    "rating": pa.float64(),
    "helpful_vote": pa.int32(),
    "verified_purchase": pa.bool_(),
    "text_len": pa.int32(),
    "has_image": pa.bool_(),
    "title_hash": pa.int64(),
}
LUOC_DO = pa.schema(list(KIEU_COT.items()))
COT_TOI_THIEU = list(KIEU_COT)
COT_CHI_MUC = ["parent_asin", "user_id", "timestamp"]


def bam_tieu_de(title):
    """Hash on dinh giua cac lan chay (md5, KHONG dung hash() co muoi cua Python)
    cua title.strip().lower() -> int64. Trung title <=> trung hash (xac suat va cham
    64-bit khong dang ke)."""
    h = hashlib.md5(title.strip().lower().encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=True)


def tuple_toi_thieu(r):
    return {
        "timestamp": r["timestamp"],
        "user_id": r["user_id"],
        "parent_asin": r["parent_asin"],
        "rating": r["rating"],
        "helpful_vote": r["helpful_vote"],
        "verified_purchase": bool(r["verified_purchase"]),
        "text_len": len(r["text"]),
        "has_image": len(r["images"]) > 0,
        "title_hash": bam_tieu_de(r["title"]),
    }


def _khoi_thanh_bang(cot):
    return pa.table({c: pa.array(cot[c], type=KIEU_COT[c]) for c in COT_TOI_THIEU},
                    schema=LUOC_DO)


def doc_va_sap_xep_bang(duong_dan_jsonl, kich_thuoc_khoi=KICH_THUOC_KHOI):
    """Bang da sap on dinh theo timestamp (trung timestamp giu thu tu trong file)."""
    return _sap_on_dinh(_doc_bang(duong_dan_jsonl, kich_thuoc_khoi), "timestamp")


def _doc_bang(duong_dan_jsonl, kich_thuoc_khoi=KICH_THUOC_KHOI):
    """Doc streaming tung dong, gom theo cot tung khoi (bang pyarrow), giu thu tu file."""
    khoi = []
    cot = {c: [] for c in COT_TOI_THIEU}

    def xa_khoi():
        if cot["timestamp"]:
            khoi.append(_khoi_thanh_bang(cot))
            for v in cot.values():
                v.clear()

    with open(duong_dan_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = tuple_toi_thieu(json.loads(line))
            for c in COT_TOI_THIEU:
                cot[c].append(t[c])
            if len(cot["timestamp"]) >= kich_thuoc_khoi:
                xa_khoi()
    xa_khoi()

    if not khoi:
        return LUOC_DO.empty_table()
    return pa.concat_tables(khoi)  # khong sao chep - chi noi cac khoi


def _sap_on_dinh(bang, cot_khoa):
    """sort_indices cua Arrow la sap xep on dinh. Sap tung cot va bo cot goc ngay sau
    do: dinh RAM = bang + 1 cot, thay vi 2-3 ban sao ca bang."""
    thu_tu = pc.sort_indices(bang, sort_keys=[(cot_khoa, "ascending")])
    luoc_do = bang.schema
    da_sap = {}
    for ten in luoc_do.names:
        da_sap[ten] = bang.column(ten).take(thu_tu)
        bang = bang.drop_columns([ten])
    return pa.table(da_sap, schema=luoc_do)


def doc_va_sap_xep_theo_thoigian(duong_dan_jsonl, kich_thuoc_khoi=KICH_THUOC_KHOI):
    """Ban tra ve DataFrame (tien kiem tra/xem nhanh); Pass A dung doc_va_sap_xep_bang."""
    return doc_va_sap_xep_bang(duong_dan_jsonl, kich_thuoc_khoi).to_pandas()


def _ma_san_pham(cot_parent_asin):
    """Ma so nguyen cho moi parent_asin (dictionary-encode) - tranh factorize chuoi."""
    return pc.dictionary_encode(cot_parent_asin).combine_chunks().indices.to_numpy()


def dung_bo_cap_phat_he_thong():
    """mimalloc mac dinh cua Arrow giu lai vung nho da giai phong -> dinh RAM Pass A
    tren MI 0.85 GB; bo cap phat he thong tra lai cho OS -> 0.64 GB (~113 byte/dong)."""
    pa.set_memory_pool(pa.system_memory_pool())


KICH_THUOC_GHI = 1_000_000  # so dong moi khuc khi ghi file da sap theo user


def _tinh_lech_theo_thu_tu_file(bang, gmean):
    """Do lech point-in-time can thu tu thoi gian, nhung khong dung bang da sap: chi
    lay 2 mang (ma SP, rating) theo thu tu thoi gian (on dinh -> trung timestamp giu thu
    tu file), tinh xong tra ve thu tu file."""
    thu_tu = pc.sort_indices(bang, sort_keys=[("timestamp", "ascending")]).to_numpy()
    ma = _ma_san_pham(bang["parent_asin"])[thu_tu]
    rating = bang["rating"].to_numpy()[thu_tu]
    lech = tinh_lech_pointintime_mang(ma, rating, gmean)
    del ma, rating
    ra = {}
    for ten in list(lech):
        mang = lech.pop(ten)
        ve_file = np.empty_like(mang)
        ve_file[thu_tu] = mang
        ra[ten] = ve_file
    return ra


def chay_pass_a(duong_dan_jsonl, duong_dan_ra_parquet, duong_dan_index_ra):
    dung_bo_cap_phat_he_thong()
    bang = _doc_bang(duong_dan_jsonl)  # thu tu file, khong sap ca bang
    if bang.num_rows == 0:
        raise ValueError(f"File rong: {duong_dan_jsonl}")

    gmean = float(pc.sum(bang["rating"]).as_py()) / bang.num_rows
    lech = _tinh_lech_theo_thu_tu_file(bang, gmean)
    for ten in list(lech):
        bang = bang.append_column(ten, pa.array(lech.pop(ten)))

    pq.write_table(bang.select(COT_CHI_MUC), duong_dan_index_ra, compression="zstd")

    # Ghi theo user_id roi timestamp (on dinh -> trung ca hai giu thu tu file) de Pass B
    # doc theo lo, xu ly tung user lien tiep. Lay va ghi tung khuc: khong dung ban sao
    # da sap cua ca bang.
    thu_tu = pc.sort_indices(bang, sort_keys=[("user_id", "ascending"),
                                              ("timestamp", "ascending")])
    with pq.ParquetWriter(duong_dan_ra_parquet, bang.schema, compression="zstd") as ghi:
        for dau in range(0, bang.num_rows, KICH_THUOC_GHI):
            ghi.write_table(bang.take(thu_tu[dau:dau + KICH_THUOC_GHI]))

    return {"so_dong": bang.num_rows, "gmean": float(gmean)}
