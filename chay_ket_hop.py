# -*- coding: utf-8 -*-
"""Chay GHEP 2 nhanh tren 1 file review: text (ViSoBERT) -> hanh vi cua dung nguoi viet
(5 chi so, model da khu trung) -> ngu canh san pham -> bang quyet dinh.

    ..\\text\\.venv\\Scripts\\python chay_ket_hop.py vi_du_dau_vao.csv ket_qua

Dau vao CSV (UTF-8): user_id, product_id, rating (1-5), thoi_gian ("YYYY-MM-DD HH:MM"),
da_mua (True/False, tuy chon), comment (tieng Viet).
Dau ra: <thu_muc_ra>/ket_qua_tung_review.csv, ket_qua_theo_nguoi_dung.csv."""
import json
import math
import os
import sys
import tempfile

GOC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(GOC, "ma_nguon"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ket_hop_3_tang import NGUONG_XEP_HANG, quyet_dinh, trang_thai_hanh_vi  # noqa: E402
from nhanh_san_pham import tinh_nhanh_san_pham  # noqa: E402
from pass_a_thoigian import chay_pass_a  # noqa: E402
from pass_b_gop_user import chay_pass_b  # noqa: E402

MO_HINH = os.path.join(GOC, "mo_hinh")
MODEL_TEXT = os.path.join(MO_HINH, "text_visobert")
MODEL_HANH_VI = os.path.join(MO_HINH, "hanh_vi_khu_trung.pkl")
THAM_CHIEU = os.path.join(MO_HINH, "tham_chieu_hanh_vi.npz")
NGAY_TOI_THIEU_BUNG_NO = 60

# Chi so hanh vi: neo giong gan_nhan.LOAI_NHAN; vai tro theo ket qua kiem chung tren YelpZip
# (bao_cao/BAO_CAO_KET_HOP.md). T_ring bo: can so tap san pham giua nguoi dung ca san.
CHI_SO = {
    "T_burst": {"neo": ["B_burst_2day_ti_le", "B_burst_7day_ti_le", "B_gap_cv"],
                "ten": "dang don dap", "vai_tro": "quyet_dinh", "tin_cay": "da kiem chung"},
    "T_promoter": {"neo": ["B_pos_dev_rate", "B_promoter_dev"],
                   "ten": "khen qua muc", "vai_tro": "theo_doi", "tin_cay": "bang chung yeu"},
    "T_demoter": {"neo": ["B_neg_dev_rate", "B_demoter_dev"],
                  "ten": "che qua muc", "vai_tro": "bao_cao", "tin_cay": "khong khop nhan Yelp"},
    "T_unverified": {"neo": ["A_unverified_rate"],
                     "ten": "chua mua ma review", "vai_tro": "bao_cao", "tin_cay": "chua kiem chung"},
    "T_variant_spam": {"neo": ["B_burst_prod_lap", "B_variant_rate"],
                       "ten": "lap tren bien the", "vai_tro": "bao_cao", "tin_cay": "thu nghiem"},
}


def xep_hang_tham_chieu(diem, phan_vi):
    """Vi tri (0..1) cua diem trong phan bo tham chieu (1,85 trieu user Electronics da khu
    trung), de 'top 5 %' co nghia ca khi file dau vao chi co vai nguoi."""
    diem = np.asarray(diem, dtype=float)
    # diem giua trai/phai: gia tri trung nhau (vd rat nhieu user co ti le burst = 0) nhan
    # hang trung binh, giong rank(pct=True) khi gan nhan
    ra = (np.searchsorted(phan_vi, diem, side="left")
          + np.searchsorted(phan_vi, diem, side="right")) / (2 * len(phan_vi))
    return np.where(np.isnan(diem), np.nan, np.clip(ra, 0.0, 1.0))


def xep_hang_chi_so(dac_trung, tc, loai):
    """Diem proxy cua 1 loai (cong thuc gan nhan: 0,5 * trung binh + 0,5 * max hang cac neo)
    nhung hang tinh SO VOI Electronics thay vi trong file dau vao, roi dat vi tri cua diem do
    trong phan bo diem proxy Electronics. Model bi cam nhin neo (chong ro ri) va dac trung boi
    canh san (do pho bien SP...) bi lech khi file nho, nen quyet dinh dua tren diem proxy."""
    R = np.column_stack([xep_hang_tham_chieu(dac_trung[c].to_numpy(dtype=float), tc[c])
                         for c in CHI_SO[loai]["neo"]])
    R = np.where(np.isnan(R), 1.0, R)   # rank(na_option="bottom") khi gan nhan
    diem = 0.5 * R.mean(axis=1) + 0.5 * R.max(axis=1)
    return xep_hang_tham_chieu(diem, tc[f"{loai}_diem_proxy"])


def ap_dung_theo_doi(muc, xh_promoter):
    """Review ket luan 'binh_thuong' nhung nguoi viet lot top 5 % khen qua muc -> 'theo_doi'
    (YelpZip: 3,9 % bi loc so voi 1,5 %). Khong doi cac muc khac."""
    if muc == "binh_thuong" and not (xh_promoter is None or math.isnan(xh_promoter)) \
            and xh_promoter >= NGUONG_XEP_HANG:
        return "theo_doi"
    return muc


def _sang_jsonl(df, duong_dan):
    ts = pd.to_datetime(df["thoi_gian"])
    ms = ((ts - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).to_numpy()
    da_mua = df["da_mua"] if "da_mua" in df else pd.Series(True, index=df.index)
    with open(duong_dan, "w", encoding="utf-8") as f:
        for i, r in enumerate(df.itertuples(index=False)):
            f.write(json.dumps({"user_id": str(r.user_id), "parent_asin": str(r.product_id),
                                "rating": float(r.rating), "timestamp": int(ms[i]), "helpful_vote": 0,
                                "verified_purchase": bool(da_mua.iloc[i]), "text": str(r.comment),
                                "title": f"r{i}", "images": []}, ensure_ascii=False) + "\n")


def bo_sung_boi_canh_san(duong_dan_pass_a, df, shrink_v=10.0, min_prod_reviews=10):
    """Do lech so voi dam dong can SP co >= 10 review TRUOC do; file nho gan nhu khong co.
    Neu dau vao co diem_tb_sp + so_danh_gia_sp (cong khai tren san), tinh lai do lech cua tung
    review theo dung cong thuc Pass A (dac_trung_thoigian.tinh_lech_pointintime_mang) voi diem
    trung binh cua san. Tra ve so review 'du moc' cua moi user."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pass_a_thoigian import bam_tieu_de
    t = pq.read_table(duong_dan_pass_a)
    if {"diem_tb_sp", "so_danh_gia_sp"} <= set(df.columns):
        # moi review co title 'r<i>' duy nhat (xem _sang_jsonl) -> hash tro ve dong dau vao
        theo_hash = {bam_tieu_de(f"r{i}"): i for i in range(len(df))}
        i_vao = np.array([theo_hash[h] for h in t.column("title_hash").to_pylist()])
        tb = df["diem_tb_sp"].to_numpy(dtype=float)[i_vao]
        n = df["so_danh_gia_sp"].to_numpy(dtype=float)[i_vao]
        r = t.column("rating").to_numpy()
        g = float(r.mean())
        co = ~np.isnan(tb) & ~np.isnan(n) & (n >= min_prod_reviews)
        pmean = (n * tb + shrink_v * g) / (n + shrink_v)
        d = r - pmean
        dev = np.where(d > 0, d / ((5.0 - pmean) + 0.1), np.where(d < 0, -d / ((pmean - 1.0) + 0.1), 0.0))
        moi = {"dev_pointintime": np.where(co, dev, t.column("dev_pointintime").to_numpy()),
               "signed_pointintime": np.where(co, d, t.column("signed_pointintime").to_numpy()),
               "vi_tri_sp": np.where(co, n, t.column("vi_tri_sp").to_numpy()).astype(np.int64),
               "du_moc": co | t.column("du_moc").to_numpy(zero_copy_only=False)}
        for ten, mang in moi.items():
            t = t.set_column(t.schema.get_field_index(ten), ten, pa.array(mang, type=t.schema.field(ten).type))
        pq.write_table(t, duong_dan_pass_a)
    return pd.Series(t.column("du_moc").to_numpy(zero_copy_only=False)).groupby(
        t.column("user_id").to_pylist()).sum()


def tat_bung_no_file_ngan(sp, thoi_gian):
    """File ngan hon NGAY_TOI_THIEU_BUNG_NO ngay: SP nao co >= 5 review deu 'don vao 7 ngay' (ti le
    tap trung ~100 %) va gan nhu ai cung < 5 review trong file -> co bung no khong phan biet duoc
    chien dich voi ban hang binh thuong, nen tat. Tra ve so ngay file phu."""
    t = pd.to_datetime(thoi_gian)
    so_ngay = (t.max() - t.min()).days
    if so_ngay < NGAY_TOI_THIEU_BUNG_NO:
        sp["bung_no_tk_moi"] = False
        print(f"File chi phu {so_ngay} ngay < {NGAY_TOI_THIEU_BUNG_NO}: tat co bung no tai khoan moi")
    return so_ngay


def cham_hanh_vi(dac_trung, goi, tc):
    """Xep hang 5 chi so (xh_<loai>, 0..1 so voi Electronics) + diem model T_burst de tham khao
    (chi dang tin khi chay tren du lieu ca san)."""
    m = goi["models"]["T_burst"]
    X = dac_trung[m["features"]].replace([np.inf, -np.inf], np.nan)
    ra = pd.DataFrame({"user_id": dac_trung["user_id"],
                       "T_burst_diem_model": m["model"].predict_proba(X)[:, 1]})
    for loai in CHI_SO:
        ra[f"xh_{loai}"] = xep_hang_chi_so(dac_trung, tc, loai)
    ra["T_burst_xep_hang"] = ra["xh_T_burst"]
    return ra


def chi_so_noi_bat(dong):
    """Liet ke chi so lot top 5 %, kem vai tro va muc tin cay - de doc, khong phai de tinh."""
    ra = [f"{c['ten']} {dong[f'xh_{l}']:.0%} ({c['vai_tro']}, {c['tin_cay']})"
          for l, c in CHI_SO.items()
          if pd.notna(dong[f"xh_{l}"]) and dong[f"xh_{l}"] >= NGUONG_XEP_HANG]
    return "; ".join(ra)


def chay(duong_dan_csv, thu_muc_ra):
    os.makedirs(thu_muc_ra, exist_ok=True)
    df = pd.read_csv(duong_dan_csv)
    df["user_id"] = df["user_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)

    # Tang 1 - text: cham tung review
    from du_doan import nap_model, du_doan
    tok, model, thiet_bi = nap_model(MODEL_TEXT)
    xs = du_doan(df["comment"].astype(str).tolist(), tok, model, thiet_bi)
    df["p_spam"] = 1 - xs[:, 0]
    df["ten_lop"] = np.array(["khong_spam", "spam_1_review_gia", "spam_2_chi_noi_shop",
                              "spam_3_khong_lien_quan"])[xs.argmax(1)]

    # Tang 2 - hanh vi cua dung nguoi viet (can >= 5 review trong file)
    cot_hv = ["T_burst_diem_model", "T_burst_xep_hang"] + [f"xh_{l}" for l in CHI_SO]
    with tempfile.TemporaryDirectory() as tam:
        jl, pa_, idx, uf = (os.path.join(tam, f) for f in ("in.jsonl", "a.parquet", "i.parquet", "u.parquet"))
        _sang_jsonl(df, jl)
        chay_pass_a(jl, pa_, idx)
        so_du_moc = bo_sung_boi_canh_san(pa_, df)
        chay_pass_b(pa_, uf)
        dac_trung = pd.read_parquet(uf)
        if len(dac_trung):
            hv = cham_hanh_vi(dac_trung, joblib.load(MODEL_HANH_VI), np.load(THAM_CHIEU))
            # khong co review nao du boi canh SP -> khong do duoc lech so voi dam dong
            khong_bc = hv["user_id"].map(so_du_moc).fillna(0).to_numpy() == 0
            hv.loc[khong_bc, ["xh_T_promoter", "xh_T_demoter"]] = np.nan
        else:
            hv = pd.DataFrame({"user_id": pd.Series(dtype=str), **{c: pd.Series(dtype=float) for c in cot_hv}})
        # Tang 3 - san pham: bung no tai khoan moi + diem co trong so tin cay (chi giam trong so
        # nguoi bi coi la bat thuong, giong nguong cua bang quyet dinh)
        nghi = (hv["T_burst_xep_hang"] >= NGUONG_XEP_HANG).astype(float)
        sp = tinh_nhanh_san_pham(pa_, pd.DataFrame({"user_id": hv["user_id"], "diem_toi_da": nghi}))
        tat_bung_no_file_ngan(sp, df["thoi_gian"])
    df = df.merge(hv, on="user_id", how="left").merge(
        sp[["parent_asin", "bung_no_tk_moi", "nhan"]].rename(
            columns={"parent_asin": "product_id", "bung_no_tk_moi": "sp_bung_no_tk_moi", "nhan": "sp_nhan"}),
        on="product_id", how="left")

    # Bang quyet dinh (T_burst) + muc 'theo_doi' tu chi so phu da co bang chung (T_promoter)
    kq = [quyet_dinh(p, trang_thai_hanh_vi(x), sp_bung_no=bool(b),
                     sp_nhan=n if isinstance(n, str) else None,
                     verified=bool(v) if "da_mua" in df else None)
          for p, x, b, n, v in zip(df["p_spam"], df["T_burst_xep_hang"], df["sp_bung_no_tk_moi"],
                                   df["sp_nhan"], df.get("da_mua", pd.Series(True, index=df.index)))]
    for k in ("muc", "hanh_dong", "ly_do"):
        df[k] = [r[k] for r in kq]
    theo_doi = [ap_dung_theo_doi(m, x) == "theo_doi" for m, x in zip(df["muc"], df["xh_T_promoter"])]
    df.loc[theo_doi, "muc"] = "theo_doi"
    df.loc[theo_doi, "hanh_dong"] = "theo doi tai khoan, chua xu ly review"
    df.loc[theo_doi, "ly_do"] = df.loc[theo_doi, "ly_do"] + "; nguoi viet khen qua muc (top 5 %)"

    theo_user = df.groupby("user_id").agg(
        so_review=("comment", "size"), ti_le_text_spam=("p_spam", lambda s: float((s >= 0.5).mean())),
        **{f"xh_{l}": (f"xh_{l}", "first") for l in CHI_SO},
        T_burst_diem_model=("T_burst_diem_model", "first"),
        muc_cao_nhat=("muc", lambda s: max(s, key=_THU_TU.get))).reset_index()
    theo_user["T_burst_xep_hang"] = theo_user["xh_T_burst"]
    theo_user["chi_so_noi_bat"] = theo_user.apply(chi_so_noi_bat, axis=1)
    df.to_csv(os.path.join(thu_muc_ra, "ket_qua_tung_review.csv"), index=False, encoding="utf-8-sig")
    theo_user.to_csv(os.path.join(thu_muc_ra, "ket_qua_theo_nguoi_dung.csv"), index=False, encoding="utf-8-sig")
    return df, theo_user


_THU_TU = {"binh_thuong": 0, "theo_doi": 1, "chua_du_du_lieu": 1, "review_xau": 2,
           "nghi_viet_kheo": 3, "can_xem": 3, "rat_nghi": 4}


if __name__ == "__main__":
    review, user = chay(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ket_qua")
    pd.set_option("display.width", 250)
    print(user[["user_id", "so_review", "ti_le_text_spam"] + [f"xh_{l}" for l in CHI_SO]
               + ["muc_cao_nhat"]].round(3).to_string(index=False))
    print()
    print(user[["user_id", "chi_so_noi_bat"]].to_string(index=False))
    print()
    print(review[["user_id", "product_id", "comment", "p_spam", "sp_bung_no_tk_moi", "muc"]]
          .to_string(index=False, max_colwidth=45))
