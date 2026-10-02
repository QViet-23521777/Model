import os
import sys

import numpy as np
import pandas as pd
import pytest

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GOC)
from chay_ket_hop import chay, tat_bung_no_file_ngan, xep_hang_tham_chieu  # noqa: E402


def test_xep_hang_tham_chieu():
    q = np.linspace(0, 1, 1001)                      # tham chieu phan bo deu
    assert np.allclose(xep_hang_tham_chieu(np.array([0.0, 0.5, 0.96, 1.0]), q), [0.0, 0.5, 0.96, 1.0], atol=0.002)
    assert np.isnan(xep_hang_tham_chieu(np.array([np.nan]), q)[0])


@pytest.fixture(scope="module")
def ket_qua(tmp_path_factory):
    ra = tmp_path_factory.mktemp("ra")
    return chay(os.path.join(GOC, "vi_du_dau_vao.csv"), str(ra))


def test_moi_review_co_ket_luan(ket_qua):
    review, _ = ket_qua
    dau_vao = pd.read_csv(os.path.join(GOC, "vi_du_dau_vao.csv"))
    assert len(review) == len(dau_vao)
    assert {"p_spam", "ten_lop", "T_burst_xep_hang", "sp_bung_no_tk_moi", "muc", "hanh_dong", "ly_do"} <= set(review.columns)


def test_nguoi_moi_khong_bao_gio_binh_thuong(ket_qua):
    review, _ = ket_qua
    moi = review[review["user_id"].str.startswith(("TK_MOI", "KHACH_MOI"))]
    assert moi["T_burst_xep_hang"].isna().all()
    assert (moi["muc"] != "binh_thuong").all()


def test_san_pham_bi_don_tai_khoan_moi(ket_qua):
    review, _ = ket_qua
    assert review.loc[review["product_id"] == "SP_99", "sp_bung_no_tk_moi"].all()
    assert (review.loc[review["product_id"] == "SP_99", "muc"] == "rat_nghi").all()


def test_review_quang_cao_bi_text_bat(ket_qua):
    review, _ = ket_qua
    qc = review[review["comment"].str.contains("zalo")]
    assert (qc["p_spam"] >= 0.5).all()


def test_hanh_vi_phan_biet_dong_dap_voi_khach_that(ket_qua):
    _, user = ket_qua
    xh = user.set_index("user_id")["T_burst_xep_hang"]
    assert xh["TK_DONG_DAP_01"] >= 0.95
    assert xh["KHACH_THAT_01"] < 0.95 and xh["KHACH_THAT_02"] < 0.95


def test_bao_cao_du_5_chi_so_hanh_vi(ket_qua):
    _, user = ket_qua
    for loai in ("T_burst", "T_promoter", "T_demoter", "T_unverified", "T_variant_spam"):
        assert f"xh_{loai}" in user.columns
    assert "chi_so_noi_bat" in user.columns
    moi = user[user["user_id"].str.startswith("TK_MOI")]
    assert moi["xh_T_promoter"].isna().all()


def test_muc_theo_doi_chi_nang_review_binh_thuong():
    from chay_ket_hop import ap_dung_theo_doi
    assert ap_dung_theo_doi("binh_thuong", 0.97) == "theo_doi"
    assert ap_dung_theo_doi("binh_thuong", 0.50) == "binh_thuong"
    assert ap_dung_theo_doi("binh_thuong", float("nan")) == "binh_thuong"
    assert ap_dung_theo_doi("review_xau", 0.99) == "review_xau"


def test_co_boi_canh_san_thi_bat_duoc_khen_che_chua_mua(ket_qua):
    _, user = ket_qua
    u = user.set_index("user_id")
    assert u.loc["TK_DONG_DAP_01", "xh_T_promoter"] >= 0.95      # 5 sao tren SP trung binh 3,3
    assert u.loc["TK_CHUYEN_CHE_01", "xh_T_demoter"] >= 0.95     # 1 sao tren SP trung binh 4,6
    assert u.loc["TK_CHUYEN_CHE_01", "xh_T_unverified"] >= 0.95  # khong mua don nao
    assert u.loc["KHACH_THAT_01", "xh_T_promoter"] < 0.95 and u.loc["KHACH_THAT_01", "xh_T_demoter"] < 0.95


def test_thieu_boi_canh_san_thi_khen_che_de_trong(tmp_path):
    d = pd.read_csv(os.path.join(GOC, "vi_du_dau_vao.csv")).drop(columns=["diem_tb_sp", "so_danh_gia_sp"])
    p = tmp_path / "vao.csv"
    d.to_csv(p, index=False, encoding="utf-8-sig")
    _, user = chay(str(p), str(tmp_path / "ra"))
    du = user[user["so_review"] >= 5]
    assert du["xh_T_promoter"].isna().all() and du["xh_T_demoter"].isna().all()
    assert du["xh_T_burst"].notna().all()


def test_file_ngan_tat_co_bung_no():
    # 5 khach that rai 6 ngay trong file chi dai 1 tuan: dieu kien 'don 7 ngay' luon dung -> tat
    sp = pd.DataFrame({"parent_asin": ["P"], "bung_no_tk_moi": [True]})
    tg = ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-03 10:00", "2024-01-05 10:00", "2024-01-07 10:00"]
    assert tat_bung_no_file_ngan(sp, pd.Series(tg)) == 6
    assert not sp["bung_no_tk_moi"].any()
    # file dai 2 nam (nhu vi_du_dau_vao.csv) -> giu nguyen co
    sp2 = pd.DataFrame({"parent_asin": ["P"], "bung_no_tk_moi": [True]})
    tat_bung_no_file_ngan(sp2, pd.Series(["2022-03-02 19:30", "2024-09-18 22:40"]))
    assert sp2["bung_no_tk_moi"].all()
