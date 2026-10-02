# -*- coding: utf-8 -*-
"""Luat ket hop co dinh (khong hoc tu du lieu - khong co nhan chung cho 3 nhanh).
Tang 1 text (p_spam cua review) -> tang 2 hanh vi (xep_hang cua user) -> tang 3 quyet dinh,
nguoi moi dung them ngu canh san pham + verified_purchase."""
import math

NGUONG_TEXT = 0.5
NGUONG_XEP_HANG = 0.95      # top 5% diem_toi_da; OR-flag cua 6 model phu 72% user nen khong dung


def trang_thai_hanh_vi(xep_hang):
    # float(): bat ca NaN kieu numpy (np.float32 khong phai float cua Python)
    if xep_hang is None or math.isnan(float(xep_hang)):
        return "chua_du_du_lieu"
    return "bat_thuong" if xep_hang >= NGUONG_XEP_HANG else "binh_thuong"


_BANG = {
    (True, "bat_thuong"): ("rat_nghi", "an review + kiem tra tai khoan (uu tien)"),
    (True, "binh_thuong"): ("review_xau", "chi an review"),
    (False, "bat_thuong"): ("nghi_viet_kheo", "gui nguoi kiem duyet"),
    (False, "binh_thuong"): ("binh_thuong", "cho qua"),
}


def quyet_dinh(p_spam, hanh_vi, sp_bung_no=False, sp_nhan=None, verified=None):
    text_spam = p_spam >= NGUONG_TEXT
    ly_do = [f"text p_spam={p_spam:.2f}", f"hanh vi={hanh_vi}"]
    if sp_nhan in ("bi_bom", "bi_dim"):
        ly_do.append(f"san pham {sp_nhan}")
    if hanh_vi != "chua_du_du_lieu":
        muc, hanh_dong = _BANG[(text_spam, hanh_vi)]
    else:
        tin_hieu = int(text_spam) + int(bool(sp_bung_no)) + int(verified is False)
        if sp_bung_no:
            ly_do.append("san pham dang bung no tai khoan moi")
        if verified is False:
            ly_do.append("chua mua hang")
        if tin_hieu >= 2:
            muc, hanh_dong = "rat_nghi", "an review + gui kiem duyet"
        elif tin_hieu == 1:
            muc, hanh_dong = "can_xem", "gui nguoi kiem duyet"
        else:
            muc, hanh_dong = "chua_du_du_lieu", "cho qua, trong so thap trong diem san pham"
    return {"muc": muc, "hanh_dong": hanh_dong, "ly_do": "; ".join(ly_do)}
