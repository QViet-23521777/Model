"""Cham diem review tieng Viet bang model ViSoBERT da train (nhanh text).

    .venv\\Scripts\\python du_doan.py "San pham tot, giao hang nhanh"
    .venv\\Scripts\\python du_doan.py --csv test.csv --ra ket_qua_text.csv
    .venv\\Scripts\\python du_doan.py --kiem_tra test.csv     # cham lai tap test, so voi results.json

Model multiclass: lop 0 = khong spam, 1-3 = cac loai spam, nen p_spam = 1 - p(lop 0).
ViSoBERT nhan cau tho (--no_segment luc train), khong can VnCoreNLP.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from chung import THU_MUC, MODEL_MAC_DINH, TEN_LOP, bang_ket_qua, kiem_tra  # noqa: F401


def nap_model(duong_dan=MODEL_MAC_DINH):
    thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(duong_dan)
    model = AutoModelForSequenceClassification.from_pretrained(duong_dan).to(thiet_bi).eval()
    return tok, model, thiet_bi


@torch.no_grad()
def du_doan(cau, tok, model, thiet_bi, max_len=128, batch=64):
    """Tra ve ma tran xac suat (n_cau, n_lop)."""
    ket_qua = []
    for i in range(0, len(cau), batch):
        enc = tok([str(c) for c in cau[i:i + batch]], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(thiet_bi)
        ket_qua.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(ket_qua)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cau", nargs="*", help="cac cau review can cham")
    ap.add_argument("--csv", help="file csv co cot 'comment'")
    ap.add_argument("--ra", help="ghi ket qua ra csv")
    ap.add_argument("--kiem_tra", help="csv co nhan (vd test.csv) de cham lai")
    ap.add_argument("--model", default=MODEL_MAC_DINH)
    a = ap.parse_args()

    tok, model, thiet_bi = nap_model(a.model)
    print(f"thiet bi: {thiet_bi}")
    if a.kiem_tra:
        kiem_tra(a.kiem_tra, lambda c: du_doan(c, tok, model, thiet_bi), a.model)
        return
    cau = list(a.cau)
    if a.csv:
        d = pd.read_csv(a.csv)
        d.columns = [c.lower() for c in d.columns]
        cau += d["comment"].astype(str).tolist()
    if not cau:
        ap.error("can it nhat 1 cau hoac --csv")
    df = bang_ket_qua(cau, du_doan(cau, tok, model, thiet_bi))
    if a.ra:
        df.to_csv(a.ra, index=False, encoding="utf-8-sig")
        print(f"da ghi {len(df)} dong -> {a.ra}")
    else:
        pd.set_option("display.max_colwidth", 60)
        print(df[["comment", "ten_lop", "p_spam"]].to_string(index=False))


if __name__ == "__main__":
    main()
