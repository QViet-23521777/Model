"""Phan dung chung cua nhanh text, KHONG import torch (dung duoc cho ban ONNX)."""
import json
import os
import time

import numpy as np
import pandas as pd

THU_MUC = os.path.dirname(os.path.abspath(__file__))
MODEL_MAC_DINH = os.path.join(THU_MUC, "VisoBertOutput", "V2", "Multi",
                              "out_ViSpamReviewsV2_multiclass_visobert_seed42", "best")
TEN_LOP = {0: "khong_spam", 1: "spam_1", 2: "spam_2", 3: "spam_3"}


def bang_ket_qua(cau, xs):
    df = pd.DataFrame({"comment": cau, "lop": xs.argmax(1), "p_spam": 1 - xs[:, 0]})
    df["ten_lop"] = df["lop"].map(TEN_LOP)
    for k in range(xs.shape[1]):
        df[f"p_{k}"] = xs[:, k]
    return df


def kiem_tra(duong_dan_csv, ham_du_doan, thu_muc_model):
    from sklearn.metrics import f1_score, accuracy_score
    df = pd.read_csv(duong_dan_csv)
    df.columns = [c.lower() for c in df.columns]
    df = df[["comment", "label", "spam_label"]].dropna()   # giong load_split luc train
    t = time.time()
    xs = ham_du_doan(df["comment"].astype(str).tolist())
    giay = time.time() - t
    y_pred = xs.argmax(1)
    f1 = f1_score(df["spam_label"], y_pred, average="macro")
    print(f"multiclass: macro-F1 {f1:.4f}, acc {accuracy_score(df['spam_label'], y_pred):.4f}, {giay:.0f} giay")
    print(f"binary (suy tu multiclass): macro-F1 {f1_score(df['label'], (y_pred != 0).astype(int), average='macro'):.4f}")
    kq = {"macro_f1": float(f1), "giay": giay, "trung_du_doan": None}
    goc = os.path.dirname(thu_muc_model)
    if os.path.exists(os.path.join(goc, "results.json")):
        cu = json.load(open(os.path.join(goc, "results.json"), encoding="utf-8"))["test"]["macro_f1"]
        print(f"results.json luc train: macro-F1 {cu:.4f}  -> chenh {abs(f1 - cu):.4f}")
    if os.path.exists(os.path.join(goc, "test_preds.npy")):
        p_cu = np.load(os.path.join(goc, "test_preds.npy"))
        if len(p_cu) == len(y_pred):
            kq["trung_du_doan"] = float((p_cu == y_pred).mean())
            print(f"trung du doan voi test_preds.npy: {kq['trung_du_doan'] * 100:.2f}% so dong")
    return kq
