# Gói kết hợp 2 nhánh phát hiện đánh giá giả

Thư mục này ghép **nhánh text** (ViSoBERT, đọc nội dung review tiếng Việt) và **nhánh hành vi** (LightGBM, xem thói quen đăng bài của người viết) thành một quy trình chạy được trên một file review:

```
Review ─► [1. Text]      ViSoBERT chấm nội dung → p_spam, loại spam
            │
            ▼  (tra theo người viết)
         [2. Hành vi]    5 chỉ số của ĐÚNG người viết, xếp hạng so với 1,85 triệu người Electronics
            │            (< 5 review trong file → "chưa đủ dữ liệu")
            ▼
         [3. Sản phẩm]   nhiều tài khoản mới dồn vào 1 sản phẩm trong 7 ngày?
            │
            ▼
         [Bảng quyết định]  → bình thường / theo dõi / review xấu / nghi viết khéo / cần xem / rất nghi / chưa đủ dữ liệu
```

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `chay_ket_hop.py` | Script chạy chính |
| `mo_hinh/text_visobert/` | Model text ViSoBERT (V2 multiclass seed 42, macro-F1 0,7785) — **~375 MB, không có trong git** |
| `mo_hinh/hanh_vi_khu_trung.pkl` | 6 model hành vi train trên Electronics **đã khử trùng** — **không có trong git** |
| `mo_hinh/tham_chieu_hanh_vi.npz` | Phân bố tham chiếu (Electronics đã khử trùng) của các chỉ số, để xếp hạng người dùng |
| `ma_nguon/` | Bản chép các module cần thiết từ `code/` và `text/` |
| `vi_du_dau_vao.csv` | **Dữ liệu tự tạo để minh hoạ** (29 review, 11 người) |
| `ket_qua_vi_du/` | Kết quả chạy trên ví dụ |
| `tests/` | 10 test (`pytest tests`) |
| `bao_cao/BAO_CAO_KET_HOP.md`, `.html` | **Báo cáo tổng hợp** (số liệu sau khi làm sạch dữ liệu) |

## Cách chạy

Dùng môi trường của nhánh text (đã có torch CPU):

```
cd D:\hanh_vi_v2\goi_ket_hop
..\text\.venv\Scripts\python -m pip install -r requirements.txt     # lần đầu
..\text\.venv\Scripts\python chay_ket_hop.py vi_du_dau_vao.csv ket_qua_vi_du
```

**File đầu vào** (CSV, UTF-8), mỗi dòng một review:

| Cột | Bắt buộc | Ý nghĩa |
|---|---|---|
| `user_id` | ✔ | Mã người viết |
| `product_id` | ✔ | Mã sản phẩm |
| `rating` | ✔ | Số sao 1–5 |
| `thoi_gian` | ✔ | `YYYY-MM-DD HH:MM` |
| `comment` | ✔ | Nội dung review tiếng Việt |
| `da_mua` | | `True`/`False` — cần cho chỉ số "chưa mua mà review" |
| `diem_tb_sp` | | Điểm trung bình **công khai** của sản phẩm trên sàn |
| `so_danh_gia_sp` | | Số lượt đánh giá **công khai** của sản phẩm |

`diem_tb_sp` và `so_danh_gia_sp` cần cho chỉ số **khen / chê quá mức** (độ lệch so với đám đông chỉ tính khi sản phẩm có ≥ 10 đánh giá). Thiếu hai cột này, hai chỉ số đó để **trống** thay vì ra số sai.

**Đầu ra:**
- `ket_qua_tung_review.csv`: mỗi review có `p_spam`, `ten_lop` (loại spam), 5 chỉ số hành vi `xh_*` (0–1), `sp_bung_no_tk_moi`, `muc`, `hanh_dong`, `ly_do`.
- `ket_qua_theo_nguoi_dung.csv`: mỗi người có số review, tỉ lệ review bị text coi là spam, **5 chỉ số hành vi**, `chi_so_noi_bat` (các chỉ số lọt top 5 %, kèm vai trò và mức tin cậy), mức nặng nhất.

## 5 chỉ số hành vi và vai trò

Mỗi chỉ số tính theo đúng công thức gán nhãn của đồ án, rồi xếp hạng **so với 1,85 triệu người Electronics** (0,95 = cao hơn 95 % người dùng). Vai trò được đặt theo kết quả kiểm chứng trên YelpZip (nhãn lọc độc lập của Yelp):

| Chỉ số | Đo gì | Vai trò | Bằng chứng |
|---|---|---|---|
| **Đăng dồn dập** (`xh_T_burst`) | Tỉ lệ review đăng sát nhau trong 2 / 7 ngày, nhịp đăng thất thường | **Ra quyết định** (top 5 % = bất thường) | AUC 0,71 so nhãn Yelp; đứng vững sau khử trùng |
| **Khen quá mức** (`xh_T_promoter`) | Hay chấm cao hơn điểm trung bình sản phẩm | **Theo dõi**: review "bình thường" của người top 5 % → mức `theo_doi` | Nhóm này bị Yelp lọc 3,9 % so với 1,5 % (gấp 2,6); thêm vào ghép có trọng số: AUC 0,774 → 0,784 |
| Chê quá mức (`xh_T_demoter`) | Hay chấm thấp hơn điểm trung bình sản phẩm | Chỉ báo cáo | Không khớp Yelp (2,1 % so với 1,5 %) |
| Chưa mua mà review (`xh_T_unverified`) | Tỉ lệ review không xác minh mua hàng | Chỉ báo cáo | Chưa kiểm chứng (Yelp không có dữ liệu "đã mua") |
| Lặp trên biến thể (`xh_T_variant_spam`) | Review lặp lại trên cùng sản phẩm / biến thể trong đợt dồn | Chỉ báo cáo | Thử nghiệm — sụp sau khử trùng |

Vì sao không cho mọi chỉ số cùng ra quyết định: thử trên YelpZip, coi khen/chê quá mức **ngang hàng** với dồn dập làm tăng số review bắt được (59 % → 63 %) nhưng nhóm "rất nghi" **kém chính xác hẳn** (24,9 % → 16,7 % bị lọc thật).

T_ring (nhóm cấu kết) **không có trong gói**: cần so tập sản phẩm giữa người dùng trên cả sàn.

## Kết quả trên dữ liệu ví dụ

| Người dùng | Tình huống | Chỉ số nổi bật | Kết luận |
|---|---|---|---|
| KHACH_THAT_01 | 6 review cụ thể, rải 2 năm | — | bình thường |
| KHACH_THAT_02 | 5 review + 1 review "ib zalo… mã giảm giá" | — | chỉ review quảng cáo bị bắt (**text**) |
| TK_DONG_DAP_01 | 5 review 5★ khen chung chung trong 17 phút, trên sản phẩm chỉ được 3,3★ | dồn dập 97 %, **khen quá mức 100 %** | review sạch → **nghi viết khéo**, review spam → **rất nghi** |
| TK_CHUYEN_CHE_01 | 5 review 1★, chưa mua, trên sản phẩm 4,6★ | **chê quá mức 100 %, chưa mua 100 %** | bình thường về quyết định, nhưng **được liệt kê trong báo cáo** (hai chỉ số chưa được kiểm chứng) |
| TK_MOI_01…06 | 6 tài khoản mới, cùng 1 sản phẩm trong 2 ngày, chưa mua | chưa đủ dữ liệu | **rất nghi** (bùng nổ tài khoản mới + chưa mua) |
| KHACH_MOI_07 | 1 review bình thường | chưa đủ dữ liệu | chưa đủ dữ liệu |

## Cấu hình và lý do

- **Xếp hạng so với Electronics, không so trong file**: file nhỏ chỉ có vài người nên "top 5 %" trong file vô nghĩa.
- **Dùng công thức nhãn (điểm proxy), không dùng điểm model** để xếp hạng: các model được thiết kế không nhìn cột dùng để gán nhãn (chống rò rỉ) và dựa nhiều vào đặc trưng bối cảnh cả sàn (độ phổ biến sản phẩm…). Trên file nhỏ các đặc trưng này lệch hẳn (vd độ phổ biến = 0 so với trung vị 1.724 ở Electronics) và model T_burst cho mọi người ~0,96–0,98. Điểm model T_burst vẫn ghi ở `T_burst_diem_model` để tham khảo khi chạy trên dữ liệu cả sàn.
- Ngưỡng: `p_spam ≥ 0,5` là spam; xếp hạng `≥ 0,95` là nổi bật; bùng nổ = ≥ 5 review người mới trong 7 ngày **và** ≥ 50 % review người mới của sản phẩm; cờ bùng nổ tự tắt nếu file phủ dưới 60 ngày (file ngắn thì sản phẩm nào có ≥ 5 review cũng "dồn vào 7 ngày").

## Giới hạn

- **Không có số đánh giá cho phần ghép trên dữ liệu tiếng Việt**: chưa có bộ dữ liệu Việt có cả nội dung, mã người dùng và nhãn. Kiến trúc ghép được kiểm chứng trên YelpZip với model text tiếng Anh thay thế.
- Nhánh hành vi train trên Amazon (tiếng Anh, Mỹ), **chưa kiểm chứng trên sàn Việt Nam**.
- Người có **< 5 review trong file** không được chấm hành vi.
- Khen / chê quá mức cần `diem_tb_sp`, `so_danh_gia_sp`; chưa mua cần `da_mua`.
- Điểm mù đã biết của chỉ số dồn dập: người **chỉ đăng một phiên duy nhất với khoảng cách đều tuyệt đối** (vd đúng 5 phút một bài) xếp hạng thấp hơn thực tế (~0,82).
- `vi_du_dau_vao.csv` là **dữ liệu tự tạo** để minh hoạ, không phải dữ liệu thật.
