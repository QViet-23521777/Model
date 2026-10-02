# Báo cáo: Kết hợp nhánh text và nhánh hành vi trong phát hiện đánh giá giả

*Số liệu dùng dữ liệu đã làm sạch (khử review trùng) và cấu hình đề xuất. Cập nhật 30/09/2026.*

## Tóm tắt

- Đồ án có hai nhánh độc lập: **nhánh text** (ViSoBERT) đọc nội dung từng review tiếng Việt; **nhánh hành vi** (LightGBM) xem thói quen đăng bài của người viết trên dữ liệu Amazon.
- Hai nhánh **không gộp thành một model** mà **nối tiếp**: text chấm review → tra hành vi của đúng người viết → bối cảnh sản phẩm → bảng quyết định.
- Trên YelpZip (bộ duy nhất có nội dung, lịch sử người viết và nhãn thật trên cùng người), ghép hai nhánh nâng AUC từ **0,753 (chỉ text) lên 0,774**; nhóm bị kết luận "rất nghi" có **25,4 %** review bị Yelp lọc, nhóm "bình thường" chỉ **1,5 %**.
- Sau khi khử 1,09 % review trùng, **T_burst (đăng dồn dập)** là loại hành vi duy nhất vừa đứng vững vừa được nhãn độc lập xác nhận nên được dùng để **ra quyết định**; **khen quá mức** có bằng chứng yếu nên dùng cho mức **theo dõi**; chê quá mức, chưa mua, lặp trên biến thể được **báo cáo cho từng người** kèm mức tin cậy.
- Thư mục `goi_ket_hop` chạy được cả hai nhánh trên một file review.

## 1. Hệ thống phát hiện những gì

| Cấp | Câu hỏi | Nhánh | Phát hiện |
|---|---|---|---|
| Review | Review này viết có bất thường? | Text | Review giả (khen/chê theo mẫu), review chỉ nói về shop/giao hàng, nội dung không liên quan (quảng cáo, rác) |
| Người viết | Tài khoản này đăng bài có bất thường? | Hành vi | 5 chỉ số: **đăng dồn dập** (ra quyết định), **khen quá mức** (theo dõi), chê quá mức, chưa mua mà review, lặp trên biến thể (báo cáo) |
| Sản phẩm | Sản phẩm có bị thao túng điểm? | Sản phẩm | **Nhiều tài khoản mới dồn vào** trong vài ngày; bị bơm / bị dìm điểm (gần như không xảy ra trên Amazon) |

## 2. Nhánh text (ViSoBERT)

Dữ liệu ViSpamReviews (Shopee, tiếng Việt). 3 seed (1, 2, 42), 5 epoch:

| Bộ dữ liệu | Bài toán | Macro-F1 (trung bình ± độ lệch chuẩn) | Accuracy |
|---|---|---|---|
| V1 | binary | 89,92 ± 0,21 | 91,92 |
| V1 | multiclass | 78,74 ± 0,70 | 90,66 |
| V2 | binary | 89,51 ± 0,11 | 91,84 |
| V2 | multiclass | 77,54 ± 0,54 | 91,18 |

Model dùng trong hệ thống: **V2 multiclass, seed 42** (lớp 0 = không spam; 1 = review giả; 2 = chỉ nói về shop; 3 = không liên quan; `p_spam = 1 − p(lớp 0)`). Chạy lại trên máy bằng chính file trọng số: macro-F1 **0,7785** (lúc train 0,7782), trùng 99,97 % dự đoán. Suy ra bài toán binary từ model này đạt macro-F1 0,8991.

F1 từng lớp (V2 multiclass seed 42): không spam 0,95; review giả 0,60; chỉ nói về shop 0,67; không liên quan 0,89 → nhánh text **bắt tốt rác/quảng cáo, yếu với review giả viết khéo**.

Triển khai nhẹ (chấm 3.974 câu trên CPU):

| Cách chạy | Môi trường | Model | Macro-F1 | Thời gian |
|---|---|---|---|---|
| torch | 1.247 MB | 390 MB | 0,7785 | 236 giây |
| ONNX (không torch) | 425 MB | 390 MB | 0,7785 | 194 giây |
| ONNX nén int8 | 425 MB | 98 MB | 0,7758 | 85 giây |

## 3. Nhánh hành vi (dữ liệu đã làm sạch)

**Làm sạch:** 478.019 review Electronics (1,09 %) là bản trùng hoàn toàn (cùng người, sản phẩm, thời điểm, độ dài, tiêu đề). Sau khi bỏ và chạy lại: 43.408.925 review, 1.850.383 người có ≥ 5 review. Model chính: `model_hanh_vi_v2_dedup.pkl`.

Chỉ số tái tạo nhãn proxy (30 % dữ liệu giữ lại) và so với **nhãn lọc độc lập của Yelp**:

| Loại | AUC (Electronics) | PR-AUC | AUC so nhãn Yelp | Trạng thái |
|---|---|---|---|---|
| **T_burst** — đăng dồn dập | **0,925** | 0,756 | **0,71** | ✅ Tin cậy |
| T_unverified — chưa mua | 0,802 | 0,492 | — | Chưa kiểm chứng độc lập |
| T_promoter — khen quá mức | 0,889 | 0,509 | 0,61 | ⚠️ Thận trọng |
| T_demoter — chê quá mức | 0,848 | 0,483 | 0,44 | ⚠️ Không khớp Yelp |
| T_variant_spam — lặp trên biến thể | 0,788 | 0,127 | 0,44 | 🧪 Thử nghiệm |
| T_ring — nhóm cấu kết | 0,670 | 0,002 | 0,54 | 🧪 Thử nghiệm |

Ảnh hưởng của làm sạch (trước → sau): T_burst 0,933 → 0,925 (đứng vững); T_unverified, T_promoter, T_demoter không đổi; **T_variant_spam PR-AUC 0,75 → 0,13 và T_ring 0,14 → 0,002** — phần lớn tín hiệu cũ của hai loại này đến từ review trùng.

Chân dung người bị T_burst gắn cờ (5 % điểm cao nhất, Electronics): đăng **toàn bộ review trong ~1 ngày**, khoảng cách giữa hai review gần 0, ~5 review/ngày; người bình thường rải trên ~6 ngày với khoảng cách trung vị ~170 ngày.

## 4. Kiểm chứng bằng nhãn độc lập — YelpZip

YelpZip (Rayana & Akoglu, KDD 2015): 608.598 review nhà hàng, 260.277 người dùng, 13,2 % review bị bộ lọc Yelp gắn cờ. Pipeline chạy nguyên vẹn, model Electronics **không train lại**. Người "bị lọc" = ≥ 50 % review bị lọc (3,9 % trong 23.831 người có ≥ 5 review).

- **T_burst** (công thức nhãn): nhóm "đáng ngờ" bị lọc 8,9 % so với 2,4 % ở nhóm bình thường (**gấp 2,3 lần** mức ngẫu nhiên), AUC 0,70. Model T_burst đã làm sạch: AUC **0,71**.
- **Model có giá trị ngoài "ít hoạt động":** mốc đơn giản "ít ngày hoạt động" đạt AUC 0,73, nhưng khi so **trong từng nhóm cùng mức hoạt động**, T_burst vẫn đạt **0,65 / 0,69 / 0,70 / 0,74**, mạnh nhất ở nhóm hoạt động nhiều.
- **Người mới là nơi gian lận tập trung:** review của người < 5 review bị lọc **21,5 %**, gấp 7 lần người đủ review (2,9 %).
- **Bùng nổ tài khoản mới được xác nhận:** review của người mới trên sản phẩm đang bùng nổ bị lọc **42,3 %**, gấp đôi người mới ở sản phẩm khác (21,5 %). Tuy vậy phần lớn chênh lệch đến từ việc đó là sản phẩm nhỏ, nhiều người mới (xem mục 8b).

## 5. Nhánh sản phẩm và người dùng mới

Nhánh hành vi chỉ chấm người có ≥ 5 review; phần còn lại rất lớn:

| | Electronics (đã làm sạch) | Musical Instruments | Home_and_Kitchen |
|---|---|---|---|
| Người dùng < 5 review | 89,9 % | 95,2 % | 85,2 % |
| Review do nhóm này viết | 59,2 % | 73,7 % | 49,3 % |

Nhánh sản phẩm tính điểm sản phẩm **có trọng số tin cậy** (review của tài khoản đáng ngờ được tính ít) và cờ **bùng nổ tài khoản mới** (≥ 5 review người mới trong 7 ngày **và** ≥ 50 % review người mới của sản phẩm):

| | Electronics (đã làm sạch) | Musical Instruments | Home_and_Kitchen |
|---|---|---|---|
| Sản phẩm đủ dữ liệu (≥ 20 review) | 245.367 | 23.840 | 430.758 |
| Bị bơm / bị dìm (chênh ≥ 0,5★) | 0 / 0 | 0 / 0 | 0 / 25 |
| Bùng nổ tài khoản mới | 1.010 | 182 | 2.931 |

Gần như không có sản phẩm bị "chiếm" điểm: tài khoản đáng ngờ viết quá ít review để kéo lệch trung bình (25 sản phẩm "bị dìm" của Home_and_Kitchen đều nhỏ, trung vị 24 review). Nguy cơ thật nằm ở người mới, được xử lý bằng cờ bùng nổ. Cờ bùng nổ tự tắt khi file đầu vào phủ dưới 60 ngày: khi đó gần như ai cũng có < 5 review trong file và mọi review đều nằm trong vài tuần, nên điều kiện "dồn vào 7 ngày" không còn phân biệt được chiến dịch với bán hàng bình thường.

> **Đính chính (2026-10-01).** Bản trước ghi 16 / 201, 0 / 2, 33 / 386. Các số đó sai: 10 review "ảo" dùng để co điểm về trung bình chung mang trọng số 1 trong khi review thật mang trọng số < 1, nên sản phẩm điểm thấp bị gắn "bị dìm" dù không ai bị hạ trọng số; ngoài ra lần chạy hàng loạt dùng trọng số `1 − diem_toi_da` (bão hòa) nên người có lịch sử bị tin ít hơn người mới. Đã sửa công thức và dùng cùng trọng số với gói (chỉ hạ top 5 % T_burst). Bảng quyết định không đổi vì nhãn bơm/dìm chỉ ghi vào lý do.

## 6. Cách kết hợp

```
Review ─► [1. Text]      ViSoBERT → p_spam, loại spam
            │
            ▼  (tra theo người viết)
         [2. Hành vi]    5 chỉ số của đúng người viết, xếp hạng so với 1,85 triệu người Electronics
            │            (< 5 review → "chưa đủ dữ liệu")
            ▼
         [3. Sản phẩm]   bùng nổ tài khoản mới?
            │
            ▼
         [Bảng quyết định]
```

| Text (p_spam ≥ 0,5) | Hành vi (top 5 % T_burst) | Kết luận | Xử lý |
|---|---|---|---|
| Spam | Bất thường | **Rất nghi** | Ẩn review + kiểm tra tài khoản |
| Spam | Bình thường | Review xấu | Chỉ ẩn review |
| Sạch | Bất thường | **Nghi viết khéo** | Gửi người kiểm duyệt |
| Sạch | Bình thường | Bình thường | Cho qua |
| Sạch | Bình thường, nhưng **khen quá mức** top 5 % | **Theo dõi** | Theo dõi tài khoản, chưa xử lý review |
| Bất kỳ | Chưa đủ dữ liệu | Đếm 3 tín hiệu: text spam, sản phẩm bùng nổ, chưa mua (≥ 2 → rất nghi; 1 → cần xem; 0 → chưa đủ dữ liệu) | |

Review mà text cho là sạch **vẫn được xét hành vi** — đó là cách bắt review giả viết khéo. Người mới **không bao giờ** bị kết luận "bình thường".

### Vì sao dồn dập ra quyết định, các chỉ số khác chỉ theo dõi / báo cáo

Thử trên YelpZip (người có ≥ 5 review, review text sạch và không dồn dập — mức nền 1,5 % bị lọc):

| Chỉ số phụ lọt top 5 % | Số review | % bị Yelp lọc |
|---|---|---|
| Khen quá mức | 4.461 | **3,9 %** (gấp 2,6) → đủ để **theo dõi** |
| Chê quá mức | 4.736 | 2,1 % (gấp 1,4) → chỉ báo cáo |

Cho khen / chê quá mức **ra quyết định ngang hàng** với dồn dập: bắt nhiều hơn (recall 59 % → 63 %) nhưng nhóm "rất nghi" **kém chính xác hẳn** (24,9 % → 16,7 % bị lọc thật). Ghép **có trọng số** thì khen quá mức có ích (AUC text + hành vi 0,774 → 0,784), chê quá mức không thêm gì.

**Vì sao nối tiếp, không gộp model:** hai bộ dữ liệu không có người dùng chung nên không thể train model gộp; luật nối tiếp dễ giải thích (mỗi kết luận chỉ ra do nhánh nào) và sát triển khai thật (text chấm ngay, điểm hành vi tính theo lô rồi tra).

## 7. Thử quy trình ghép trên dữ liệu thật (YelpZip)

Tầng text dùng model tiếng Anh thay thế (TF-IDF + hồi quy logistic, chia 5 phần theo người dùng) vì ViSoBERT là model tiếng Việt; tầng hành vi dùng model đã làm sạch, chỉ T_burst.

| (người có ≥ 5 review) | AUC |
|---|---|
| Chỉ text | 0,753 |
| Chỉ hành vi (T_burst) | 0,695 |
| **Text + hành vi** | **0,774** |

| Kết luận của hệ thống | Số review | % bị Yelp lọc |
|---|---|---|
| Bình thường | 211.748 | 1,5 % |
| Review xấu | 45.468 | 6,7 % |
| Nghi viết khéo | 9.937 | 7,8 % |
| **Rất nghi** | 3.592 | **25,4 %** |
| Cần xem (người mới) | 138.062 | 32,8 % |
| Chưa đủ dữ liệu (người mới) | 199.791 | 13,6 % |

Bảng quyết định bắt được **59 %** review bị lọc của người đủ review (chỉ text: 49 %) với độ chính xác không đổi. 64 ví dụ thật chia 8 tình huống: `bao_cao/vi_du_ket_hop_yelp.csv` (thư mục gốc dự án).

## 8. Thư mục kết hợp `goi_ket_hop`

Chạy: `..\text\.venv\Scripts\python chay_ket_hop.py <file.csv> <thu_muc_ra>` với các cột `user_id, product_id, rating, thoi_gian, comment` và tuỳ chọn `da_mua`, `diem_tb_sp`, `so_danh_gia_sp` (điểm trung bình và số đánh giá **công khai** của sản phẩm — cần cho khen / chê quá mức). Kết quả trên dữ liệu minh hoạ tự tạo (`vi_du_dau_vao.csv`):

| Người dùng | Tình huống | Kết luận | Nhánh bắt |
|---|---|---|---|
| Khách thật | 6 review cụ thể, rải 2 năm | Bình thường | — |
| Khách thật | 1 review "ib zalo… mã giảm giá" | Chỉ review đó bị đánh dấu | Text |
| Tài khoản dồn dập | 5 review 5★ khen chung chung trong 17 phút, sản phẩm chỉ 3,3★ | Dồn dập 97 %, khen quá mức 100 %: review sạch → nghi viết khéo; review spam → rất nghi | **Hành vi** bắt được thứ text bỏ sót |
| Tài khoản chuyên chê | 5 review 1★, chưa mua, sản phẩm 4,6★ | Chê quá mức 100 %, chưa mua 100 % → **liệt kê trong báo cáo từng người**, không tự động kết luận | Hành vi (báo cáo) |
| 6 tài khoản mới | Cùng 1 sản phẩm trong 2 ngày, chưa mua | Rất nghi | Sản phẩm |

Với file nhỏ, mỗi chỉ số hành vi dùng công thức gán nhãn tương ứng, xếp hạng so với Electronics, vì điểm model cần đặc trưng bối cảnh cả sàn (độ phổ biến sản phẩm…) vốn bị lệch khi ít dữ liệu.

## 8b. Làm rõ nhánh sản phẩm và nâng cấp tín hiệu hành vi

Ba bước bổ sung, đều đo bằng nhãn lọc độc lập của YelpZip (`code/danh_gia_san_pham_yelp.py`, `quy_tac_nang_hang.py`, `tin_hieu_moi.py`, `danh_gia_tin_hieu_moi.py`).

### Bước 1 — Nhánh sản phẩm đóng góp gì

**Bỏ từng tầng khỏi bảng quyết định** (608.598 review):

| Cấu hình | Bắt được review bị lọc (recall) | Precision | "Rất nghi": số / % bị lọc |
|---|---|---|---|
| Đủ 3 tầng | 62,2 % | 25,4 % | 3.592 / 25,4 % |
| Bỏ nhánh sản phẩm | 62,2 % | 25,4 % | 3.507 / 24,9 % |
| Bỏ nhánh hành vi | 61,2 % (người đủ review: 59 % → **49 %**) | 26,3 % | 85 / 48,2 % |
| Bỏ nhánh text | **2,1 %** | 12,5 % | 0 |

- **Text là xương sống**; bỏ text thì hệ thống gần như không bắt được gì.
- **Hành vi bù 10 điểm recall** cho người đủ review và tạo ra mức "rất nghi".
- **Nhánh sản phẩm gần như không đổi kết quả tổng**: chỉ đổi kết luận của 123 review (các review này bị lọc 42 %, cao gấp 1,5–2 lần review cùng mức). Ở cấp sản phẩm, 12 sản phẩm bùng nổ bị lọc 36,2 %, nhưng sản phẩm **cùng cỡ và cùng tỉ lệ người mới** cũng bị lọc 31,3 % → phần lớn "tác động" là do đó là **sản phẩm nhỏ nhiều người mới**, đóng góp riêng của cờ bùng nổ nhỏ.
- Ở cấp sản phẩm, **tỉ lệ người mới** tìm ra nhà hàng bị lọc nhiều (top 10 %) với AUC 0,83; thêm độ tập trung + bùng nổ lên 0,84; độ chênh điểm (sau khi sửa công thức) không thêm gì (AUC riêng 0,46 — bản trước ghi 0,85 là do lỗi công thức).

### Bước 2 — Quy tắc nâng hạng tín hiệu

Áp chung cho mọi chỉ số: **lift** = tỉ lệ bị lọc ở nhóm bị gắn cờ / nhóm còn lại (người có ≥ 5 review), khoảng tin cậy 95 % bootstrap.

| Điều kiện | Vai trò |
|---|---|
| lift ≥ 2, khoảng tin cậy > 1, đứng vững sau khử trùng | **Ra quyết định** |
| lift ≥ 1,5, khoảng tin cậy > 1 | **Theo dõi** |
| Còn lại | **Chỉ báo cáo** |

| Chỉ số (top 5 %) | Lift | Khoảng 95 % | Khử trùng | Vai trò theo quy tắc |
|---|---|---|---|---|
| Đăng dồn dập | **4,70** | 4,03 – 5,44 | giữ | quyết định |
| Khen quá mức | **2,21** | 1,78 – 2,65 | giữ | quyết định * |
| Chê quá mức | 0,82 | 0,56 – 1,09 | giữ | báo cáo |
| Nhóm cấu kết (cũ, so tập sản phẩm) | không đo được (33 % người hoà điểm tối đa) | — | sụp | báo cáo |
| Lặp trên biến thể | 0,44 | 0,27 – 0,66 | sụp | báo cáo |

\* Khen quá mức đạt ngưỡng "quyết định", nhưng khi cho nó quyết định **ngang hàng** với dồn dập thì nhóm "rất nghi" giảm độ chính xác (24,9 % → 19,8 %). Gói hiện để ở mức **theo dõi**; đổi hay không là lựa chọn giữa độ phủ và độ chính xác.

### Bước 3 — Tín hiệu mới, định nghĩa chung cho mọi sàn

Định nghĩa lấy từ nghiên cứu đã công bố, **chỉ thử vài ngưỡng hợp lý**; review dưới 8 từ không xét (câu ngắn kiểu "good" trùng tự nhiên).

| Tín hiệu | Định nghĩa | Người bị gắn cờ | Lift | Khoảng 95 % | Ngoài nhóm dồn dập | Người mới (mức review) | Vai trò |
|---|---|---|---|---|---|---|---|
| **Chép nội dung giữa các tài khoản** | Review có bản gần giống (Jaccard ≥ 0,8, MinHash-LSH) do **tài khoản khác** đăng | 1,1 % | **9,75** | 8,01 – 11,52 | 9,93 | **2,41** | quyết định |
| **Cấu kết theo thời gian** | Cùng một người khác review **≥ 3 sản phẩm chung, mỗi lần cách ≤ 3 ngày** | 2,6 % | **2,41** | 1,80 – 3,06 | 2,24 | 1,70 | quyết định |
| Cấu kết theo thời gian, ≥ 2 sản phẩm | | 17 % | 1,53 | 1,32 – 1,76 | 1,37 | 0,93 | theo dõi |
| Lặp nội dung trong 1 tài khoản (≥ 8 từ) | Bản gần giống (Jaccard ≥ 0,7) của **chính mình** ở sản phẩm khác | 0,3 % | 1,23 | 0,00 – 2,77 | — | 1,17 | báo cáo |

- **Chép nội dung giữa tài khoản là tín hiệu mạnh nhất tìm được**, độc lập với dồn dập (lift ngoài nhóm dồn dập vẫn 9,9) và **giúp được người mới** (2,4 lần) — nhóm nhánh hành vi cũ không chấm được. Lưu ý: bộ lọc Yelp có thể tự dùng dấu hiệu trùng nội dung, nên độ khớp có thể phần nào "cùng cách đo".
- **Cấu kết theo thời gian** thay được T_ring cũ (sụp sau khử trùng). Áp nguyên định nghĩa sang Amazon Musical Instruments (đã khử 33.659 review trùng): gắn cờ 0,26 % người có ≥ 5 review; tỉ lệ ở nhóm dồn dập (0,53 %) gấp đôi nhóm còn lại (0,24 %) — **cùng chiều với Yelp**.
- **Đính chính:** lần chạy đầu, "lặp nội dung trong 1 tài khoản" đạt lift 3,5–4,0, nhưng đó là do **review cực ngắn lặp lại**; trên Amazon MI định nghĩa đó gắn cờ 12,7 % người dùng. Sau khi áp quy tắc ≥ 8 từ, tín hiệu không còn ý nghĩa (lift 1,2–1,4) → chỉ báo cáo.
- Chưa làm: chép nội dung giữa tài khoản trên Amazon (MinHash cho 3 triệu review MI quá nặng với RAM hiện có); kiểm chứng trên YelpNYC / YelpChi để khẳng định "quy chuẩn chung". **Hai tín hiệu mới chưa được đưa vào `chay_ket_hop.py`.**

## 9. Giới hạn

- Text và hành vi **không có người dùng chung** → không có số đánh giá ghép trên dữ liệu tiếng Việt; kiến trúc ghép được kiểm chứng trên YelpZip với model text tiếng Anh thay thế.
- Nhãn nhánh hành vi là **proxy**; kiểm chứng độc lập hợp lệ duy nhất là YelpZip (nhãn lọc Yelp cũng không phải sự thật tuyệt đối; dữ liệu từ bản sao không chính thức). Hai bộ Amazon SNAP (Beauty, Clothing & Accessories) không dùng được: sau khi bỏ review bị chép sang biến thể chỉ còn 3 và 1 người có nhãn.
- Phần lớn độ khớp với Yelp phản ánh **mức độ hoạt động** của người dùng; phần model học thêm có nhưng nhỏ.
- Mọi ngưỡng đặt tay; nhánh hành vi train trên Amazon, **chưa kiểm chứng trên sàn Việt Nam**.
- Musical Instruments và Home_and_Kitchen **chưa được làm sạch** (số liệu T_variant_spam / T_ring ở đó có thể bị thổi phồng).
- Điểm mù T_burst: người chỉ đăng một phiên với khoảng cách đều tuyệt đối xếp hạng thấp hơn thực tế.
- Khen / chê quá mức cần điểm trung bình và số đánh giá của sản phẩm (sản phẩm ≥ 10 đánh giá); thiếu thì hai chỉ số này để trống.
- Nhánh sản phẩm: "người mới" là người có < 5 review trong dữ liệu, không phải tài khoản mới tạo (dữ liệu không có ngày tạo tài khoản); cờ bùng nổ tắt khi file < 60 ngày (ngưỡng đặt tay); Home_and_Kitchen còn 25 sản phẩm "bị dìm" vì vẫn chấm bằng model hành vi bản gốc.
- Không phát hiện được: review giả viết khéo từ tài khoản hoạt động bình thường; phân biệt người được tặng hàng hợp lệ với gian lận.

## 10. Câu hỏi thường gặp khi bảo vệ

**Hai nhánh khác dữ liệu, khác ngôn ngữ, sao ghép được?** — Ghép ở mức quy trình, không gộp model. Mỗi nhánh đánh giá trên dữ liệu riêng; quy trình ghép được kiểm chứng trên YelpZip, bộ có đủ hai loại thông tin trên cùng người.

**Sao chỉ dồn dập được ra quyết định?** — Đó là chỉ số duy nhất qua cả ba phép thử: khớp nhãn Yelp, vẫn đúng trong từng nhóm cùng mức hoạt động, đứng vững sau làm sạch dữ liệu. Các chỉ số khác vẫn được dùng: khen quá mức cho mức theo dõi (nhóm này bị Yelp lọc gấp 2,6 lần), chê quá mức / chưa mua / lặp trên biến thể được báo cáo cho từng người. Cho tất cả ra quyết định ngang hàng làm nhóm "rất nghi" kém chính xác (24,9 % → 16,7 %).

**Người mới thì sao?** — Không bị nhánh hành vi kết luận; dựa vào text, cờ bùng nổ tài khoản mới và việc đã mua chưa. Trên Yelp, người mới ở sản phẩm bùng nổ bị lọc gấp đôi, nhưng sản phẩm cùng cỡ và cùng tỉ lệ người mới cũng bị lọc gần như vậy (31 % so với 36 %), nên đóng góp riêng của cờ bùng nổ nhỏ.

**Nhánh sản phẩm có tác dụng gì?** — Nhãn bị bơm / bị dìm gần như không bật (Electronics 0 / 0): tài khoản đáng ngờ viết quá ít review để kéo lệch điểm sản phẩm quá 0,5★. Đây là kết quả, không phải lỗi; số cũ (16 / 201) sai do công thức và đã được đính chính ở mục 5. Phần dùng được là cờ bùng nổ cho người mới, và nó chỉ đổi kết luận của 123 review trên Yelp.

**Ngưỡng lấy ở đâu?** — Đặt tay, vì không có dữ liệu chung có nhãn để học ngưỡng tối ưu (giới hạn đã nêu).

**Áp dụng cho Việt Nam được không?** — Chưa chứng minh; cần dữ liệu sàn Việt có cả nội dung lẫn lịch sử người dùng (hướng phát triển).

## 11. Hướng phát triển

- Thu thập dữ liệu tiếng Việt có nội dung, lịch sử người dùng và nhãn để đo trực tiếp hiệu quả ghép và học ngưỡng.
- Kiểm chứng thêm trên YelpNYC, YelpChi.
- Làm sạch Musical Instruments, Home_and_Kitchen; xem lại định nghĩa T_promoter / T_demoter, T_ring.
- Học cách ghép (thay luật đặt tay) khi có dữ liệu chung có nhãn; tín hiệu riêng cho người mới.

## Nguồn dữ liệu

- ViSpamReviews V1/V2 — Van Dinh & Luu (2022, 2024).
- Amazon Reviews 2023 — McAuley Lab (Electronics, Musical Instruments, Home_and_Kitchen).
- YelpZip — Rayana & Akoglu, "Collective Opinion Spam Detection: Bridging Review Networks and Metadata", KDD 2015 (bản sao Google Drive).
- Amazon SNAP (Beauty, Clothing & Accessories) — McAuley & Leskovec (2013).
