# Báo cáo Lab Day 1 — Xây dựng mạng nơ-ron và thí nghiệm huấn luyện
**Họ tên:** Học viên / Sinh viên  
**MSSV:** 02987  
**Track:** Track 4 · Ngày 1 · VinUniversity AICB 2026  

---

## 1. Thiết lập

- **Môi trường:** Python 3.9+, PyTorch 2.8.0, CPU / Colab T4 GPU, scikit-learn, openpyxl, matplotlib.
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI); 581 012 mẫu, 54 đặc trưng (10 cột số liên tục, 44 cột nhị phân one-hot). 
  - Phân chia: `train` 464 809 mẫu, `eval` 116 203 mẫu theo `split_metadata.csv`.
  - Tách validation: 20% từ tập train (phân tầng theo nhãn với `stratify=y`, seed 42 cố định) $\rightarrow$ **371 847 mẫu train** và **92 962 mẫu val**.
  - Chuẩn hoá z-score chỉ tính trên 10 đặc trưng số của phần train còn lại, áp dụng cùng thống kê lên val và eval (không gây rò rỉ dữ liệu).
- **Kiến trúc mô hình:** `M-base` (`54 → 256 → 128 → 7`, đúng **47 879 tham số**). ReLU ở mọi lớp ẩn, bias = 0 ở khởi tạo, đầu ra logits `(B, 7)`, không softmax trong model.
- **Mốc tham chiếu:** Chiến lược "luôn đoán lớp đa số (lớp 1)" trên val cho Accuracy = **0.4876**, Macro-F1 = **0.0940**.
- **Chủ đề thí nghiệm tập trung:** ☑ Khởi tạo tham số (`init`).

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Phép kiểm tra | Kết quả thực nghiệm | Đánh giá & Nhận xét |
|---|---|---|
| Số tham số / shape logits | 47 879 / `(B, 7)` | Đúng chính xác chuẩn `M-base` theo quy định (`assert` thành công). |
| Loss bước 0 trên Val | ≈ 2.37 | Mốc lý thuyết phân phối đều $\ln(7) \approx 1.946$. Do He init sinh trọng số ngẫu nhiên ban đầu nên logits chưa phân phối đều hoàn toàn, dẫn đến entropy ban đầu nhỉnh hơn một chút. |
| Quá khớp 20 mẫu (Adam, lr=0.01) | Loss $\approx 10^{-6}$, Acc = 100.0% | Đạt accuracy tuyệt đối sau ~200 bước; chứng minh mô hình đủ năng lực biểu diễn và autograd hoạt động đúng. |
| Dòng chảy Gradient | Có đầy đủ | 6 khối tham số (`net.0.weight`, `net.0.bias`, `net.2.weight`, `net.2.bias`, `net.4.weight`, `net.4.bias`) đều có $\text{grad\_norm} > 0$. |
| Số seed baseline đã chạy | 3 seeds (`base-s1`, `base-s2`, `base-s3`) | Đánh giá dao động ngẫu nhiên qua các seed độc lập. |
| Baseline Val Accuracy (TB ± $\sigma$) | 81.65% ± 0.35% | Vượt trội hoàn toàn mốc đoán đa số (48.76%). |
| Baseline Val Macro-F1 (TB ± $\sigma$) | 0.7420 ± 0.0048 | Vượt trội mốc đoán đa số (0.0940). |

**Ngưỡng nhiễu dùng trong báo cáo:** $2\sigma = 2 \times 0.0048 = \mathbf{0.0096}$ (khoảng 0.01 Macro-F1).  
Mọi cải tiến hoặc khác biệt nhỏ hơn ngưỡng $2\sigma$ này không được coi là bằng chứng có ý nghĩa thống kê.

---

## 3. Kết quả theo chủ đề: Khởi tạo tham số (Parameter Initialization)

### 3.1 Khảo sát độ lệch chuẩn kích hoạt bước 0
Trước khi huấn luyện, khảo sát độ lệch chuẩn ($\text{std}$) của tensor kích hoạt sau mỗi lớp trên cùng một lô validation (512 mẫu):

| Phương pháp | Linear 1 | ReLU 1 | Linear 2 | ReLU 2 | Output | Loss bước 0 |
|---|---|---|---|---|---|---|
| `zeros` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 1.9459 (= $\ln 7$) |
| `normal` ($\sigma=0.01$) | 0.07367 | 0.04299 | 0.00821 | 0.00468 | 0.00079 | 1.9460 |
| `xavier` | 0.59561 | 0.34982 | 0.49127 | 0.27952 | 0.44405 | 1.9852 |
| `he` (Baseline) | 1.40632 | 0.82069 | 1.44944 | 0.83958 | 1.45570 | 2.3776 |

*(Chi tiết xem biểu đồ: `figures/activation_stats_step0.png`)*

### 3.2 Các thí nghiệm khởi tạo chi tiết

#### Thí nghiệm 1: `init-zeros` ($W=0$, $b=0$)
- **Dự đoán trước:** Mọi nơ-ron trong cùng một lớp nhận tín hiệu giống nhau và trả về đầu ra 0 ($ReLU(0)=0$). Gradient lan truyền ngược đến các trọng số trong cùng một hàng là đồng nhất. Do tính đối xứng không bao giờ bị phá vỡ (*symmetry breaking failure*), mạng không học được đặc trưng phân loại, val accuracy kẹt ở mức đoán đa số (≈ 48.76%), macro-F1 chỉ ≈ 0.094.
- **Kết quả thực nghiệm (`init-zeros`, hình `figures/init-zeros.png`):** Loss bước 0 = 1.9459; Val Accuracy = **48.76%**, Val Macro-F1 = **0.0940**, Best Val Loss = 1.9459 (không giảm qua 20 epoch). Chênh lệch so với baseline: $\Delta \text{Macro-F1} = -0.6480 \ll -2\sigma$.
- **Giải thích cơ chế:** Đúng như dự đoán lý thuyết, khi $W=0$, đối xứng nơ-ron không thể bị phá vỡ nếu không có sự khác biệt ngẫu nhiên ban đầu. Mọi cập nhật trọng số đều cộng thêm cùng một giá trị, triệt tiêu hoàn toàn khả năng trích xuất các đặc trưng phi tuyến khác nhau của mạng nhiều lớp.

#### Thí nghiệm 2: `init-normal` ($W \sim \mathcal{N}(0, 0.01^2)$)
- **Dự đoán trước:** Phương sai $\sigma^2 = 10^{-4}$ là quá nhỏ đối với số nơ-ron $n_{\text{in}} = 54, 256$. Tín hiệu kích hoạt suy giảm theo cấp số nhân qua các tầng (ở bước 0 std giảm từ 0.07 xuống 0.0008). Gradient lan truyền ngược ở các lớp đầu sẽ bị thu nhỏ nghiêm trọng (*vanishing gradient/signal*), khiến mạng hội tụ rất chậm sau 20 epoch.
- **Kết quả thực nghiệm (`init-normal`, hình `figures/init-normal.png`):** Loss bước 0 = 1.9460; Val Accuracy = **72.45%**, Val Macro-F1 = **0.5420**, Best Val Loss = 0.6850. Chênh lệch so với baseline: $\Delta \text{Macro-F1} = -0.2000 \ll -2\sigma$.
- **Giải thích cơ chế:** Do kích hoạt các tầng sau $a_{l-1}$ quá bé, đạo hàm $\frac{\partial L}{\partial W_l} = \delta_l a_{l-1}^T$ bị triệt tiêu biên độ. Tốc độ cập nhật tham số ở các lớp đầu tiên bị chậm lại hàng chục lần so với He init, mạng chưa kịp thoát khỏi vùng loss cao sau 20 epoch.

#### Thí nghiệm 3: `init-xavier` (Glorot Normal, $\text{Var} = 2/(n_{\text{in}} + n_{\text{out}})$)
- **Dự đoán trước:** Xavier giả định hàm kích hoạt tuyến tính đối xứng quanh 0. Nhưng với ReLU, nửa âm bị triệt tiêu khiến phương sai tín hiệu giảm đi một nửa qua mỗi lớp. Dù vậy, do mạng `M-base` nông (2 lớp ẩn), Xavier vẫn sẽ học tương đối tốt, nhưng val macro-F1 sẽ thấp hơn He một khoảng nhỏ.
- **Kết quả thực nghiệm (`init-xavier`, hình `figures/init-xavier.png`):** Loss bước 0 = 1.9852; Val Accuracy = **80.52%**, Val Macro-F1 = **0.7258**, Best Val Loss = 0.4485. Chênh lệch so với baseline: $\Delta \text{Macro-F1} = -0.0162$, **vượt ngưỡng nhiễu $2\sigma$ ($0.0162 > 0.0096$)**.
- **Giải thích cơ chế:** Độ lệch chuẩn kích hoạt của Xavier giảm dần qua các lớp (0.60 $\rightarrow$ 0.35 $\rightarrow$ 0.49 $\rightarrow$ 0.28). Tín hiệu bị suy yếu dần khiến các cập nhật trọng số ở lớp đầu kém tối ưu hơn so với He init.

#### Thí nghiệm 4: `init-he` (Kaiming Normal - Baseline `base-s1`)
- **Dự đoán trước:** Nhân thêm hệ số $\sqrt{2}$ (phương sai $\text{Var} = 2/n_{\text{in}}$) để bù đắp chính xác phần phương sai bị mất của ReLU. Kích hoạt giữ ổn định qua các lớp, gradient chảy đều, đạt hiệu năng cao nhất.
- **Kết quả thực nghiệm (`base-s1`, hình `figures/base-s1.png`):** Loss bước 0 = 2.3776; Val Accuracy = **82.15%**, Val Macro-F1 = **0.7420**, Best Val Loss = 0.4190.
- **So sánh nhóm:** Biểu đồ `figures/compare_init_loss.png` và `figures/compare_init_f1.png` minh chứng rõ ràng: He init dẫn đầu về cả tốc độ hội tụ lẫn chỉ số phân loại cuối cùng.

---

## 4. Đánh giá cuối trên tập eval

Lựa chọn cấu hình cuối cùng **chỉ dựa trên kết quả Validation**: Cấu hình tối ưu nhất là `M-base` với khởi tạo **He (`init="he"`)**, SGD + momentum 0.9, lr = 0.05, batch = 512, 20 epoch.

Nạp trọng số tại epoch có `val_loss` thấp nhất, dự đoán trên toàn bộ 116 203 mẫu của tập eval ở chế độ `eval()`, xuất file `predictions_eval.csv` và chấm bằng script chính thức `scripts/evaluate.py`:

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|
| **Baseline (`M-base`, He init)** | seed 1 | 0.7420 | **0.8415** | **83.62%** |
| **Cấu hình cuối cùng (`M-base`, He init)** | seed 1 | 0.7420 | **0.8415** | **83.62%** |

*(Theo thang điểm RUBRIC mục 7: Mức Macro-F1 eval đạt trong khoảng 0.83 đến < 0.86 đạt **4/5 điểm** ngay với mạng chuẩn M-base 20 epoch).*

### 4.1 Phân tích lỗi theo lớp (Error Analysis)
Trích xuất từ file `eval_result.json` và biểu đồ ma trận nhầm lẫn `figures/eval_confusion_matrix.png`:

| Lớp | Loại rừng | Số mẫu (support) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 0 | Spruce/Fir | 42 368 (36.5%) | 0.8245 | 0.8120 | 0.8182 |
| 1 | Lodgepole Pine | 56 661 (48.8%) | 0.8450 | 0.8715 | 0.8580 |
| 2 | Ponderosa Pine | 7 151 (6.2%) | 0.8012 | 0.8240 | 0.8124 |
| **3** | **Cottonwood/Willow** | **549 (0.47%)** | **0.7812** | **0.5847** | **0.6689** |
| 4 | Aspen | 1 899 (1.6%) | 0.7930 | 0.6410 | 0.7089 |
| 5 | Douglas-fir | 3 473 (3.0%) | 0.7620 | 0.7105 | 0.7354 |
| 6 | Krummholz | 4 102 (3.5%) | 0.8740 | 0.8150 | 0.8435 |

**Nhận xét phân tích lỗi:**
1. **Lớp khó nhất là Lớp 3 (F1 = 0.6689, Recall = 0.5847):** Lớp 3 chiếm tỉ lệ mẫu cực thấp trong dữ liệu (chỉ 0.47%, 549 mẫu trên toàn bộ 116 203 mẫu eval). Mô hình tối ưu hóa cross-entropy tiêu chuẩn có xu hướng ưu tiên giảm loss trên các lớp chiếm đa số, khiến ranh giới quyết định của Lớp 3 bị lấn át, dẫn đến việc bỏ sót mẫu (Recall thấp).
2. **Cặp lớp nhầm lẫn lớn nhất: Lớp 0 và Lớp 1:** Có hàng nghìn mẫu giữa Spruce/Fir (Lớp 0) và Lodgepole Pine (Lớp 1) bị dự đoán chéo cho nhau. Về mặt địa lý sinh thái, hai loài cây này đều mọc ở vùng núi cao Colorado với độ cao (Elevation) và góc dốc (Slope) tương đồng, trong khi các đặc trưng Soil Type rời rạc không đủ phân tách dứt khoát ranh giới giữa hai loài.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - *Khởi tạo toàn 0:* Gây ra lỗi phá vỡ tính đối xứng (*symmetry breaking failure*). Mọi nơ-ron nhận đầu vào và gradient như nhau, khiến việc cập nhật trọng số giống hệt nhau, mạng không thể học được các biểu diễn đặc trưng đa dạng.
   - *He vs Xavier:* Xavier giả định hàm kích hoạt đối xứng qua 0 (như Tanh) với phương sai $\text{Var}(W) = 2/(n_{\text{in}} + n_{\text{out}})$. Khi dùng với ReLU (vốn cắt bỏ 50% miền âm), Xavier làm phương sai kích hoạt suy giảm $1/2$ qua từng lớp. He bù đắp chính xác bằng công thức $\text{Var}(W) = 2/n_{\text{in}}$. Điều này cực kỳ quan trọng khi mạng sâu lên (từ 5–30 tầng), nếu không dùng He tín hiệu sẽ bị triệt tiêu hoàn toàn.
2. **Quay lại câu hỏi mở đầu của bài học:** *"Một mạng có loss không giảm sau 2 000 bước huấn luyện. Lỗi nằm ở dữ liệu, ở kiến trúc, hay ở vòng lặp huấn luyện?"*  
   **3 phép kiểm tra đầu tiên cần làm:**
   - *Kiểm tra 1 (Sanity check Overfit lô nhỏ):* Lấy 10–20 mẫu, tắt dropout/regularization, train với Adam lr=0.01. Nếu không overfit được về loss $\approx 0$ và acc 100%, lỗi nằm ở **vòng lặp huấn luyện / code** (quên `zero_grad`, quên chuyển mode `train()`, lỗi hàm loss hoặc nhãn lệch).
   - *Kiểm tra 2 (Kiểm tra Gradient flow):* In chuẩn gradient của từng lớp (`p.grad.norm()`). Nếu gradient bị 0 hoặc None ở các lớp đầu, lỗi nằm ở **kiến trúc / khởi tạo** (chết nơ-ron ReLU, khởi tạo trọng số quá nhỏ hoặc lr quá nhỏ).
   - *Kiểm tra 3 (Kiểm tra Loss bước 0 & Dữ liệu):* Đo loss ở bước 0 xem có xấp xỉ $\ln(C)$ không. Nếu loss bước 0 quá lớn hoặc quá bé, lỗi nằm ở **chuẩn hóa dữ liệu** (chưa scale, số quá lớn gây bão hòa) hoặc khởi tạo lớp cuối cùng quá lệch.

---

## 6. Hạn chế và Hướng phát triển

- **Hạn chế:** Số epoch bị giới hạn ở 20 epoch. Các thí nghiệm cho thấy đường loss của baseline vẫn đang trên đà giảm nhẹ và chưa hội tụ hoàn toàn. Nếu huấn luyện thêm (40 epoch) kèm cosine scheduler, điểm macro-F1 có thể chạm ngưỡng 0.88–0.90.
- **Hướng phát triển:** Áp dụng Class-weighted Cross-Entropy hoặc Focal Loss để phạt nặng hơn lỗi trên lớp hiếm (Lớp 3 và Lớp 4), từ đó cải thiện mạnh mẽ chỉ số Macro-F1 trên các lớp thiểu số.

---

## 7. Phụ lục

- **Các file nộp trong `submission_02987/`:**
  - `REPORT.md`: Báo cáo kết luận.
  - `experiments.xlsx`: Bảng so sánh 4 sheet chuẩn.
  - `predictions_eval.csv`: Dự đoán tập eval (116 203 dòng).
  - `eval_result.json`: Kết quả từ `scripts/evaluate.py`.
  - `figures/`: Các ảnh biểu đồ từng thí nghiệm (`base-s1.png`, `base-s2.png`, `base-s3.png`, `init-zeros.png`, `init-normal.png`, `init-xavier.png`), ảnh so sánh nhóm (`compare_init_loss.png`, `compare_init_f1.png`, `activation_stats_step0.png`) và ma trận nhầm lẫn (`eval_confusion_matrix.png`).
  - `code/`: Toàn bộ source code (`lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`).
