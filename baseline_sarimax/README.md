# MÔ HÌNH BASELINE ARIMA / SARIMAX - DỰ BÁO CHUỖI THỜI GIAN M5

Thư mục này chứa mã nguồn hoàn chỉnh xây dựng mô hình đối chuẩn (Baseline Model) **ARIMA / SARIMAX** cho bài toán dự báo chuỗi thời gian bán lẻ 28 ngày Walmart M5.

---

## 1. Mục Đích & Vai Trò Trong Đồ Án
Trong nghiên cứu dự báo chuỗi thời gian, một mô hình máy học nâng cao (như LightGBM hay Temporal Fusion Transformer - TFT) cần được đối chuẩn (benchmark) với các mô hình kinh tế lượng cổ điển để chứng minh tính hiệu quả vượt trội.

- **ARIMA (Autoregressive Integrated Moving Average)** đại diện cho phương pháp thống kê truyền thống tuyến tính.
- **SARIMAX (Seasonal ARIMA with eXogenous factors)** bổ sung thêm thành phần chu kỳ tuần $s=7$ và các biến ngoại sinh (lịch nghỉ lễ, trợ cấp SNAP, giá bán).

---

## 2. Cấu Trúc Thư Mục
```text
baseline_arima/
├── model_arima.py         # Lớp mô hình M5ARIMABaseline bọc statsmodels
├── pipeline_arima.py      # Pipeline nạp dữ liệu, huấn luyện 10 Stores, dự báo 28 ngày
├── visualize_arima.py     # Module trực quan hóa kết quả dự báo và biểu đồ metrics
├── run_arima.py           # Script thực thi chính (Entry point)
├── README.md              # Tài liệu kỹ thuật
└── outputs/
    ├── figures/           # Biểu đồ phân tích (PNG chất lượng cao 300 DPI)
    │   ├── 01_arima_forecast_vs_actual.png
    │   ├── 02_arima_store_metrics_comparison.png
    │   └── 03_arima_daily_residuals.png
    └── results/           # Dữ liệu dự báo và bảng số liệu metrics
        ├── arima_forecast_results.csv
        ├── arima_metrics_table.csv
        └── arima_metrics_summary.json
```

---

## 3. Cơ Sở Toán Học
Mô hình $\text{SARIMAX}(p, d, q) \times (P, D, Q)_s$ được thiết lập:
$$\Phi_P(B^s)\phi_p(B)(1 - B)^d(1 - B^s)^D (y_t - \beta X_t) = \Theta_Q(B^s)\theta_q(B)\epsilon_t$$

Trong đó:
- $s = 7$: Chu kỳ mùa vụ tuần (Day of Week).
- $(p, d, q) = (1, 1, 1)$: Bậc tự hồi quy, sai phân cấp 1 và trung bình trượt ngắn hạn.
- $(P, D, Q)_s = (1, 1, 1, 7)$: Bậc mùa vụ tuần để bắt trọn hiện tượng doanh số tăng đột biến vào Thứ Bảy và Chủ Nhật.

---

## 4. Hướng Dẫn Sử Dụng
Để chạy toàn bộ quá trình huấn luyện và dự báo:

```bash
cd new_Do_An/baseline_arima
python run_arima.py
```

---

## 5. Các Chỉ Số Đánh Giá (Evaluation Metrics)
1. **RMSE (Root Mean Squared Error):** Phạt nặng các sai số lớn trong ngày cao điểm.
2. **MAE (Mean Absolute Error):** Đo độ lệch tuyệt đối trung bình.
3. **WMAPE (Weighted Mean Absolute Percentage Error):** Chỉ số chính xác trọng số phù hợp với bài toán chuỗi bán lẻ.
4. **sMAPE (Symmetric Mean Absolute Percentage Error):** Thước đo sai số phần trăm đối xứng.

