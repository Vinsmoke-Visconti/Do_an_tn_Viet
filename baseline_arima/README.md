# MÔ HÌNH BASELINE ARIMA THUẦN TÚY (Pure ARIMA)

Thư mục này chứa mã nguồn xây dựng mô hình **ARIMA$(p, d, q)$ thuần túy** dùng làm đường cơ sở (baseline) nghiên cứu trong đề tài dự báo chuỗi thời gian bán lẻ M5 (Walmart).

> [!IMPORTANT]
> Folder này **chỉ** cài đặt ARIMA không có mùa vụ và không có biến ngoại sinh. Xem `../baseline_sarimax/` để so sánh với $\text{SARIMA}(1,1,1)(1,1,1)_7$.

---

## 1. Định nghĩa mô hình ARIMA thuần túy

Phương trình toán học:

$$\phi_p(B)(1-B)^d\, y_t = \theta_q(B)\, \varepsilon_t$$

| Ký hiệu | Ý nghĩa |
|:--------|:--------|
| $\phi_p(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ | Đa thức AR bậc $p$ |
| $(1-B)^d$ | Toán tử sai phân bậc $d$ (khử xu thế) |
| $\theta_q(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ | Đa thức MA bậc $q$ |
| $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ | Nhiễu trắng (White noise) |

### Cấu hình mặc định: ARIMA(7, 1, 1)
- **$p = 7$**: Sử dụng 7 trễ tự hồi quy (AR) thay thế cho thành phần mùa vụ tuần — bắt sự tương quan từ cùng ngày tuần trước.  
- **$d = 1$**: Sai phân bậc 1 để chuỗi trở thành dừng (*stationary*).  
- **$q = 1$**: MA(1) hấp thụ cú sốc ngẫu nhiên ngắn hạn.

---

## 2. Mục đích trong nghiên cứu

```
So sánh hiệu năng giữa các mô hình:

ARIMA(7,1,1)  ←── (folder này) — mức sàn không mùa vụ
     ↓ cải thiện khi thêm thành phần mùa vụ
SARIMA(1,1,1)(1,1,1)_7  ← ../baseline_sarimax/
     ↓ cải thiện khi dùng mô hình phi tuyến + đặc trưng phong phú
LightGBM Tweedie        ← ../new_source/
     ↓ cải thiện khi thêm kiến trúc Attention + Multi-horizon
TFT (Temporal Fusion Transformer) ← ../new_source/
```

---

## 3. Cấu trúc thư mục

```text
baseline_arima/
├── model_arima_pure.py      # Lớp PureARIMAModel (statsmodels wrapper)
├── pipeline_arima_pure.py   # Pipeline nạp dữ liệu, train, dự báo, đánh giá
├── visualize_arima_pure.py  # Module vẽ 3 biểu đồ kết quả
├── run_arima_pure.py        # Entry point
├── README.md
└── outputs/
    ├── figures/
    │   ├── 01_arima_forecast_vs_actual.png
    │   ├── 02_arima_store_metrics.png
    │   └── 03_arima_horizon_error.png
    └── results/
        ├── arima_pure_forecast.csv
        ├── arima_pure_metrics.csv
        └── arima_pure_summary.json
```

---

## 4. Cách chạy

```bash
cd new_Do_An/baseline_arima
python run_arima_pure.py
```

---

## 5. Chỉ số đánh giá

| Metric | Công thức | Ý nghĩa |
|:-------|:----------|:--------|
| **RMSE** | $\sqrt{\frac{1}{H}\sum(y_t - \hat{y}_t)^2}$ | Phạt nặng sai số lớn (đỉnh cao điểm) |
| **MAE** | $\frac{1}{H}\sum\|y_t - \hat{y}_t\|$ | Sai số tuyệt đối trung bình |
| **WMAPE** | $\frac{\sum\|y_t - \hat{y}_t\|}{\sum\|y_t\|} \times 100\%$ | Sai số trọng số — chuẩn M5 cấp Store |
| **sMAPE** | $\frac{1}{H}\sum\frac{2\|y_t-\hat{y}_t\|}{|y_t|+|\hat{y}_t|}\times100\%$ | Sai số phần trăm đối xứng |

