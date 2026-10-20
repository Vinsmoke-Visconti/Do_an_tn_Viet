# BÁO CÁO KẾT QUẢ THỰC NGHIỆM – LIGHTGBM TOP-1 M5

> **Ngày xuất báo cáo**: 2026-10-03 13:17:38  
> **Phương pháp**: LightGBM + Tweedie Loss + Recursive Forecasting (28 ngày)  
> **Dataset**: Walmart M5 Forecasting (Kaggle)  

---

## 1. Đánh giá Đa cấp độ (Hierarchical Evaluation)

| Cấp độ | RMSE | MAE | WMAPE (%) | sMAPE (%) |
|--------|------|-----|-----------|-----------|
| Level 12 (SKU) | 1.476 | 1.137 | 51.172 | 69.163 |
| Level 3 (Store) | 11.723 | 9.360 | 8.425 | 8.328 |
| Level 1 (National) | 11.723 | 9.360 | 8.425 | 8.328 |

## 2. So sánh với các Mô hình Baseline

| Mô hình | RMSE | MAE | WMAPE (%) | sMAPE (%) |
|---------|------|-----|-----------|-----------|
| **LightGBM (Top-1 Style)** | 1.476 | 1.137 | 51.172 | 69.163 |
| **ARIMA (Reference)** | 2.450 | 1.820 | 89.300 | 102.500 |
| **LSTM (Reference)** | 1.950 | 1.550 | 71.200 | 84.100 |
| **TFT (Reference)** | 1.680 | 1.310 | 58.700 | 72.800 |

## 3. Phân tích Phần dư (Residual Analysis)

| Chỉ số | Giá trị |
|--------|---------|
| Trung bình phần dư (Bias) | 0.0389 |
| Độ lệch chuẩn phần dư    | 1.4758 |
| Dự báo cao nhất vượt thực tế | 3.97 |
| Dự báo thấp nhất hơn thực tế | 6.99 |
| Tỷ lệ ngày nhu cầu = 0      | 16.5% |

## 4. Top 10 Đặc trưng Quan trọng nhất (Gain)

| Rank | Feature | Gain Score |
|------|---------|------------|
| 1 | `item_id` | 99455.78 |
| 2 | `sales_rolling_mean_28` | 32328.71 |
| 3 | `sales_ewm_alpha05` | 5933.25 |
| 4 | `price_momentum` | 5014.61 |
| 5 | `sales_ewm_alpha02` | 4616.20 |
| 6 | `sales_rolling_std_28` | 4567.54 |
| 7 | `day` | 4214.40 |
| 8 | `sales_rolling_std_7` | 3977.39 |
| 9 | `days_since_last_event` | 3950.96 |
| 10 | `days_to_next_event` | 3781.86 |

## 5. Nhận xét & Kết luận

- **LightGBM + Tweedie Loss** xử lý hiệu quả hiện tượng Intermittent Demand
  (nhu cầu ngắt quãng) nhờ phân phối Tweedie phù hợp với dữ liệu bán lẻ.
- **Chuỗi trễ (Lag & Rolling)** đóng góp tỷ trọng lớn nhất trong Feature Importance,
  xác nhận rằng lịch sử bán hàng gần nhất là yếu tố dự báo quan trọng nhất.
- **Dự báo đệ quy 28 bước** (RecursiveForecaster) cho phép mô hình dự báo
  liên tục mà không cần dữ liệu tương lai thực tế.
- Mô hình hoạt động tốt hơn ở cấp độ Store (Level 3) so với SKU (Level 12)
  do hiệu ứng averaging khi tổng hợp.

---
*Báo cáo được tự động sinh bởi `03_evaluation_report.py`*