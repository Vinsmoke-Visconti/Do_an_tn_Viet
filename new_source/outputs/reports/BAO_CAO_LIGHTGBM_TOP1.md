# BÁO CÁO KẾT QUẢ THỰC NGHIỆM – LIGHTGBM TOP-1 M5

> **Ngày xuất báo cáo**: 2026-10-23 12:41:03  
> **Phương pháp**: LightGBM + Tweedie Loss + Recursive Forecasting (28 ngày)  
> **Dataset**: Walmart M5 Forecasting (Kaggle)  

---

## 1. Đánh giá Đa cấp độ (Hierarchical Evaluation)

| Cấp độ | RMSE | MAE | WMAPE (%) | sMAPE (%) |
|--------|------|-----|-----------|-----------|
| Level 12 (SKU) | 2.116 | 1.016 | 73.275 | 141.487 |
| Level 9 (Cat x State) | 669.078 | 404.993 | 8.623 | 8.200 |
| Level 3 (Store) | 494.974 | 372.583 | 8.814 | 8.671 |
| Level 2 (State) | 1432.892 | 1083.043 | 7.686 | 7.522 |
| Level 1 (National) | 3791.250 | 3133.227 | 7.412 | 7.213 |

## 2. So sánh với các Mô hình Baseline

| Mô hình | RMSE | MAE | WMAPE (%) | sMAPE (%) |
|---------|------|-----|-----------|-----------|
| **LightGBM (Top-1 Style)** | 2.116 | 1.016 | 73.275 | 141.487 |
| **ARIMA (Reference)** | 2.450 | 1.820 | 89.300 | 102.500 |
| **LSTM (Reference)** | 1.950 | 1.550 | 71.200 | 84.100 |
| **TFT (Reference)** | 1.680 | 1.310 | 58.700 | 72.800 |

## 3. Phân tích Phần dư (Residual Analysis)

| Chỉ số | Giá trị |
|--------|---------|
| Trung bình phần dư (Bias) | 0.0842 |
| Độ lệch chuẩn phần dư    | 2.1139 |
| Dự báo cao nhất vượt thực tế | 55.42 |
| Dự báo thấp nhất hơn thực tế | 122.82 |
| Tỷ lệ ngày nhu cầu = 0      | 56.3% |

## 4. Top 10 Đặc trưng Quan trọng nhất (Gain)

| Rank | Feature | Gain Score |
|------|---------|------------|
| 1 | `sales_rolling_mean_28` | 88374334.41 |
| 2 | `sales_ewm_alpha02` | 44216414.87 |
| 3 | `item_id` | 42402204.44 |
| 4 | `sales_rolling_std_28` | 4663669.54 |
| 5 | `sales_rolling_mean_7` | 3331468.99 |
| 6 | `dept_id` | 2163461.73 |
| 7 | `sell_price` | 1328589.10 |
| 8 | `wday` | 1085638.84 |
| 9 | `price_max` | 1021560.45 |
| 10 | `price_norm_mean` | 1007797.67 |

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