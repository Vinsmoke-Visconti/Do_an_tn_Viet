# LightGBM Top-1 M5 Forecasting Pipeline

Bộ mã nguồn thực hiện **dự báo nhu cầu 28 ngày** cho bộ dữ liệu **Walmart M5 Forecasting**  
theo phương pháp của đội **Top 1 M5 Competition (YJ_STU)**.

---

## Cấu trúc thư mục

```
new_source/
├── 01_preprocessing.py        # Bước 1: Tiền xử lý dữ liệu
├── 02_train_and_forecast.py   # Bước 2: Huấn luyện & Dự báo 28 ngày đệ quy
├── 03_evaluation_report.py    # Bước 3: Đánh giá & Xuất báo cáo Markdown
├── run_all.py                 # Chạy toàn bộ 3 bước
│
├── src/
│   ├── data/
│   │   ├── m5_pipeline.py         # Pipeline tiền xử lý M5 (Melt, Merge, Feature Engineering)
│   │   └── memory_optimizer.py    # Tối ưu RAM (ép kiểu dữ liệu Pandas)
│   ├── models/
│   │   ├── lightgbm_tweedie.py    # Class LightGBM + Tweedie Loss
│   │   └── recursive_forecaster.py# Dự báo đệ quy 28 bước
│   └── utils/
│       ├── metrics.py             # RMSE, MAE, WMAPE, sMAPE
│       ├── hierarchical_evaluation.py  # Đánh giá 3 cấp: SKU/Store/National
│       └── visualizer.py          # 5 loại biểu đồ
│
├── data/
│   ├── raw/                   # calendar.csv, sell_prices.csv, sales_train_validation.csv
│   └── processed/             # Parquet files sau khi tiền xử lý
│
└── outputs/
    ├── figures/               # Biểu đồ PNG
    ├── models/                # File mô hình .pkl
    ├── results/               # forecast_results.parquet, metrics_summary.json, ...
    └── reports/               # Báo cáo Markdown tự động
```

---

## Hướng dẫn chạy

### Cài đặt thư viện
```bash
pip install lightgbm pandas numpy scikit-learn matplotlib seaborn pyarrow
```

### Cách 1: Chạy toàn bộ pipeline (khuyến nghị)
```bash
cd new_source
python run_all.py
```

### Cách 2: Chạy từng bước riêng lẻ
```bash
# Bước 1: Tiền xử lý dữ liệu M5
python 01_preprocessing.py

# Bước 2: Huấn luyện LightGBM & Dự báo 28 ngày đệ quy
python 02_train_and_forecast.py

# Bước 3: Đánh giá hoàn chỉnh & Xuất báo cáo
python 03_evaluation_report.py
```

### Bỏ qua bước 1 (khi đã có data processed)
```bash
python run_all.py --skip 1
```

---

## Phương pháp (Top 1 M5 Competition)

| Thành phần | Chi tiết |
|---|---|
| **Mô hình** | LightGBM (Gradient Boosted Decision Trees) |
| **Hàm mục tiêu** | Tweedie Loss (variance_power=1.1) — xử lý Intermittent Demand |
| **Dự báo** | Recursive Multi-Step Forecasting (28 bước đệ quy) |
| **Mô hình hóa** | Per-Store (1 mô hình riêng cho mỗi cửa hàng) |
| **Feature Engineering** | Lag (28/35/42/56), Rolling Mean/Std (7/28 ngày), EWM, Cyclical, Event, Price |
| **Đánh giá** | RMSE, WMAPE, sMAPE ở 3 cấp: SKU / Store / National |
| **Diễn giải** | Feature Importance (Gain), Nhóm đặc trưng (Pie chart) |

---

## Kết quả sinh ra

| File | Mô tả |
|---|---|
| `outputs/results/forecast_results.parquet` | Kết quả dự báo 28 ngày cho toàn bộ SKU |
| `outputs/results/feature_importance_gain.csv` | Bảng Feature Importance (Gain) |
| `outputs/results/hierarchical_evaluation.csv` | Đánh giá 3 cấp độ |
| `outputs/results/metrics_summary.json` | Tóm tắt độ đo dạng JSON |
| `outputs/reports/BAO_CAO_LIGHTGBM_TOP1.md` | Báo cáo tự động dạng Markdown |
| `outputs/figures/forecast_28day_*.png` | Biểu đồ dự báo 28 ngày |
| `outputs/figures/feature_importance_gain.png` | Biểu đồ Feature Importance |
| `outputs/figures/feature_group_pie.png` | Pie chart đóng góp nhóm đặc trưng |
| `outputs/figures/daily_wmape_heatmap.png` | Heatmap WMAPE 28 ngày |
| `outputs/figures/model_comparison_*.png` | So sánh các mô hình |
| `outputs/models/lgb_tweedie_<store>.pkl` | Mô hình LightGBM đã huấn luyện |
