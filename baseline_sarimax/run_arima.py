"""
Script: run_arima.py
Mục đích: Chạy toàn bộ pipeline Baseline ARIMA / SARIMAX cho bài toán M5:
          1. Đọc dữ liệu Store-level đã xử lý.
          2. Huấn luyện mô hình ARIMA/SARIMAX trên từng Store và Toàn quốc.
          3. Dự báo 28 ngày (H=28).
          4. Tính toán toàn bộ các metrics (RMSE, MAE, WMAPE, sMAPE).
          5. Xuất báo cáo, bảng kết quả và tạo 3 biểu đồ trực quan khoa học.
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pipeline_arima import ARIMAPipeline
from visualize_arima import (
    plot_arima_forecast_comparison,
    plot_store_metrics_comparison,
    plot_daily_error_distribution
)


def main():
    print("=" * 80)
    print("      [BASELINE] TRAINING & EVALUATION PIPELINE: ARIMA / SARIMAX")
    print("               DATASET: M5 FORECASTING ACCURACY (WALMART)")
    print("=" * 80)

    start_time = time.time()

    # 1. Khoi tao Pipeline
    pipeline = ARIMAPipeline(
        test_days=28,
        test_start_d=1886,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),  # SARIMAX chu ky tuan s=7
        use_exog=False
    )

    # 2. Chay huan luyen va du bao cho toan bo cac Store
    full_forecast_df, metrics_df, store_details = pipeline.run_all()

    # 3. Truc quan hoa ket qua
    print("\n[*] Generating high-resolution scientific visualization plots...")
    fig1_path = os.path.join(pipeline.figures_dir, "01_arima_forecast_vs_actual.png")
    fig2_path = os.path.join(pipeline.figures_dir, "02_arima_store_metrics_comparison.png")
    fig3_path = os.path.join(pipeline.figures_dir, "03_arima_daily_residuals.png")

    plot_arima_forecast_comparison(store_details, fig1_path)
    plot_store_metrics_comparison(metrics_df, fig2_path)
    plot_daily_error_distribution(full_forecast_df, fig3_path)

    # 4. Hien thi bang tong hop ket qua
    print("\n" + "=" * 80)
    print("              SUMMARY EVALUATION METRICS (ARIMA BASELINE)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)

    elapsed = time.time() - start_time
    print(f"\n[OK] Pipeline completed successfully in {elapsed:.2f} seconds!")
    print(f"[OK] Outputs directory: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
