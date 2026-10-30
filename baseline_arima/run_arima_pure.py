"""
Script: run_arima_pure.py
Mục đích: Entry point chạy toàn bộ pipeline Baseline ARIMA(7,1,1) thuần túy.

Mô hình:
    ARIMA(p=7, d=1, q=1) — không có Seasonal, không có Exogenous.
    Đây là mô hình tuyến tính cổ điển Box-Jenkins đơn giản nhất.

Chạy lệnh:
    python run_arima_pure.py

Kết quả đầu ra:
    outputs/results/arima_pure_forecast.csv  — dự báo 28 ngày mỗi Store
    outputs/results/arima_pure_metrics.csv   — bảng RMSE / MAE / WMAPE / sMAPE
    outputs/results/arima_pure_summary.json  — tóm tắt JSON
    outputs/figures/01_arima_forecast_vs_actual.png
    outputs/figures/02_arima_store_metrics.png
    outputs/figures/03_arima_horizon_error.png
"""

import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pipeline_arima_pure import PureARIMAPipeline
from visualize_arima_pure import (
    plot_forecast_comparison,
    plot_store_metrics,
    plot_horizon_error,
)


def main():
    t0 = time.time()

    # ── Khởi tạo pipeline ─────────────────────────────────────────────────────
    pipeline = PureARIMAPipeline(
        order=(7, 1, 1),    # AR(7): trễ 7 ngày bắt chu kỳ tuần không cần seasonal
        test_days=28,
        test_start_d=1886,  # d_1886 ~ ngày 29-03-2016 trong M5
    )

    # ── Chạy huấn luyện + dự báo ──────────────────────────────────────────────
    full_df, metrics_df, details = pipeline.run_all()

    # ── Trực quan hóa ─────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    plot_forecast_comparison(
        details,
        os.path.join(pipeline.figures_dir, "01_arima_forecast_vs_actual.png"),
    )
    plot_store_metrics(
        metrics_df,
        os.path.join(pipeline.figures_dir, "02_arima_store_metrics.png"),
    )
    plot_horizon_error(
        full_df,
        os.path.join(pipeline.figures_dir, "03_arima_horizon_error.png"),
    )

    # ── Bảng tổng hợp ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY — Pure ARIMA(7,1,1) Baseline (M5 Walmart)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)
    print(f"\n  Done in {time.time() - t0:.2f} s")
    print(f"  Outputs: {pipeline.output_dir}")


if __name__ == "__main__":
    main()

