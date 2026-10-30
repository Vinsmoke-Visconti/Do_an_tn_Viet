"""
Module: pipeline_arima_pure.py
Mục đích: Pipeline huấn luyện và đánh giá mô hình Baseline ARIMA(7,1,1)
          thuần túy trên 10 Stores + cấp Toàn quốc của bộ dữ liệu M5 (Walmart).

Mục tiêu nghiên cứu:
    Cung cấp mức sàn (performance floor) của mô hình thống kê tuyến tính
    không có thành phần mùa vụ, dùng để so sánh với:
        - SARIMA  (có thêm seasonal component)
        - LightGBM (mô hình Gradient Boosting)
        - TFT      (Temporal Fusion Transformer)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_arima_pure import PureARIMAModel


# ── Metrics ────────────────────────────────────────────────────────────────────

def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))

def _wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.sum(np.abs(y))
    return 0.0 if d == 0 else float(np.sum(np.abs(y - yhat)) / d * 100.0)

def _smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-5) -> float:
    denom = (np.abs(y) + np.abs(yhat)) / 2.0 + eps
    return float(np.mean(np.abs(yhat - y) / denom) * 100.0)

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).flatten(), 0.0, None)
    return {
        "RMSE":      round(_rmse(y_true, y_pred),  4),
        "MAE":       round(_mae(y_true, y_pred),   4),
        "WMAPE (%)": round(_wmape(y_true, y_pred), 4),
        "sMAPE (%)": round(_smape(y_true, y_pred), 4),
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────

class PureARIMAPipeline:
    """
    Pipeline thực nghiệm cho mô hình Baseline ARIMA thuần túy.

    Không có:
      - Seasonal component (không phải SARIMA)
      - Exogenous variables (không phải ARIMAX)

    Chỉ có:
      - ARIMA(p, d, q) với statsmodels
    """

    def __init__(
        self,
        data_path: str = None,
        order: Tuple[int, int, int] = (7, 1, 1),
        test_days: int = 28,
        test_start_d: int = 1886,
    ):
        """
        Args:
            data_path:    Đường dẫn tới m5_store_level_features.parquet.
            order:        (p, d, q) cho ARIMA. Mặc định (7, 1, 1):
                            p=7 → bắt trễ 7 ngày thay cho mùa vụ tuần.
                            d=1 → sai phân bậc 1 để làm chuỗi dừng.
                            q=1 → MA(1) xử lý shock ngắn hạn.
            test_days:    Số ngày dự báo (M5 = 28).
            test_start_d: Ngày bắt đầu test (d_1886 trong M5).
        """
        if data_path is None:
            data_path = os.path.join(
                CURRENT_DIR, "..", "new_source", "data", "processed",
                "m5_store_level_features.parquet",
            )
        self.data_path    = os.path.abspath(data_path)
        self.order        = order
        self.test_days    = test_days
        self.test_start_d = test_start_d

        self.output_dir  = os.path.join(CURRENT_DIR, "outputs")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["store_id", "date"]).reset_index(drop=True)

    # ── Per-series ─────────────────────────────────────────────────────────────

    def run_series(
        self,
        df_series: pd.DataFrame,
        series_id: str,
    ) -> Dict[str, Any]:
        """
        Huấn luyện và dự báo ARIMA cho một chuỗi (1 store hoặc national aggregate).

        Args:
            df_series: DataFrame đã chứa cột [d_num, total_sales, date].
            series_id: Tên định danh (vd: "CA_1", "NATIONAL_TOTAL").
        """
        df_series = df_series.sort_values("d_num").reset_index(drop=True)

        # Train: tất cả quan sát trước ngày test đầu tiên
        df_train = df_series[df_series["d_num"] < self.test_start_d].copy()

        # Test: đúng test_days hàng đầu tiên từ test_start_d
        df_test = (
            df_series[df_series["d_num"] >= self.test_start_d]
            .head(self.test_days)
            .copy()
        )

        if len(df_test) == 0:
            raise ValueError(f"[{series_id}] Khong co du lieu test tu d_num={self.test_start_d}.")

        y_train    = df_train["total_sales"].values.astype(float)
        y_test     = df_test["total_sales"].values.astype(float)
        dates_test = df_test["date"].dt.strftime("%d-%m-%Y").tolist()
        d_num_test = df_test["d_num"].tolist()

        # Fit mô hình ARIMA thuần túy
        model = PureARIMAModel(order=self.order, name=f"ARIMA_{series_id}")
        model.fit(y_train)

        # Dự báo
        y_pred, ci_lo, ci_hi = model.predict(
            horizon=len(y_test),
            alpha=0.05,
        )

        metrics = _evaluate(y_test, y_pred)

        # Tính sai số phần trăm (tránh chia 0)
        err_pct = np.where(
            y_test > 1e-5,
            np.abs(y_test - y_pred) / y_test * 100.0,
            0.0,
        )

        df_forecast = pd.DataFrame({
            "series_id":      series_id,
            "d_num":          d_num_test,
            "date":           dates_test,
            "actual_sales":   y_test,
            "forecast_arima": y_pred,
            "ci_lower_95":    ci_lo if ci_lo is not None else y_pred,
            "ci_upper_95":    ci_hi if ci_hi is not None else y_pred,
            "error_abs":      np.abs(y_test - y_pred),
            "error_pct":      err_pct,
        })

        return {
            "series_id":       series_id,
            "metrics":         metrics,
            "forecast_df":     df_forecast,
            "y_train_tail":    y_train[-56:],
            "dates_train_tail": df_train["date"].dt.strftime("%d-%m-%Y").values[-56:],
        }

    # ── Run all ────────────────────────────────────────────────────────────────

    def run_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Chạy lần lượt 10 Stores + 1 chuỗi National Aggregate.

        Returns:
            (full_forecast_df, metrics_df, series_details)
        """
        df = self.load_data()
        stores = sorted(df["store_id"].unique().tolist())

        all_forecasts: list = []
        metrics_rows: list  = []
        details: dict       = {}

        print("=" * 60)
        print("  BASELINE: Pure ARIMA(p,d,q) — No Seasonal, No Exogenous")
        print(f"  Order = {self.order}  |  Horizon = {self.test_days} days")
        print("=" * 60)

        # ── 10 Stores ─────────────────────────────────────────────────────────
        for store in stores:
            print(f"  >> {store:12s} ...", end="", flush=True)
            res = self.run_series(df[df["store_id"] == store], store)
            m = res["metrics"]
            print(f"  WMAPE={m['WMAPE (%)']:6.2f}%  RMSE={m['RMSE']:8.1f}")

            all_forecasts.append(res["forecast_df"])
            metrics_rows.append({"series_id": store, **m})
            details[store] = res

        # ── National Aggregate ─────────────────────────────────────────────────
        print(f"  >> {'NATIONAL':12s} ...", end="", flush=True)
        df_nat = (
            df.groupby(["date", "d_num"], as_index=False)["total_sales"]
            .sum()
            .sort_values("d_num")
            .reset_index(drop=True)
        )
        df_nat["store_id"] = "NATIONAL_TOTAL"

        res_nat = self.run_series(df_nat, "NATIONAL_TOTAL")
        m_nat = res_nat["metrics"]
        print(f"  WMAPE={m_nat['WMAPE (%)']:6.2f}%  RMSE={m_nat['RMSE']:8.1f}")

        all_forecasts.append(res_nat["forecast_df"])
        metrics_rows.append({"series_id": "NATIONAL_TOTAL", **m_nat})
        details["NATIONAL_TOTAL"] = res_nat

        # ── Tổng hợp ──────────────────────────────────────────────────────────
        full_df    = pd.concat(all_forecasts, ignore_index=True)
        metrics_df = pd.DataFrame(metrics_rows)

        # Trung bình 10 Stores
        store_mask = metrics_df["series_id"] != "NATIONAL_TOTAL"
        avg = {
            "series_id":   "STORE_AVERAGE",
            "RMSE":        round(float(metrics_df.loc[store_mask, "RMSE"].mean(skipna=True)),       4),
            "MAE":         round(float(metrics_df.loc[store_mask, "MAE"].mean(skipna=True)),        4),
            "WMAPE (%)":   round(float(metrics_df.loc[store_mask, "WMAPE (%)"].mean(skipna=True)), 4),
            "sMAPE (%)":   round(float(metrics_df.loc[store_mask, "sMAPE (%)"].mean(skipna=True)), 4),
        }
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([avg])], ignore_index=True
        )

        # ── Lưu kết quả ───────────────────────────────────────────────────────
        fc_path  = os.path.join(self.results_dir, "arima_pure_forecast.csv")
        met_path = os.path.join(self.results_dir, "arima_pure_metrics.csv")
        js_path  = os.path.join(self.results_dir, "arima_pure_summary.json")

        full_df.to_csv(fc_path,   index=False, encoding="utf-8-sig")
        metrics_df.to_csv(met_path, index=False, encoding="utf-8-sig")

        summary = {
            "model_type":   "ARIMA (pure — no seasonal, no exogenous)",
            "order":        list(self.order),
            "test_horizon": self.test_days,
            "test_start_d": self.test_start_d,
            "metrics":      metrics_df.to_dict(orient="records"),
        }
        with open(js_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"\n  [OK] Forecast CSV : {fc_path}")
        print(f"  [OK] Metrics CSV  : {met_path}")
        print(f"  [OK] JSON summary : {js_path}")

        return full_df, metrics_df, details

