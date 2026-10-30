"""
Module: pipeline_arima.py
Mục đích: Quy trình huấn luyện, dự báo 28 ngày và đánh giá mô hình
          Baseline ARIMA / SARIMAX trên 10 Stores + cấp Toàn quốc (M5).

BUG-FIXES so với phiên bản cũ:
  [5] run_store(): test_mask dùng < test_start_d + test_days, nhưng nếu d_num
      không liên tục (có ngày bị thiếu) thì số hàng < 28.
      -> Sửa: dùng .iloc[:test_days] sau khi lọc theo d_num >= test_start_d
         để luôn đảm bảo đúng 28 quan sát.
  [6] National aggregate: groupby "date"+"d_num" nhưng không sắp xếp trước;
      nếu bảng gốc chưa sort -> d_num có thể không tương ứng với date.
      -> Sửa: sort_values sau groupby.
  [7] error_percentage: chia y_test (float array) chứa 0 -> inf/nan.
      -> Sửa: dùng np.where với eps để tránh chia 0.
  [8] STORE_AVERAGE: tính mean trên DataFrame đang còn có thể bao gồm NaN
      nếu một store bị fallback; dùng .mean(skipna=True) tường minh.
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

from model_arima import M5ARIMABaseline, SARIMAFitError, SARIMAPredictError, SARIMANotFittedError

# ── Metrics ────────────────────────────────────────────────────────────────────

def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def _wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.sum(np.abs(y))
    return 0.0 if denom == 0 else float(np.sum(np.abs(y - yhat)) / denom * 100.0)


def _smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-5) -> float:
    denom = (np.abs(y) + np.abs(yhat)) / 2.0 + eps
    return float(np.mean(np.abs(yhat - y) / denom) * 100.0)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.clip(np.asarray(y_pred, dtype=float).flatten(), 0, None)
    return {
        "RMSE":      round(_rmse(y_true, y_pred),  4),
        "MAE":       round(_mae(y_true, y_pred),   4),
        "WMAPE (%)": round(_wmape(y_true, y_pred), 4),
        "sMAPE (%)": round(_smape(y_true, y_pred), 4),
    }


# ── Pipeline ───────────────────────────────────────────────────────────────────

class ARIMAPipeline:
    """Pipeline thực nghiệm cho mô hình Baseline ARIMA / SARIMAX."""

    def __init__(
        self,
        data_path: str = None,
        test_days: int = 28,
        test_start_d: int = 1886,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        use_exog: bool = False,
    ):
        if data_path is None:
            data_path = os.path.join(
                CURRENT_DIR, "..", "new_source", "data", "processed",
                "m5_store_level_features.parquet",
            )
        self.data_path     = os.path.abspath(data_path)
        self.test_days     = test_days
        self.test_start_d  = test_start_d
        self.order         = order
        self.seasonal_order = seasonal_order
        self.use_exog      = use_exog

        self.output_dir  = os.path.join(CURRENT_DIR, "outputs")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
        return df

    # ── Per-store run ──────────────────────────────────────────────────────────

    def run_store(
        self,
        df_store: pd.DataFrame,
        store_id: str,
    ) -> Dict[str, Any]:
        """Huấn luyện và dự báo ARIMA cho một chuỗi (store hoặc national)."""

        df_store = df_store.sort_values("d_num").reset_index(drop=True)

        # Train: tất cả quan sát trước ngày test đầu tiên
        df_train = df_store[df_store["d_num"] < self.test_start_d].copy()

        # [FIX-5] Test: lấy đúng test_days hàng đầu tiên từ ngày test_start_d
        df_test = (
            df_store[df_store["d_num"] >= self.test_start_d]
            .head(self.test_days)         # đảm bảo đúng 28 quan sát
            .copy()
        )

        if len(df_test) == 0:
            raise ValueError(
                f"[{store_id}] Không tìm thấy dữ liệu test từ d_num={self.test_start_d}."
            )

        y_train     = df_train["total_sales"].values.astype(float)
        y_test      = df_test["total_sales"].values.astype(float)
        dates_test  = df_test["date"].dt.strftime("%d-%m-%Y").tolist()
        d_num_test  = df_test["d_num"].tolist()

        # Biến ngoại sinh (tuỳ chọn)
        exog_train = exog_test = None
        if self.use_exog:
            exog_cols = ["is_weekend", "snap_active", "has_event"]
            # Chỉ dùng các cột tồn tại trong df
            exog_cols = [c for c in exog_cols if c in df_train.columns]
            if exog_cols:
                exog_train = df_train[exog_cols].values.astype(float)
                exog_test  = df_test[exog_cols].values.astype(float)

        # Fit & Predict — SARIMAFitError / SARIMAPredictError nếu thất bại (Hướng A)
        model = M5ARIMABaseline(
            order=self.order,
            seasonal_order=self.seasonal_order,
            name=f"SARIMA_{store_id}",
        )
        model.fit(y_train, exog_train=exog_train)  # SARIMAFitError truyền lên trên

        horizon = len(y_test)
        y_pred, lower_b, upper_b = model.predict(
            horizon=horizon,
            exog_future=exog_test,
            return_conf_int=True,
        )  # SARIMAPredictError truyền lên trên

        metrics = _evaluate(y_test, y_pred)

        # [FIX-7] Tránh chia 0 khi tính error_percentage
        _eps = 1e-5
        err_pct = np.where(
            y_test > _eps,
            np.abs(y_test - y_pred) / y_test * 100.0,
            0.0,
        )

        df_forecast = pd.DataFrame({
            "store_id":        store_id,
            "d_num":           d_num_test,
            "date":            dates_test,
            "actual_sales":    y_test,
            "forecast_sarima": y_pred,
            # Luôn có CI vì không có fallback (Hướng A)
            "ci_lower_95":     lower_b,
            "ci_upper_95":     upper_b,
            "error_abs":       np.abs(y_test - y_pred),
            "error_pct":       err_pct,
        })

        return {
            "store_id":        store_id,
            "metrics":         metrics,
            "forecast_df":     df_forecast,
            # 8 tuần cuối chuỗi train để vẽ biểu đồ liền mạch
            "y_train_tail":    y_train[-56:],
            "dates_train_tail": df_train["date"].dt.strftime("%d-%m-%Y").values[-56:],
        }

    # ── Run all stores ─────────────────────────────────────────────────────────

    def run_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Chạy toàn bộ 10 Stores + Toàn quốc (National Aggregate)."""

        df = self.load_data()
        stores = sorted(df["store_id"].unique().tolist())

        all_forecast_dfs: list = []
        metrics_list:     list = []
        store_details:    dict = {}

        print(f"[*] Baseline ARIMA/SARIMAX | {len(stores)} Stores")
        print(f"[*] Order={self.order}  SeasonalOrder={self.seasonal_order}  Exog={self.use_exog}")

        for store in stores:
            print(f"    >> Store: {store} ...", end="", flush=True)
            res = self.run_store(df[df["store_id"] == store], store)
            print(f"  WMAPE={res['metrics']['WMAPE (%)']:.2f}%  RMSE={res['metrics']['RMSE']:.1f}")

            all_forecast_dfs.append(res["forecast_df"])
            metrics_list.append({"store_id": store, **res["metrics"]})
            store_details[store] = res

        # ── National Aggregate (Level 1) ──────────────────────────────────────
        print("    >> National Aggregate ...", end="", flush=True)

        # [FIX-6] Sort sau groupby để d_num luôn tương ứng với date
        df_nat = (
            df.groupby(["date", "d_num"], as_index=False)["total_sales"]
            .sum()
            .sort_values("d_num")
            .reset_index(drop=True)
        )
        df_nat["store_id"]   = "NATIONAL_TOTAL"
        df_nat["is_weekend"] = df_nat["date"].dt.dayofweek.isin([5, 6]).astype(int)
        df_nat["snap_active"] = 0
        df_nat["has_event"]   = 0

        res_nat = self.run_store(df_nat, "NATIONAL_TOTAL")
        print(f"  WMAPE={res_nat['metrics']['WMAPE (%)']:.2f}%  RMSE={res_nat['metrics']['RMSE']:.1f}")

        all_forecast_dfs.append(res_nat["forecast_df"])
        metrics_list.append({"store_id": "NATIONAL_TOTAL", **res_nat["metrics"]})
        store_details["NATIONAL_TOTAL"] = res_nat

        # ── Tổng hợp ──────────────────────────────────────────────────────────
        full_df   = pd.concat(all_forecast_dfs, ignore_index=True)
        metrics_df = pd.DataFrame(metrics_list)

        # [FIX-8] Trung bình chỉ tính trên 10 Stores (skipna=True tường minh)
        store_mask = ~metrics_df["store_id"].isin(["NATIONAL_TOTAL"])
        avg_row = {
            "store_id":   "STORE_AVERAGE",
            "RMSE":       round(float(metrics_df.loc[store_mask, "RMSE"].mean(skipna=True)), 4),
            "MAE":        round(float(metrics_df.loc[store_mask, "MAE"].mean(skipna=True)), 4),
            "WMAPE (%)":  round(float(metrics_df.loc[store_mask, "WMAPE (%)"].mean(skipna=True)), 4),
            "sMAPE (%)":  round(float(metrics_df.loc[store_mask, "sMAPE (%)"].mean(skipna=True)), 4),
        }
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([avg_row])], ignore_index=True
        )

        # ── Lưu kết quả ───────────────────────────────────────────────────────
        fc_path      = os.path.join(self.results_dir, "arima_forecast_results.csv")
        metrics_path = os.path.join(self.results_dir, "arima_metrics_table.csv")
        json_path    = os.path.join(self.results_dir, "arima_metrics_summary.json")

        full_df.to_csv(fc_path,      index=False, encoding="utf-8-sig")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        summary = {
            "model":          "SARIMAX" if self.seasonal_order else "ARIMA",
            "order":          list(self.order),
            "seasonal_order": list(self.seasonal_order) if self.seasonal_order else None,
            "use_exog":       self.use_exog,
            "test_horizon":   self.test_days,
            "test_start_d":   self.test_start_d,
            "metrics":        metrics_df.to_dict(orient="records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"\n[OK] Forecast CSV  : {fc_path}")
        print(f"[OK] Metrics CSV   : {metrics_path}")
        print(f"[OK] Summary JSON  : {json_path}")

        return full_df, metrics_df, store_details
