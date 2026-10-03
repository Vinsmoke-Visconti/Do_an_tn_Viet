"""
Module: recursive_forecaster.py  (new_source)
Mục đích: Thực hiện dự báo đệ quy 28 ngày (Recursive Multi-Step Forecasting).

Cách thức hoạt động:
  Tại mỗi bước t+h (h = 1..28):
    1. Lấy ngữ cảnh lịch sử (lag features) từ window [t-max_lag .. t+h-1].
    2. Nếu h <= 28: các giá trị lag trong tương lai được thay bằng
       dự báo đã sinh ở bước trước đó (dự báo đệ quy).
    3. Dự báo bán hàng t+h bằng mô hình LightGBM.
    4. Thêm dự báo vào buffer để làm input cho bước tiếp theo.

Chiến lược này là cốt lõi của đội Top 1 M5 và giúp mô hình có thể
sinh ra chuỗi 28 ngày liên tục mà không cần dữ liệu tương lai thực tế.
"""
import numpy as np
import pandas as pd
from typing import List, Optional


class RecursiveForecaster:
    """
    Bộ dự báo đệ quy 28 bước cho LightGBM.

    Sử dụng khi cần dự báo nhiều bước tương lai mà không có ground truth
    của các ngày giữa chừng (đây là tình huống thực tế khi triển khai).
    """

    # Các cột lag và rolling cần cập nhật ở mỗi bước đệ quy
    LAG_COLS    = ["sales_lag_28", "sales_lag_35", "sales_lag_42", "sales_lag_56"]
    ROLLING_COLS = [
        "sales_rolling_mean_7",  "sales_rolling_std_7",
        "sales_rolling_mean_28", "sales_rolling_std_28",
        "sales_ewm_alpha02",     "sales_ewm_alpha05",
    ]

    def __init__(self, model, horizon: int = 28):
        """
        Args:
            model   : Đối tượng có phương thức `.predict(X) -> np.ndarray`.
            horizon : Số bước dự báo tương lai (mặc định 28).
        """
        self.model   = model
        self.horizon = horizon

    def forecast(
        self,
        history_df: pd.DataFrame,
        future_features_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "sales",
        id_col: str = "id",
    ) -> pd.DataFrame:
        """
        Dự báo đệ quy cho từng chuỗi thời gian.

        Args:
            history_df         : Lịch sử bán hàng của TẤT CẢ chuỗi (để tính lag).
                                 Phải có cột `id_col`, `d_num`, và `target_col`.
            future_features_df : Đặc trưng của 28 ngày tương lai (calendar, giá đã biết trước).
                                 Phải có cột `id_col`, `d_num`.
            feature_cols       : Danh sách tên cột đặc trưng dùng cho mô hình.
            target_col         : Cột doanh số ("sales" hoặc "total_sales").
            id_col             : Cột định danh chuỗi ("id" hoặc "store_id").

        Returns:
            pd.DataFrame: Kết quả dự báo, có thêm cột `pred_{target_col}`.
        """
        all_ids = future_features_df[id_col].unique()
        results = []

        for series_id in all_ids:
            hist   = history_df[history_df[id_col] == series_id].copy()
            future = future_features_df[future_features_df[id_col] == series_id].copy()
            future = future.sort_values("d_num").reset_index(drop=True)

            # Buffer dự báo cho bước đệ quy
            pred_buffer = hist[[target_col]].values.flatten().tolist()

            preds = []
            for h in range(self.horizon):
                row = future.iloc[[h]].copy()

                # Cập nhật lag features dựa trên buffer dự báo
                for lag_name, lag_days in [
                    ("sales_lag_28", 28), ("sales_lag_35", 35),
                    ("sales_lag_42", 42), ("sales_lag_56", 56)
                ]:
                    if lag_name in feature_cols:
                        idx = -(lag_days - h)    # vị trí trong buffer
                        if abs(idx) <= len(pred_buffer):
                            row[lag_name] = pred_buffer[idx]

                # Cập nhật rolling features (tính trên pred_buffer)
                window_28 = np.array(pred_buffer[-28:], dtype=np.float32)
                window_7  = window_28[-7:]

                if "sales_rolling_mean_7" in feature_cols:
                    row["sales_rolling_mean_7"]  = float(np.mean(window_7))
                if "sales_rolling_std_7" in feature_cols:
                    row["sales_rolling_std_7"]   = float(np.std(window_7)) if len(window_7) > 1 else 0.0
                if "sales_rolling_mean_28" in feature_cols:
                    row["sales_rolling_mean_28"] = float(np.mean(window_28))
                if "sales_rolling_std_28" in feature_cols:
                    row["sales_rolling_std_28"]  = float(np.std(window_28)) if len(window_28) > 1 else 0.0
                if "sales_ewm_alpha02" in feature_cols:
                    row["sales_ewm_alpha02"] = float(
                        pd.Series(pred_buffer).ewm(alpha=0.2, min_periods=1).mean().iloc[-1])
                if "sales_ewm_alpha05" in feature_cols:
                    row["sales_ewm_alpha05"] = float(
                        pd.Series(pred_buffer).ewm(alpha=0.5, min_periods=1).mean().iloc[-1])

                # Đảm bảo đúng thứ tự cột và kiểu dữ liệu
                X_row = row[feature_cols].copy()
                for col in X_row.select_dtypes(include=["object"]).columns:
                    X_row[col] = X_row[col].astype("category")

                pred_val = float(np.clip(self.model.predict(X_row)[0], 0, None))
                preds.append(pred_val)
                pred_buffer.append(pred_val)   # Đẩy dự báo vào buffer đệ quy

            future[f"pred_{target_col}"] = preds
            results.append(future)

        return pd.concat(results, ignore_index=True)
