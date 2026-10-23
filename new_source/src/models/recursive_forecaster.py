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
        unique_ids = future_features_df[id_col].unique()
        future_days = sorted(future_features_df["d_num"].unique())[:self.horizon]

        # Trích xuất 60 ngày lịch sử gần nhất để tạo lag/rolling buffer
        max_hist_d = history_df["d_num"].max()
        hist_recent = history_df[history_df["d_num"] > (max_hist_d - 60)][[id_col, "d_num", target_col]]
        pivot_hist = (
            hist_recent.pivot(index=id_col, columns="d_num", values=target_col)
            .reindex(unique_ids)
            .fillna(0)
            .values.astype(np.float32)
        )
        buffer = pivot_hist.copy()

        results = []
        for h, d_val in enumerate(future_days):
            day_mask = (future_features_df["d_num"] == d_val)
            day_rows = future_features_df[day_mask].copy()
            day_rows = day_rows.set_index(id_col).reindex(unique_ids).reset_index()

            # 1. Cập nhật lag features dựa trên buffer đệ quy
            for lag_name, lag_days in [
                ("sales_lag_28", 28), ("sales_lag_35", 35),
                ("sales_lag_42", 42), ("sales_lag_56", 56)
            ]:
                if lag_name in feature_cols:
                    idx = -(lag_days - h)
                    if abs(idx) <= buffer.shape[1]:
                        day_rows[lag_name] = buffer[:, idx]

            # 2. Cập nhật rolling features (tính trên buffer đệ quy)
            w28 = buffer[:, -28:]
            w7  = w28[:, -7:]

            if "sales_rolling_mean_7" in feature_cols:
                day_rows["sales_rolling_mean_7"] = np.mean(w7, axis=1).astype(np.float32)
            if "sales_rolling_std_7" in feature_cols:
                day_rows["sales_rolling_std_7"] = np.std(w7, axis=1).astype(np.float32)
            if "sales_rolling_mean_28" in feature_cols:
                day_rows["sales_rolling_mean_28"] = np.mean(w28, axis=1).astype(np.float32)
            if "sales_rolling_std_28" in feature_cols:
                day_rows["sales_rolling_std_28"] = np.std(w28, axis=1).astype(np.float32)

            # 3. Chuyển kiểu categorical nếu cần
            for col in day_rows.select_dtypes(include=["object"]).columns:
                day_rows[col] = day_rows[col].astype("category")

            # 4. Dự báo vector hóa toàn bộ SKUs cho ngày h
            preds = self.model.predict(day_rows[feature_cols])
            preds = np.clip(preds, 0, None).astype(np.float32)

            day_rows[f"pred_{target_col}"] = preds
            # Cập nhật buffer cho các bước tiếp theo
            buffer = np.hstack([buffer, preds[:, None]])
            results.append(day_rows)

        return pd.concat(results, ignore_index=True)
