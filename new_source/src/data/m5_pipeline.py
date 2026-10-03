"""
Module: m5_pipeline.py  (new_source version)
Mục đích: Pipeline tiền xử lý M5 hoàn chỉnh.
  1. Load & tối ưu hóa 3 bảng: calendar, sell_prices, sales_train_validation.
  2. Melt Wide -> Long format.
  3. Xử lý Missing Values.
  4. Trích xuất đặc trưng thời gian (Cyclical, Event, SNAP).
  5. Trích xuất đặc trưng giá (Discount, Momentum, Rolling Stats).
  6. Tạo Lags & Rolling Stats (CHỐNG LEAKAGE: min lag = 28 ngày).
  7. Tổng hợp đa cấp độ (SKU-level, Store-level).
"""

import os
import gc
import numpy as np
import pandas as pd
from typing import Optional, List
from .memory_optimizer import reduce_mem_usage


class M5DataPipeline:
    """Pipeline tiền xử lý dữ liệu M5 Walmart theo phương pháp Top 1."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.calendar_path  = os.path.join(data_dir, "calendar.csv")
        self.prices_path    = os.path.join(data_dir, "sell_prices.csv")
        self.sales_path     = os.path.join(data_dir, "sales_train_validation.csv")
        self._calendar_cache = None   # Cache để tránh load lại nhiều lần

    # ------------------------------------------------------------------
    # 1. Load Calendar
    # ------------------------------------------------------------------
    def load_calendar(self) -> pd.DataFrame:
        if self._calendar_cache is not None:
            return self._calendar_cache

        print("[+] Load calendar.csv ...")
        cal = pd.read_csv(self.calendar_path)
        cal = reduce_mem_usage(cal, verbose=False)
        cal["date"] = pd.to_datetime(cal["date"])

        # Trích xuất các biến thời gian cơ bản
        cal["day"]        = cal["date"].dt.day.astype(np.int8)
        cal["week"]       = cal["date"].dt.isocalendar().week.astype(np.int8)
        cal["month"]      = cal["date"].dt.month.astype(np.int8)
        cal["year"]       = cal["date"].dt.year.astype(np.int16)
        cal["is_weekend"] = cal["wday"].isin([1, 2]).astype(np.int8)

        # Mã hóa tuần hoàn (Cyclical Encoding)
        cal["sin_wday"]   = np.sin(2 * np.pi * cal["wday"] / 7).astype(np.float32)
        cal["cos_wday"]   = np.cos(2 * np.pi * cal["wday"] / 7).astype(np.float32)
        cal["sin_month"]  = np.sin(2 * np.pi * cal["month"] / 12).astype(np.float32)
        cal["cos_month"]  = np.cos(2 * np.pi * cal["month"] / 12).astype(np.float32)

        # Xử lý sự kiện (Events)
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            cal[col] = cal[col].fillna("None").astype("category")
        cal["has_event"] = (cal["event_name_1"] != "None").astype(np.int8)

        # One-hot theo loại sự kiện (4 loại)
        for evt in ["Cultural", "Religious", "Sporting", "National"]:
            cal[f"evt_{evt}"] = (cal["event_type_1"] == evt).astype(np.int8)

        # Số lượng sự kiện trong ngày
        cal["event_count"] = (
            (cal["event_name_1"] != "None").astype(np.int8)
            + (cal["event_name_2"] != "None").astype(np.int8)
        )

        # Khoảng cách đến sự kiện gần nhất (tiền/hậu kỳ ngày lễ)
        is_evt  = cal["has_event"].values
        n       = len(cal)
        evt_idx = np.where(is_evt == 1)[0]
        days_to_next   = np.full(n, 999, dtype=np.int16)
        days_since_last = np.full(n, 999, dtype=np.int16)
        if len(evt_idx) > 0:
            nxt = np.searchsorted(evt_idx, np.arange(n))
            v = nxt < len(evt_idx)
            days_to_next[v] = (evt_idx[nxt[v]] - np.arange(n)[v]).astype(np.int16)
            prev = np.searchsorted(evt_idx, np.arange(n), side="right") - 1
            vp = prev >= 0
            days_since_last[vp] = (np.arange(n)[vp] - evt_idx[prev[vp]]).astype(np.int16)
        cal["days_to_next_event"]   = np.clip(days_to_next, 0, 60).astype(np.int16)
        cal["days_since_last_event"]= np.clip(days_since_last, 0, 60).astype(np.int16)

        self._calendar_cache = cal
        return cal

    # ------------------------------------------------------------------
    # 2. Load Prices
    # ------------------------------------------------------------------
    def load_prices(self) -> pd.DataFrame:
        print("[+] Load sell_prices.csv ...")
        prices = pd.read_csv(self.prices_path)
        prices = reduce_mem_usage(prices, verbose=False)

        # Thống kê giá lịch sử theo item/store
        stats = prices.groupby(["store_id", "item_id"])["sell_price"].agg(
            price_max="max", price_min="min",
            price_mean="mean", price_std="std"
        ).reset_index()
        prices = prices.merge(stats, on=["store_id", "item_id"], how="left")

        # Tỷ lệ discount so với giá cao nhất
        prices["price_discount"]  = (1.0 - prices["sell_price"] / prices["price_max"]).fillna(0).astype(np.float32)
        # Tỷ lệ so với trung bình
        prices["price_norm_mean"] = (prices["sell_price"] / prices["price_mean"]).fillna(1.0).astype(np.float32)
        # Momentum giá (% thay đổi so với tuần trước)
        prices["price_momentum"]  = prices.groupby(["store_id", "item_id"])["sell_price"].pct_change().fillna(0).astype(np.float32)

        return reduce_mem_usage(prices, verbose=False)

    # ------------------------------------------------------------------
    # 3. Tạo dữ liệu SKU-Level (Level 12)
    # ------------------------------------------------------------------
    def create_sku_level_data(
        self,
        start_day: int = 1000,
        end_day: int   = 1913,
        items_subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Melt & merge toàn bộ dữ liệu cấp SKU (Level 12)."""
        print(f"[+] Load sales từ d_{start_day} -> d_{end_day} ...")
        sales = pd.read_csv(self.sales_path)
        if items_subset:
            sales = sales[sales["id"].isin(items_subset)]

        day_cols = [f"d_{i}" for i in range(start_day, end_day + 1)]
        id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        print("[-] Melt Wide -> Long ...")
        df = pd.melt(sales, id_vars=id_cols, value_vars=day_cols,
                     var_name="d", value_name="sales")
        del sales
        gc.collect()
        df = reduce_mem_usage(df, verbose=False)

        cal    = self.load_calendar()
        prices = self.load_prices()

        print("[-] Merge Calendar ...")
        df = df.merge(cal, on="d", how="left")
        print("[-] Merge Prices ...")
        df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

        # Điền thiếu giá
        for col, fill in [("sell_price", 0), ("price_discount", 0),
                          ("price_norm_mean", 1.0), ("price_momentum", 0),
                          ("price_max", 0), ("price_min", 0), ("price_mean", 0), ("price_std", 0)]:
            if col in df.columns:
                df[col] = df[col].fillna(fill).astype(np.float32)

        # Chuyển "d_xxx" -> số nguyên
        df["d_num"] = df["d"].str.replace("d_", "").astype(np.int16)

        print("[-] Tạo Lag & Rolling features (min lag=28, chống leakage) ...")
        df = self._add_lag_rolling(df, group_col="id", target_col="sales")

        df = reduce_mem_usage(df, verbose=True)
        print("[✓] SKU-Level hoàn tất!")
        return df

    # ------------------------------------------------------------------
    # 4. Tổng hợp Store-Level (Level 3)
    # ------------------------------------------------------------------
    def create_store_level_data(self, sku_df: pd.DataFrame) -> pd.DataFrame:
        print("[+] Tổng hợp Store-Level ...")
        agg = sku_df.groupby(["store_id", "date", "d_num"]).agg(
            total_sales   =("sales", "sum"),
            mean_price    =("sell_price", "mean"),
            avg_discount  =("price_discount", "mean"),
            is_weekend    =("is_weekend", "first"),
            sin_wday      =("sin_wday", "first"),
            cos_wday      =("cos_wday", "first"),
            sin_month     =("sin_month", "first"),
            cos_month     =("cos_month", "first"),
            has_event     =("has_event", "first"),
            state_id      =("state_id", "first"),
        ).reset_index()

        cal = self.load_calendar()
        # Thêm SNAP theo bang
        agg = agg.merge(cal[["date", "snap_CA", "snap_TX", "snap_WI"]], on="date", how="left")
        agg["snap_active"] = np.where(
            agg["state_id"] == "CA", agg["snap_CA"],
            np.where(agg["state_id"] == "TX", agg["snap_TX"], agg["snap_WI"])
        ).astype(np.int8)
        agg.drop(columns=["snap_CA", "snap_TX", "snap_WI"], inplace=True)

        # Thêm event features
        evt_cols = ["date", "event_count",
                    "evt_Cultural", "evt_Religious", "evt_Sporting", "evt_National",
                    "days_to_next_event", "days_since_last_event"]
        agg = agg.merge(cal[evt_cols], on="date", how="left")

        agg = agg.sort_values(["store_id", "d_num"]).reset_index(drop=True)
        agg = self._add_lag_rolling(agg, group_col="store_id", target_col="total_sales")

        agg = reduce_mem_usage(agg, verbose=True)
        print("[✓] Store-Level hoàn tất!")
        return agg

    # ------------------------------------------------------------------
    # 5. Helper: Lag & Rolling Features (chống leakage)
    # ------------------------------------------------------------------
    def _add_lag_rolling(self, df: pd.DataFrame,
                          group_col: str, target_col: str) -> pd.DataFrame:
        """
        Tạo lag & rolling features chống rò rỉ dữ liệu (data leakage):
        - Tất cả lag >= 28 (dự báo horizon 28 ngày)
        - Rolling tính trên chuỗi đã shift 28 ngày
        """
        df = df.sort_values([group_col, "d_num"]).reset_index(drop=True)
        grp = df.groupby(group_col)[target_col]

        # Lag features: 28, 35, 42, 56 ngày
        for lag in [28, 35, 42, 56]:
            df[f"sales_lag_{lag}"] = grp.shift(lag).astype(np.float32)

        # Rolling Mean & Std (7 ngày) — trên chuỗi shift-28
        df["sales_rolling_mean_7"] = (
            grp.transform(lambda x: x.shift(28).rolling(7, min_periods=1).mean())
            .astype(np.float32)
        )
        df["sales_rolling_std_7"] = (
            grp.transform(lambda x: x.shift(28).rolling(7, min_periods=1).std())
            .fillna(0).astype(np.float32)
        )
        # Rolling Mean & Std (28 ngày) — trên chuỗi shift-28
        df["sales_rolling_mean_28"] = (
            grp.transform(lambda x: x.shift(28).rolling(28, min_periods=1).mean())
            .astype(np.float32)
        )
        df["sales_rolling_std_28"] = (
            grp.transform(lambda x: x.shift(28).rolling(28, min_periods=1).std())
            .fillna(0).astype(np.float32)
        )
        # EWM (Exponentially Weighted Moving Average) — bổ sung so với gốc
        df["sales_ewm_alpha02"] = (
            grp.transform(lambda x: x.shift(28).ewm(alpha=0.2, min_periods=1).mean())
            .astype(np.float32)
        )
        df["sales_ewm_alpha05"] = (
            grp.transform(lambda x: x.shift(28).ewm(alpha=0.5, min_periods=1).mean())
            .astype(np.float32)
        )

        return df

    # ------------------------------------------------------------------
    # 6. Lưu dữ liệu
    # ------------------------------------------------------------------
    def save_processed_data(self, df: pd.DataFrame, file_name: str,
                            output_dir: str = "data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, file_name)
        print(f"[-] Lưu: {path} ...")
        df.to_parquet(path, compression="snappy", index=False)
        size_mb = os.path.getsize(path) / 1024**2
        print(f"[✓] Đã lưu! Kích thước: {size_mb:.2f} MB")
