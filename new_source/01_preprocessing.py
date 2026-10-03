"""
Script: 01_preprocessing.py
Mục đích: Bước 1 — Tiền xử lý dữ liệu M5 và lưu ra Parquet.

Chạy lệnh:
    python 01_preprocessing.py

Kết quả:
    data/processed/m5_sku_level_features.parquet
    data/processed/m5_store_level_features.parquet
"""

import os, sys
import numpy as np
import pandas as pd

# Đặt thư mục gốc là new_source để import tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data.m5_pipeline import M5DataPipeline


# ─────────────────────────────────────────────
# Sinh Mock Data nếu chưa có file raw
# ─────────────────────────────────────────────
def generate_mock_if_missing(raw_dir: str):
    cal_f   = os.path.join(raw_dir, "calendar.csv")
    price_f = os.path.join(raw_dir, "sell_prices.csv")
    sales_f = os.path.join(raw_dir, "sales_train_validation.csv")

    if os.path.exists(cal_f) and os.path.exists(price_f) and os.path.exists(sales_f):
        return  # Đã có — không cần tạo mock

    print("[!] Không tìm thấy dữ liệu raw. Đang tạo MOCK DATA nhỏ để test pipeline ...")
    os.makedirs(raw_dir, exist_ok=True)

    # --- Calendar (1913 ngày) ---
    dates = pd.date_range("2011-01-29", periods=1913, freq="D")
    rng   = np.random.default_rng(42)
    cal_df = pd.DataFrame({
        "date":         dates,
        "wm_yr_wk":     11101 + (np.arange(1913) // 7),
        "weekday":      dates.day_name(),
        "wday":         (dates.dayofweek + 2) % 7 + 1,
        "month":        dates.month,
        "year":         dates.year,
        "d":            [f"d_{i+1}" for i in range(1913)],
        "event_name_1": rng.choice(["SuperBowl", "Easter", "Christmas", np.nan],
                                   size=1913, p=[0.01, 0.01, 0.01, 0.97]),
        "event_type_1": rng.choice(["Sporting", "Religious", "Cultural", np.nan],
                                   size=1913, p=[0.01, 0.01, 0.01, 0.97]),
        "event_name_2": [np.nan] * 1913,
        "event_type_2": [np.nan] * 1913,
        "snap_CA": rng.integers(0, 2, size=1913),
        "snap_TX": rng.integers(0, 2, size=1913),
        "snap_WI": rng.integers(0, 2, size=1913),
    })
    cal_df.to_csv(cal_f, index=False)

    # --- Sales (100 series: 50 items × 2 stores) ---
    num_items, stores = 50, ["CA_1", "TX_1"]
    item_ids = [f"FOODS_3_{i:03d}" for i in range(1, num_items + 1)]
    rows = []
    for store in stores:
        state = store.split("_")[0]
        for item in item_ids:
            lam = rng.uniform(0.3, 5.0)
            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item, "dept_id": "FOODS_3",
                "cat_id": "FOODS", "store_id": store, "state_id": state
            }
            sales_series = rng.poisson(lam=lam, size=1913)
            for di in range(1, 1914):
                row[f"d_{di}"] = int(sales_series[di - 1])
            rows.append(row)
    pd.DataFrame(rows).to_csv(sales_f, index=False)

    # --- Prices ---
    wm_weeks = cal_df["wm_yr_wk"].unique()
    pr = []
    for store in stores:
        for item in item_ids:
            base = rng.uniform(1.5, 12.0)
            for wk in wm_weeks:
                pr.append({"store_id": store, "item_id": item, "wm_yr_wk": int(wk),
                           "sell_price": round(base * rng.uniform(0.85, 1.05), 2)})
    pd.DataFrame(pr).to_csv(price_f, index=False)
    print("[✓] Mock data đã tạo xong!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU M5 (TOP-1 PIPELINE)")
    print("=" * 70)

    raw_dir       = os.path.join(BASE_DIR, "data", "raw")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")

    generate_mock_if_missing(raw_dir)

    pipeline = M5DataPipeline(data_dir=raw_dir)

    # SKU-Level (Level 12) — lấy từ ngày 1500 để giảm RAM
    print("\n--- SKU-LEVEL (LEVEL 12) ---")
    sku_df = pipeline.create_sku_level_data(start_day=1500, end_day=1913)
    pipeline.save_processed_data(sku_df, "m5_sku_level_features.parquet", output_dir=processed_dir)

    # Store-Level (Level 3)
    print("\n--- STORE-LEVEL (LEVEL 3) ---")
    store_df = pipeline.create_store_level_data(sku_df)
    pipeline.save_processed_data(store_df, "m5_store_level_features.parquet", output_dir=processed_dir)

    print("\n" + "=" * 70)
    print("  BƯỚC 1 HOÀN TẤT!")
    print(f"  SKU-level  : {sku_df.shape[0]:,} rows × {sku_df.shape[1]} features")
    print(f"  Store-level: {store_df.shape[0]:,} rows × {store_df.shape[1]} features")
    print(f"  Lưu tại    : {processed_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
