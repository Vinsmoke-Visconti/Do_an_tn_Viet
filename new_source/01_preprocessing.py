"""
Script: 01_preprocessing.py
Mục đích: Bước 1 — Tiền xử lý dữ liệu M5 FULL (30,490 SKU) và lưu ra Parquet.

Chạy lệnh:
    python 01_preprocessing.py

Kết quả:
    data/processed/m5_sku_level_features.parquet  (~12.6M rows, ~1.5–3 GB)
    data/processed/m5_store_level_features.parquet

Lưu ý:
    Dataset đầy đủ cần ~4–8 GB RAM. Script dùng memory_optimizer nội bộ.
    Nếu hết RAM, hãy tăng start_day (ví dụ 1700) hoặc dùng items_subset.
"""

import os, sys, gc, time
import numpy as np
import pandas as pd

# Đặt thư mục gốc là new_source để import tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data.m5_pipeline import M5DataPipeline


# ─────────────────────────────────────────────────────────────
# Kiểm tra dataset thật – KHÔNG sinh mock data
# ─────────────────────────────────────────────────────────────
def check_and_report_dataset(raw_dir: str) -> int:
    """
    Kiểm tra sự tồn tại của 3 file M5 thật và báo cáo kích thước.
    Raise FileNotFoundError nếu thiếu bất kỳ file nào.
    Trả về số lượng chuỗi SKU trong dataset.
    """
    cal_f   = os.path.join(raw_dir, "calendar.csv")
    price_f = os.path.join(raw_dir, "sell_prices.csv")
    sales_f = os.path.join(raw_dir, "sales_train_validation.csv")

    print("\n[*] Kiểm tra dataset M5 tại:", raw_dir)
    missing = []
    for name, path in [
        ("calendar.csv",                 cal_f),
        ("sell_prices.csv",              price_f),
        ("sales_train_validation.csv",   sales_f),
    ]:
        if not os.path.exists(path):
            missing.append(name)
            print(f"  [✗] {name:45s} — THIẾU!")
        else:
            size_mb = os.path.getsize(path) / 1024**2
            print(f"  [✓] {name:45s} ({size_mb:>7.1f} MB)")

    if missing:
        raise FileNotFoundError(
            f"\n[ERROR] Thiếu {len(missing)} file dữ liệu thật trong:\n  {raw_dir}\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\n\nHãy download M5 dataset từ Kaggle (m5-forecasting-accuracy)\n"
            + "và đặt vào thư mục data/raw/."
        )

    # Đếm số dòng sales mà không load hết vào RAM
    with open(sales_f, "r", encoding="utf-8") as fh:
        n_series = sum(1 for _ in fh) - 1  # trừ header
    print(f"  [i] Tổng SKU series: {n_series:,}")
    return n_series


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("  BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU M5 FULL DATASET")
    print("  (Áp dụng với toàn bộ 30,490 chuỗi SKU × 414 ngày cuối)")
    print("=" * 70)

    raw_dir       = os.path.join(BASE_DIR, "data", "raw")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Xác nhận dataset thật tồn tại (KHÔNG tạo mock)
    n_series = check_and_report_dataset(raw_dir)

    # ----------------------------------------------------------
    # Thông số: lấy 414 ngày cuối (d_1500 → d_1913)
    # ≈ 30,490 SKU × 414 ngày = ~12.6 triệu dòng
    # Tăng start_day lên nếu gặp lỗi OOM (Out of Memory)
    # ----------------------------------------------------------
    START_DAY = 1500
    END_DAY   = 1913

    print(f"\n[i] Sẽ xử lý ngày d_{START_DAY} → d_{END_DAY} "
          f"({END_DAY - START_DAY + 1} ngày × {n_series:,} SKU "
          f"= ~{(END_DAY - START_DAY + 1) * n_series / 1e6:.1f}M dòng)")

    pipeline = M5DataPipeline(data_dir=raw_dir)

    # ==================================================
    # SKU-Level (Level 12 — toàn bộ series)
    # ==================================================
    print("\n" + "─" * 60)
    print("  SKU-LEVEL (Level 12) — Toàn bộ 30,490 chuỗi")
    print("─" * 60)
    t1 = time.time()

    sku_df = pipeline.create_sku_level_data(
        start_day=START_DAY,
        end_day=END_DAY,
        items_subset=None   # None = xử lý TẤT CẢ series
    )
    pipeline.save_processed_data(
        sku_df, "m5_sku_level_features.parquet", output_dir=processed_dir
    )
    t2 = time.time()
    print(f"  ↳ Thời gian xử lý SKU-level: {(t2 - t1) / 60:.1f} phút")

    # ==================================================
    # Store-Level (Level 3 — tổng hợp theo cửa hàng)
    # ==================================================
    print("\n" + "─" * 60)
    print("  STORE-LEVEL (Level 3)")
    print("─" * 60)
    store_df = pipeline.create_store_level_data(sku_df)
    pipeline.save_processed_data(
        store_df, "m5_store_level_features.parquet", output_dir=processed_dir
    )

    # Giải phóng bộ nhớ
    del sku_df, store_df
    gc.collect()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  BƯỚC 1 HOÀN TẤT!")
    sku_path   = os.path.join(processed_dir, "m5_sku_level_features.parquet")
    store_path = os.path.join(processed_dir, "m5_store_level_features.parquet")
    print(f"  SKU-level  : {os.path.getsize(sku_path)   / 1024**2:>7.1f} MB  →  {sku_path}")
    print(f"  Store-level: {os.path.getsize(store_path) / 1024**2:>7.1f} MB  →  {store_path}")
    print(f"  Tổng thời gian: {elapsed / 60:.1f} phút")
    print("=" * 70)


if __name__ == "__main__":
    main()
