"""
Script: 02_train_and_forecast.py
Mục đích: Bước 2 — Huấn luyện LightGBM (Tweedie) và thực hiện dự báo
          đệ quy 28 ngày theo phương pháp Top 1 M5 Competition (YJ_STU).

Pipeline:
  1. Load dữ liệu SKU-level đã xử lý từ Bước 1.
  2. Chia tập Train / Val / Test theo thời gian (Temporal Split).
  3. Huấn luyện LightGBM per-Store (10 mô hình riêng biệt cho 10 stores).
  4. Thực hiện dự báo đệ quy 28 ngày (RecursiveForecaster).
  5. Đánh giá kết quả cấp SKU-Level và Store-Level.
  6. Phân tích Feature Importance (Gain) và lưu toàn bộ kết quả.

Chạy lệnh:
    python 02_train_and_forecast.py

Kết quả sinh ra:
    outputs/results/forecast_results.parquet
    outputs/results/feature_importance_gain.csv
    outputs/models/lgb_tweedie_<store_id>.pkl
    outputs/figures/*.png
"""

import os, sys, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.models.lightgbm_tweedie   import LightGBMTweedie
from src.models.recursive_forecaster import RecursiveForecaster
from src.utils.metrics             import evaluate_forecast
from src.utils.hierarchical_evaluation import evaluate_hierarchical
from src.utils.visualizer import (
    plot_28day_forecast,
    plot_feature_importance,
    plot_daily_wmape_heatmap,
    plot_feature_group_pie,
)

# ─────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIG_DIR     = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR  = os.path.join(OUTPUT_DIR, "results")
HORIZON     = 28

for d in [FIG_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Helper: chuẩn bị dữ liệu
# ─────────────────────────────────────────────
EXCLUDE_COLS = {"id", "date", "d", "d_num", "sales",
                "event_name_1", "event_name_2",
                "event_type_1", "event_type_2",
                "weekday", "wm_yr_wk"}

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def prepare_dataset(df: pd.DataFrame):
    """Chia Train/Val/Test và chuyển đổi kiểu dữ liệu categorical."""
    # Chuyển object → category (LightGBM yêu cầu)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c in df.columns]

    max_d = df["d_num"].max()
    test_start  = max_d - HORIZON + 1
    val_start   = test_start - HORIZON

    train_df = df[df["d_num"] < val_start].copy()
    val_df   = df[(df["d_num"] >= val_start) & (df["d_num"] < test_start)].copy()
    test_df  = df[df["d_num"] >= test_start].copy()

    print(f"  Train: d_num < {val_start}  ({len(train_df):,} rows)")
    print(f"  Val  : [{val_start}, {test_start}) ({len(val_df):,} rows)")
    print(f"  Test : d_num >= {test_start}  ({len(test_df):,} rows)")

    return train_df, val_df, test_df, feature_cols


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BƯỚC 2: HUẤN LUYỆN LIGHTGBM TOP-1 & DỰ BÁO 28 NGÀY ĐỆ QUY")
    print("=" * 70)

    # 1. Load SKU-level data
    sku_path = os.path.join(DATA_DIR, "m5_sku_level_features.parquet")
    if not os.path.exists(sku_path):
        print(f"[!] Không tìm thấy {sku_path}. Vui lòng chạy 01_preprocessing.py trước!")
        return
    print(f"\n[1] Đang load dữ liệu: {sku_path}")
    df = pd.read_parquet(sku_path)
    print(f"    Shape: {df.shape}")

    # 2. Chia tập
    print("\n[2] Chia tập Train / Val / Test ...")
    train_df, val_df, test_df, feature_cols = prepare_dataset(df)
    cat_in_features = [c for c in CAT_COLS if c in feature_cols]

    # 3. Huấn luyện per-Store LightGBM
    print(f"\n[3] Huấn luyện LightGBM cho từng Store ({df['store_id'].nunique()} stores) ...")
    stores = sorted(df["store_id"].unique())
    models = {}
    all_imp_dfs = []

    for store in stores:
        print(f"\n  --- Store: {store} ---")
        tr  = train_df[train_df["store_id"] == store]
        vl  = val_df[val_df["store_id"] == store]

        if len(tr) == 0 or len(vl) == 0:
            print(f"  [!] Không đủ dữ liệu cho {store}, bỏ qua.")
            continue

        X_tr, y_tr = tr[feature_cols], tr["sales"]
        X_vl, y_vl = vl[feature_cols], vl["sales"]

        mdl = LightGBMTweedie(store_id=store)
        mdl.train(X_tr, y_tr, X_vl, y_vl,
                  categorical_features=cat_in_features,
                  num_boost_round=1500, early_stopping_rounds=50)
        mdl.save(output_dir=MODEL_DIR)
        models[store] = mdl

        imp = mdl.get_feature_importance("gain")
        imp["store_id"] = store
        all_imp_dfs.append(imp)

    # 4. Dự báo đệ quy 28 ngày
    print("\n[4] Thực hiện dự báo đệ quy 28 ngày ...")
    all_forecast_dfs = []

    for store in stores:
        if store not in models:
            continue
        mdl  = models[store]
        hist = df[df["store_id"] == store].copy()
        fut  = test_df[test_df["store_id"] == store].copy()

        # Dự báo đệ quy cho từng SKU trong cửa hàng
        forecaster = RecursiveForecaster(model=mdl, horizon=HORIZON)
        result_df  = forecaster.forecast(
            history_df=hist,
            future_features_df=fut,
            feature_cols=feature_cols,
            target_col="sales",
            id_col="id",
        )
        result_df["store_id"] = store
        all_forecast_dfs.append(result_df)
        print(f"  [✓] {store}: {len(result_df)} rows dự báo")

    forecast_df = pd.concat(all_forecast_dfs, ignore_index=True)

    # 5. Đánh giá kết quả
    print("\n[5] Đánh giá kết quả ...")
    y_true = forecast_df["sales"].values
    y_pred = forecast_df["pred_sales"].values

    metrics_sku = evaluate_forecast(y_true, y_pred)
    print("\n  ┌─ KẾT QUẢ CẤP ĐỘ SKU (LEVEL 12) ─────────────────────┐")
    for k, v in metrics_sku.items():
        print(f"  │  {k:<15}: {v:>8.3f}")
    print("  └───────────────────────────────────────────────────────┘")

    # Đánh giá Store-Level (tổng hợp từ kết quả SKU)
    store_true = forecast_df.groupby(["store_id", "d_num"])["sales"].sum().values
    store_pred = forecast_df.groupby(["store_id", "d_num"])["pred_sales"].sum().values
    metrics_store = evaluate_forecast(store_true, store_pred)
    print("\n  ┌─ KẾT QUẢ CẤP ĐỘ STORE (LEVEL 3) ────────────────────┐")
    for k, v in metrics_store.items():
        print(f"  │  {k:<15}: {v:>8.3f}")
    print("  └───────────────────────────────────────────────────────┘")

    # Lưu kết quả
    result_path = os.path.join(RESULT_DIR, "forecast_results.parquet")
    forecast_df.to_parquet(result_path, index=False, compression="snappy")
    print(f"\n  [✓] Kết quả dự báo đã lưu: {result_path}")

    metrics_all = {"SKU-Level": metrics_sku, "Store-Level": metrics_store}
    with open(os.path.join(RESULT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

    # 6. Feature Importance tổng hợp
    if all_imp_dfs:
        print("\n[6] Phân tích Feature Importance ...")
        imp_agg = (
            pd.concat(all_imp_dfs, ignore_index=True)
            .groupby("feature")["importance"].sum()
            .reset_index()
            .sort_values("importance", ascending=False)
        )
        imp_path = os.path.join(RESULT_DIR, "feature_importance_gain.csv")
        imp_agg.to_csv(imp_path, index=False)
        print(f"  [✓] Feature Importance lưu tại: {imp_path}")

        print("\n  TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (GAIN):")
        print(imp_agg.head(15).to_string(index=False))

        # Phân tích đóng góp theo nhóm
        total = imp_agg["importance"].sum()
        groups = {
            "Lags & Rolling Stats": imp_agg[imp_agg["feature"].str.contains(
                "lag|rolling|ewm", regex=True)]["importance"].sum() / total * 100,
            "Giá (Price)": imp_agg[imp_agg["feature"].str.contains(
                "price|discount|momentum", regex=True)]["importance"].sum() / total * 100,
            "Định danh (ID)": imp_agg[imp_agg["feature"].str.contains(
                "item_id|store_id|dept_id|cat_id|state_id", regex=True)]["importance"].sum() / total * 100,
            "Lịch (Calendar)": imp_agg[imp_agg["feature"].str.contains(
                "wday|month|day|week|sin|cos|weekend", regex=True)]["importance"].sum() / total * 100,
            "Ngày lễ (Events)": imp_agg[imp_agg["feature"].str.contains(
                "event|snap|holiday|evt_", regex=True)]["importance"].sum() / total * 100,
        }
        print("\n  ĐÓNG GÓP THEO NHÓM ĐẶC TRƯNG:")
        for g, v in groups.items():
            print(f"    {g:<30}: {v:>5.1f}%")

    # 7. Vẽ biểu đồ
    print("\n[7] Vẽ biểu đồ ...")

    # Biểu đồ dự báo 28 ngày cho 1 store đại diện
    sample_store = stores[0] if stores else None
    if sample_store:
        sf = forecast_df[forecast_df["store_id"] == sample_store]
        agg_sf = sf.groupby("d_num")[["sales", "pred_sales"]].sum().reset_index()
        plot_28day_forecast(
            dates=agg_sf["d_num"].values,
            y_true=agg_sf["sales"].values,
            y_pred=agg_sf["pred_sales"].values,
            title=f"LightGBM Dự báo 28 ngày – Store {sample_store}",
            store_id=sample_store,
            save_path=os.path.join(FIG_DIR, f"forecast_28day_{sample_store}.png"),
        )

    # Feature Importance chart
    if all_imp_dfs:
        plot_feature_importance(
            imp_agg, top_n=20,
            save_path=os.path.join(FIG_DIR, "feature_importance_gain.png")
        )

    # Pie chart nhóm đặc trưng
    if all_imp_dfs:
        plot_feature_group_pie(
            groups,
            save_path=os.path.join(FIG_DIR, "feature_group_pie.png")
        )

    # WMAPE Heatmap 28 ngày
    pivot_true = forecast_df.pivot_table(index="id", columns="d_num", values="sales", aggfunc="first")
    pivot_pred = forecast_df.pivot_table(index="id", columns="d_num", values="pred_sales", aggfunc="first")
    if pivot_true.shape[1] >= HORIZON:
        plot_daily_wmape_heatmap(
            pivot_true.values[:, :HORIZON],
            pivot_pred.values[:, :HORIZON],
            save_path=os.path.join(FIG_DIR, "daily_wmape_heatmap.png"),
        )

    print("\n" + "=" * 70)
    print("  BƯỚC 2 HOÀN TẤT!")
    print(f"  Kết quả lưu tại: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
