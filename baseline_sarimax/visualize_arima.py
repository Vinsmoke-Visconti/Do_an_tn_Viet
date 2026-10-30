"""
Module: visualize_arima.py
Mục đích: Trực quan hóa kết quả mô hình Baseline ARIMA / SARIMAX.
  - Plot 1: Actual vs Forecast + CI 95% (4 đại diện Store).
  - Plot 2: So sánh WMAPE & RMSE bar chart ngang (10 Stores).
  - Plot 3: Diễn biến WMAPE theo từng bước dự báo (ngày 1..28).

BUG-FIXES so với phiên bản cũ:
  [9]  plot_daily_error_distribution: groupby "d_num" nhưng "date" trong
       df_forecast lưu chuỗi string (%d-%m-%Y), gọi group["date"].iloc[0]
       sẽ lấy string chứ không phải datetime -> tránh dùng để sort/label.
       -> Sửa: thêm cột "step" (1..28) đúng thứ tự d_num thay vì count từ 0.
  [10] plot_arima_forecast_comparison: `x_dates` dùng pd.to_datetime với
       format "%d-%m-%Y" - nếu matplotlib vẽ x_dates là DatetimeIndex thì
       mdates.DayLocator(interval=5) có thể nhảy tick sai khi tháng ngắn.
       -> Sửa: dùng mdates.AutoDateLocator() linh hoạt hơn.
  [11] Slope label trong plot 3 in "+{z[0]:.3f}%" có thể sai dấu khi
       sai số thực sự giảm theo ngày (SARIMAX tốt ở ngắn hạn hơn dài hạn).
       -> Sửa: in giá trị thực z[0] (có thể âm) và đổi label cho đúng nghĩa.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Style ──────────────────────────────────────────────────────────────────────
_STYLE = "seaborn-v0_8-whitegrid"
if _STYLE not in plt.style.available:
    _STYLE = "seaborn-whitegrid" if "seaborn-whitegrid" in plt.style.available else "default"
plt.style.use(_STYLE)

plt.rcParams.update({
    "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor":   "#bbbbbb",
    "axes.linewidth":   0.8,
    "figure.dpi":       150,   # screen preview
})

_COLORS = {
    "actual": "#1f77b4",
    "pred":   "#d62728",
    "ci":     "#ff9896",
    "bar1":   "#2ca02c",
    "bar2":   "#1f77b4",
    "avg":    "#d62728",
    "trend":  "#555555",
}


# ── Plot 1 ─────────────────────────────────────────────────────────────────────

def plot_arima_forecast_comparison(
    store_details: dict,
    output_path: str,
    selected_stores: list = None,
):
    """Actual vs Forecast + CI 95% cho 4 đại diện (2x2 grid)."""
    if selected_stores is None:
        selected_stores = ["CA_1", "TX_1", "WI_1", "NATIONAL_TOTAL"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for idx, sid in enumerate(selected_stores):
        ax = axes[idx]
        if sid not in store_details:
            ax.set_visible(False)
            continue

        res   = store_details[sid]
        df_fc = res["forecast_df"]
        m     = res["metrics"]

        # Chuyển chuỗi ngày về datetime để matplotlib xử lý trục X đúng
        x = pd.to_datetime(df_fc["date"], format="%d-%m-%Y")

        ax.plot(x, df_fc["actual_sales"],
                label="Actual (Ground Truth)",
                color=_COLORS["actual"], lw=2, marker="o", ms=4)
        ax.plot(x, df_fc["forecast_sarima"],
                label="ARIMA/SARIMAX Forecast",
                color=_COLORS["pred"], lw=2, ls="--", marker="s", ms=4)
        ax.fill_between(x,
                        df_fc["ci_lower_95"], df_fc["ci_upper_95"],
                        color=_COLORS["ci"], alpha=0.3,
                        label="95% Confidence Interval")

        # [FIX-10] AutoDateLocator linh hoat hon DayLocator co dinh
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
        fig.autofmt_xdate(rotation=30, ha="right")

        label = "National Aggregate" if sid == "NATIONAL_TOTAL" else f"Store {sid}"
        ax.set_title(
            f"{label}  |  RMSE={m['RMSE']:.1f}  WMAPE={m['WMAPE (%)']:.2f}%",
            fontsize=11, fontweight="bold", pad=6,
        )
        ax.set_xlabel("Date (day-month)", fontsize=9)
        ax.set_ylabel("Total Sales (units)", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.6)

    fig.suptitle(
        "28-Day Forecast: ARIMA/SARIMAX Baseline vs Actual (M5 Walmart)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved: {output_path}")


# ── Plot 2 ─────────────────────────────────────────────────────────────────────

def plot_store_metrics_comparison(metrics_df: pd.DataFrame, output_path: str):
    """Bar chart ngang: WMAPE & RMSE cho 10 Stores."""
    exclude = {"NATIONAL_TOTAL", "STORE_AVERAGE"}
    df = (
        metrics_df[~metrics_df["store_id"].isin(exclude)]
        .copy()
        .sort_values("WMAPE (%)", ascending=True)
        .reset_index(drop=True)
    )

    mu_wmape = df["WMAPE (%)"].mean()
    mu_rmse  = df["RMSE"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ---- WMAPE ----
    bars1 = ax1.barh(df["store_id"], df["WMAPE (%)"],
                     color=_COLORS["bar1"], alpha=0.85, edgecolor="#1a6b1a")
    ax1.axvline(mu_wmape, color=_COLORS["avg"], ls="--", lw=1.8,
                label=f"Average: {mu_wmape:.2f}%")
    for b in bars1:
        w = b.get_width()
        ax1.text(w + 0.15, b.get_y() + b.get_height() / 2,
                 f"{w:.2f}%", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("WMAPE (%) — lower is better", fontsize=10, fontweight="bold")
    ax1.set_title("WMAPE by Store (ARIMA Baseline)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", ls=":", alpha=0.6)

    # ---- RMSE ----
    bars2 = ax2.barh(df["store_id"], df["RMSE"],
                     color=_COLORS["bar2"], alpha=0.85, edgecolor="#123a5e")
    ax2.axvline(mu_rmse, color=_COLORS["avg"], ls="--", lw=1.8,
                label=f"Average: {mu_rmse:.1f}")
    for b in bars2:
        w = b.get_width()
        ax2.text(w + 8, b.get_y() + b.get_height() / 2,
                 f"{w:.1f}", va="center", fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("RMSE (units) — lower is better", fontsize=10, fontweight="bold")
    ax2.set_title("RMSE by Store (ARIMA Baseline)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", ls=":", alpha=0.6)

    fig.suptitle(
        "ARIMA Baseline Performance Across 10 Walmart Stores (M5)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved: {output_path}")


# ── Plot 3 ─────────────────────────────────────────────────────────────────────

def plot_daily_error_distribution(full_forecast_df: pd.DataFrame, output_path: str):
    """WMAPE theo từng bước dự báo (ngày 1 đến 28) gộp chung 10 Stores."""

    # Chỉ lấy 10 Stores thực (bỏ NATIONAL_TOTAL)
    df_s = full_forecast_df[full_forecast_df["store_id"] != "NATIONAL_TOTAL"].copy()

    # [FIX-9] Sắp xếp d_num tăng dần, gán step 1..28 theo thứ tự d_num
    d_nums_sorted = sorted(df_s["d_num"].unique())
    d_to_step = {d: i + 1 for i, d in enumerate(d_nums_sorted)}

    daily = []
    for d in d_nums_sorted:
        grp = df_s[df_s["d_num"] == d]
        y    = grp["actual_sales"].values.astype(float)
        yhat = grp["forecast_sarima"].values.astype(float)
        denom = np.sum(np.abs(y))
        wmape = float(np.sum(np.abs(y - yhat)) / denom * 100) if denom > 0 else 0.0
        mae   = float(np.mean(np.abs(y - yhat)))
        daily.append({
            "step":      d_to_step[d],
            "d_num":     d,
            "WMAPE (%)": wmape,
            "MAE":       mae,
        })

    df_daily = pd.DataFrame(daily).sort_values("step").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    ax.plot(df_daily["step"], df_daily["WMAPE (%)"],
            color=_COLORS["pred"], marker="o", lw=2.2,
            label="Daily WMAPE (%) — aggregated across 10 stores")

    # [FIX-11] Hien thi gia tri thuc cua slope (co the am)
    z   = np.polyfit(df_daily["step"], df_daily["WMAPE (%)"], 1)
    pf  = np.poly1d(z)
    direction = "increase" if z[0] >= 0 else "decrease"
    ax.plot(df_daily["step"], pf(df_daily["step"]),
            color=_COLORS["trend"], ls="--", alpha=0.75,
            label=f"Trend ({direction}: {z[0]:+.3f}%/day)")

    ax.set_xlabel("Forecast Horizon Step (Day 1 → Day 28)", fontsize=11, fontweight="bold")
    ax.set_ylabel("WMAPE (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(1, 29, 2))
    ax.grid(ls=":", alpha=0.6)
    ax.legend(fontsize=10, framealpha=0.9)

    ax.set_title(
        "Error Accumulation across 28-Day Forecast Horizon (ARIMA Baseline, 10 Stores)",
        fontsize=12, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Saved: {output_path}")
