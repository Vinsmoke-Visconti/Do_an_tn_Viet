"""
Module: visualize_arima_pure.py
Mục đích: Trực quan hóa kết quả Baseline ARIMA thuần túy.
  - Plot 1: Actual vs ARIMA Forecast + CI 95% (4 đại diện Store).
  - Plot 2: So sánh WMAPE & RMSE bar chart ngang (10 Stores).
  - Plot 3: WMAPE theo từng bước dự báo Horizon Day 1→28.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Style ──────────────────────────────────────────────────────────────────────
_STYLE = next(
    (s for s in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"] if s in plt.style.available),
    "default",
)
plt.style.use(_STYLE)
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor":  "#bbbbbb",
    "axes.linewidth":  0.8,
})

_C = {
    "actual": "#1f77b4",
    "pred":   "#d62728",
    "ci":     "#ff9896",
    "bar1":   "#2ca02c",
    "bar2":   "#1f77b4",
    "avg":    "#d62728",
    "trend":  "#555555",
}


# ── Plot 1: Forecast vs Actual ─────────────────────────────────────────────────

def plot_forecast_comparison(
    details: dict,
    output_path: str,
    selected: list = None,
):
    """Actual vs Forecast + CI 95% cho 4 đại diện (2×2 grid)."""
    if selected is None:
        selected = ["CA_1", "TX_1", "WI_1", "NATIONAL_TOTAL"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for i, sid in enumerate(selected):
        ax = axes[i]
        if sid not in details:
            ax.set_visible(False)
            continue

        df_fc = details[sid]["forecast_df"]
        m     = details[sid]["metrics"]
        x     = pd.to_datetime(df_fc["date"], format="%d-%m-%Y")

        ax.plot(x, df_fc["actual_sales"],
                label="Actual (Ground Truth)",
                color=_C["actual"], lw=2.2, marker="o", ms=4)
        ax.plot(x, df_fc["forecast_arima"],
                label="ARIMA Forecast",
                color=_C["pred"], lw=2.2, ls="--", marker="s", ms=4)
        ax.fill_between(x,
                        df_fc["ci_lower_95"], df_fc["ci_upper_95"],
                        color=_C["ci"], alpha=0.3,
                        label="95% Confidence Interval")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
        fig.autofmt_xdate(rotation=30, ha="right")

        label = "National Aggregate" if sid == "NATIONAL_TOTAL" else f"Store {sid}"
        ax.set_title(
            f"{label}  |  RMSE={m['RMSE']:.1f}  WMAPE={m['WMAPE (%)']:.2f}%",
            fontsize=11, fontweight="bold", pad=6,
        )
        ax.set_xlabel("Date (dd-mm)", fontsize=9)
        ax.set_ylabel("Total Sales (units)", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(ls=":", alpha=0.6)

    fig.suptitle(
        "28-Day Forecast: Pure ARIMA(7,1,1) Baseline vs Actual (M5 Walmart)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [OK] Saved: {output_path}")


# ── Plot 2: Store metrics comparison ──────────────────────────────────────────

def plot_store_metrics(metrics_df: pd.DataFrame, output_path: str):
    """Bar chart ngang WMAPE & RMSE cho 10 Stores."""
    exclude = {"NATIONAL_TOTAL", "STORE_AVERAGE"}
    df = (
        metrics_df[~metrics_df["series_id"].isin(exclude)]
        .copy()
        .sort_values("WMAPE (%)", ascending=True)
        .reset_index(drop=True)
    )

    mu_wmape = df["WMAPE (%)"].mean()
    mu_rmse  = df["RMSE"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # WMAPE
    bars1 = ax1.barh(df["series_id"], df["WMAPE (%)"],
                     color=_C["bar1"], alpha=0.85, edgecolor="#1a6b1a")
    ax1.axvline(mu_wmape, color=_C["avg"], ls="--", lw=1.8,
                label=f"Average: {mu_wmape:.2f}%")
    for b in bars1:
        w = b.get_width()
        ax1.text(w + 0.15, b.get_y() + b.get_height() / 2,
                 f"{w:.2f}%", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("WMAPE (%) — lower is better", fontsize=10, fontweight="bold")
    ax1.set_title("WMAPE by Store — Pure ARIMA(7,1,1)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="x", ls=":", alpha=0.6)

    # RMSE
    bars2 = ax2.barh(df["series_id"], df["RMSE"],
                     color=_C["bar2"], alpha=0.85, edgecolor="#123a5e")
    ax2.axvline(mu_rmse, color=_C["avg"], ls="--", lw=1.8,
                label=f"Average: {mu_rmse:.1f}")
    for b in bars2:
        w = b.get_width()
        ax2.text(w + 8, b.get_y() + b.get_height() / 2,
                 f"{w:.1f}", va="center", fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("RMSE (units) — lower is better", fontsize=10, fontweight="bold")
    ax2.set_title("RMSE by Store — Pure ARIMA(7,1,1)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", ls=":", alpha=0.6)

    fig.suptitle(
        "Pure ARIMA(7,1,1) Baseline Performance Across 10 Walmart Stores (M5)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [OK] Saved: {output_path}")


# ── Plot 3: Error accumulation over 28-day horizon ────────────────────────────

def plot_horizon_error(full_df: pd.DataFrame, output_path: str):
    """WMAPE theo từng bước dự báo, tổng hợp 10 Stores."""
    df_s = full_df[full_df["series_id"] != "NATIONAL_TOTAL"].copy()

    d_sorted = sorted(df_s["d_num"].unique())
    d_step   = {d: i + 1 for i, d in enumerate(d_sorted)}

    rows = []
    for d in d_sorted:
        g    = df_s[df_s["d_num"] == d]
        y    = g["actual_sales"].values.astype(float)
        yhat = g["forecast_arima"].values.astype(float)
        denom = np.sum(np.abs(y))
        wmape = float(np.sum(np.abs(y - yhat)) / denom * 100) if denom > 0 else 0.0
        rows.append({"step": d_step[d], "WMAPE (%)": wmape, "MAE": np.mean(np.abs(y - yhat))})

    df_h = pd.DataFrame(rows).sort_values("step")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.plot(df_h["step"], df_h["WMAPE (%)"],
            color=_C["pred"], marker="o", lw=2.2,
            label="Daily WMAPE (%) — avg 10 stores")

    z  = np.polyfit(df_h["step"], df_h["WMAPE (%)"], 1)
    pf = np.poly1d(z)
    ax.plot(df_h["step"], pf(df_h["step"]),
            color=_C["trend"], ls="--", alpha=0.75,
            label=f"Trend ({z[0]:+.3f}%/day)")

    ax.set_xlabel("Forecast Horizon Step (Day 1 → Day 28)", fontsize=11, fontweight="bold")
    ax.set_ylabel("WMAPE (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(1, 29, 2))
    ax.legend(fontsize=10)
    ax.grid(ls=":", alpha=0.6)
    ax.set_title(
        "Error Accumulation over 28-Day Horizon — Pure ARIMA(7,1,1) Baseline",
        fontsize=12, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [OK] Saved: {output_path}")

