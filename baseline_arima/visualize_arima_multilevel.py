"""
Module: visualize_arima_multilevel.py
Purpose: Visualize ARIMA multi-level forecasting results.

Plots:
    1. Level-by-level WMAPE comparison (bar chart)
    2. Forecast vs Actual for L3 stores (sample)
    3. Error distribution by store (L3)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(CURRENT_DIR, "outputs", "results")
FIGURES_DIR  = os.path.join(CURRENT_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    "bar_l1":  "#1565C0",
    "bar_l2":  "#0288D1",
    "bar_l3":  "#00838F",
    "bar_l9":  "#2E7D32",
    "bar_l12": "#F57F17",
    "actual":  "#1A237E",
    "pred":    "#B71C1C",
    "ci":      "#FFCDD2",
    "grid":    "#ECEFF1",
}
LEVEL_COLORS = {
    "L1": COLORS["bar_l1"],
    "L2": COLORS["bar_l2"],
    "L3": COLORS["bar_l3"],
    "L9": COLORS["bar_l9"],
    "L12": COLORS["bar_l12"],
}


# ── Plot 1: Level summary bar chart ───────────────────────────────────────────

def plot_level_summary(output_path: str = None) -> str:
    """Bar chart: WMAPE per level (and per series for L1/L2/L3/L9)."""
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "01_multilevel_wmape_summary.png")

    level_data = {}
    for level in ["L1", "L2", "L3", "L9", "L12"]:
        tag    = level.lower()
        m_path = os.path.join(RESULTS_DIR, f"arima_{tag}_metrics.csv")
        if not os.path.exists(m_path):
            continue
        m_df = pd.read_csv(m_path)
        # Exclude AVERAGE row for per-series display; get average WMAPE
        avg_row = m_df[m_df["series_id"] == "AVERAGE"]
        if not avg_row.empty:
            avg_wmape = float(avg_row["WMAPE (%)"].iloc[0])
        else:
            avg_wmape = float(m_df["WMAPE (%)"].mean())
        level_data[level] = avg_wmape

    if not level_data:
        print("[viz] No metrics files found, skipping plot 1.")
        return ""

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    levels = list(level_data.keys())
    wmapes = [level_data[lv] for lv in levels]
    colors = [LEVEL_COLORS.get(lv, "#607D8B") for lv in levels]

    bars = ax.bar(levels, wmapes, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, wmapes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Aggregation Level", fontsize=12)
    ax.set_ylabel("WMAPE (%)", fontsize=12)
    ax.set_title("ARIMA(7,1,1) Baseline — Average WMAPE by Hierarchy Level\n"
                 "M5 Forecasting Accuracy (Walmart) | Train: 2011-2016 | Test: 28 days",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_facecolor(COLORS["grid"])
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="white")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(wmapes) * 1.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Plot 2: Forecast vs Actual — L3 stores ────────────────────────────────────

def plot_l3_forecast(output_path: str = None, max_stores: int = 6) -> str:
    """Grid of forecast vs actual plots for L3 stores."""
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "02_l3_forecast_vs_actual.png")

    fc_path = os.path.join(RESULTS_DIR, "arima_l3_forecast.csv")
    m_path  = os.path.join(RESULTS_DIR, "arima_l3_metrics.csv")
    if not os.path.exists(fc_path):
        print("[viz] L3 forecast CSV not found, skipping plot 2.")
        return ""

    fc_df = pd.read_csv(fc_path)
    m_df  = pd.read_csv(m_path) if os.path.exists(m_path) else pd.DataFrame()

    stores = sorted(fc_df["series_id"].unique().tolist())[:max_stores]
    ncols  = 2
    nrows  = (len(stores) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "ARIMA(7,1,1) Baseline — Forecast vs Actual (L3: Store Level)\n"
        "Train: d_1 (2011-01-29) → d_1885 | Test: d_1886 → d_1913 (28 days)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, store in enumerate(stores):
        ax   = axes[i]
        df_s = fc_df[fc_df["series_id"] == store].sort_values("d_num")
        x    = np.arange(len(df_s))

        ax.fill_between(x, df_s["ci_lower_95"], df_s["ci_upper_95"],
                        alpha=0.25, color=COLORS["ci"], label="95% CI")
        ax.plot(x, df_s["actual_sales"],   color=COLORS["actual"],
                lw=2, marker="o", ms=4, label="Actual")
        ax.plot(x, df_s["forecast_arima"], color=COLORS["pred"],
                lw=2, ls="--", marker="s", ms=4, label="Forecast")

        wmape_str = ""
        if not m_df.empty:
            row = m_df[m_df["series_id"] == store]
            if not row.empty:
                wmape_str = f"  WMAPE={float(row['WMAPE (%)'].iloc[0]):.2f}%"

        ax.set_title(f"{store}{wmape_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Day in Test Horizon (d_1886 → d_1913)", fontsize=9)
        ax.set_ylabel("Total Sales", fontsize=9)
        ax.set_facecolor(COLORS["grid"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="white")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Hide unused subplots
    for j in range(len(stores), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Plot 3: WMAPE per store (L3 bar) ──────────────────────────────────────────

def plot_l3_metrics(output_path: str = None) -> str:
    """Horizontal bar chart: WMAPE per L3 store."""
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "03_l3_store_wmape.png")

    m_path = os.path.join(RESULTS_DIR, "arima_l3_metrics.csv")
    if not os.path.exists(m_path):
        print("[viz] L3 metrics CSV not found, skipping plot 3.")
        return ""

    m_df = pd.read_csv(m_path)
    m_df = m_df[~m_df["series_id"].isin(["AVERAGE"])].sort_values("WMAPE (%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    bars = ax.barh(m_df["series_id"], m_df["WMAPE (%)"],
                   color=COLORS["bar_l3"], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, m_df["WMAPE (%)"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("WMAPE (%)", fontsize=11)
    ax.set_title("ARIMA(7,1,1) — WMAPE per Store (L3)\n"
                 "Train: 2011-01-29 → 2016-03-27 | Test: 28 days",
                 fontsize=12, fontweight="bold")
    ax.set_facecolor(COLORS["grid"])
    ax.xaxis.grid(True, linestyle="--", alpha=0.7, color="white")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Entry ──────────────────────────────────────────────────────────────────────

def generate_all():
    """Generate all visualization plots."""
    print("[viz] Generating multi-level ARIMA visualizations ...")
    saved = []

    p1 = plot_level_summary()
    if p1:
        print(f"[OK] Plot 1: {p1}")
        saved.append(p1)

    p2 = plot_l3_forecast()
    if p2:
        print(f"[OK] Plot 2: {p2}")
        saved.append(p2)

    p3 = plot_l3_metrics()
    if p3:
        print(f"[OK] Plot 3: {p3}")
        saved.append(p3)

    if not saved:
        print("[viz] No plots generated (no metrics files found).")
    return saved


if __name__ == "__main__":
    generate_all()

