"""
Module: visualizer.py  (new_source)
Mục đích: Trực quan hóa kết quả LightGBM Top-1 M5:
  1. Biểu đồ dự báo 28 ngày so với thực tế (với dải ±std)
  2. Feature Importance (Gain) theo nhóm
  3. Heatmap WMAPE 28 ngày
  4. So sánh LightGBM vs Baseline (ARIMA/LSTM/TFT)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Optional, Dict

# Style học thuật
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120,
})

_FIG_DIR = "outputs/figures"


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else _FIG_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Biểu đồ dự báo 28 ngày
# -----------------------------------------------------------------------
def plot_28day_forecast(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "LightGBM – Dự báo 28 ngày vs Thực tế",
    store_id: str = "ALL",
    save_path: Optional[str] = None,
):
    """Vẽ đường dự báo 28 ngày so với thực tế."""
    save_path = save_path or os.path.join(_FIG_DIR, f"forecast_28day_{store_id}.png")
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(y_true))

    ax.plot(x, y_true, color="#ef4444", lw=2.2, linestyle="--", marker="s",
            markersize=4, label="Thực tế (Actual)")
    ax.plot(x, y_pred, color="#1d4ed8", lw=2.5, marker="o",
            markersize=4, label="Dự báo LightGBM")

    # Dải sai số dự báo ± std cục bộ
    residual = np.abs(y_true - y_pred)
    sigma = residual.std()
    ax.fill_between(x, y_pred - sigma, y_pred + sigma,
                    alpha=0.18, color="#1d4ed8", label=f"±1σ Band ({sigma:.2f})")

    step = max(1, len(x) // 7)
    ax.set_xticks(x[::step])
    if dates is not None and len(dates) > 0:
        lbl = [pd.to_datetime(d).strftime("%d/%m") for d in dates[::step]]
        ax.set_xticklabels(lbl, rotation=20)

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Ngày trong Horizon 28")
    ax.set_ylabel("Doanh số (Sales Units)")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Fig] Đã lưu: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# 2. Feature Importance (LightGBM Gain)
# -----------------------------------------------------------------------
def plot_feature_importance(
    imp_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "LightGBM Feature Importance (Gain) – Top 1 M5",
    save_path: Optional[str] = None,
):
    """
    imp_df phải có cột 'feature' và 'importance'.
    """
    save_path = save_path or os.path.join(_FIG_DIR, "lgb_feature_importance.png")
    _ensure_dir(save_path)

    df = imp_df.head(top_n).sort_values("importance")
    colors = sns.color_palette("RdYlGn", len(df))

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
    bars = ax.barh(df["feature"], df["importance"], color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="{:.0f}", padding=3, fontsize=9)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score (Gain)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Fig] Đã lưu: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# 3. Heatmap WMAPE theo ngày dự báo (t+1 ... t+28)
# -----------------------------------------------------------------------
def plot_daily_wmape_heatmap(
    y_true_matrix: np.ndarray,
    y_pred_matrix: np.ndarray,
    title: str = "WMAPE từng ngày trong Horizon 28 (LightGBM Top-1)",
    save_path: Optional[str] = None,
):
    """
    y_true_matrix, y_pred_matrix: shape [n_series, 28].
    """
    save_path = save_path or os.path.join(_FIG_DIR, "daily_wmape_heatmap.png")
    _ensure_dir(save_path)

    wmape_per_day = []
    for day in range(y_true_matrix.shape[1]):
        yt = y_true_matrix[:, day]
        yp = np.clip(y_pred_matrix[:, day], 0, None)
        total = np.sum(np.abs(yt))
        wmape = float(np.sum(np.abs(yt - yp)) / total * 100) if total > 0 else 0.0
        wmape_per_day.append(wmape)

    # Reshape thành 4 tuần × 7 ngày để dễ đọc
    arr = np.array(wmape_per_day[:28]).reshape(4, 7)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(arr, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=[f"N{d+1}" for d in range(7)],
                yticklabels=[f"Tuần {w+1}" for w in range(4)],
                cbar_kws={"label": "WMAPE (%)"}, ax=ax)
    ax.set_title(title, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Fig] Đã lưu: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# 4. Biểu đồ cột so sánh các mô hình (Model Comparison)
# -----------------------------------------------------------------------
def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "WMAPE (%)",
    title: str = "So sánh WMAPE các mô hình – M5 Forecasting",
    save_path: Optional[str] = None,
):
    """
    results = {
        "ARIMA":   {"RMSE": ..., "WMAPE (%)": ...},
        "LSTM":    {...},
        "LightGBM": {...},
        "TFT":     {...},
    }
    """
    save_path = save_path or os.path.join(_FIG_DIR, "model_comparison.png")
    _ensure_dir(save_path)

    models  = list(results.keys())
    values  = [results[m].get(metric, 0) for m in models]
    colors  = sns.color_palette("tab10", len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, values, color=colors, edgecolor="black", width=0.55, alpha=0.88)
    ax.bar_label(bars, fmt="{:.2f}", padding=4, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Fig] Đã lưu: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------
# 5. Phân tích Feature Importance theo nhóm (Pie chart)
# -----------------------------------------------------------------------
def plot_feature_group_pie(
    group_contributions: Dict[str, float],
    title: str = "Đóng góp của nhóm đặc trưng (%)",
    save_path: Optional[str] = None,
):
    save_path = save_path or os.path.join(_FIG_DIR, "feature_group_pie.png")
    _ensure_dir(save_path)

    labels = list(group_contributions.keys())
    sizes  = list(group_contributions.values())
    colors = sns.color_palette("Set2", len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title(title, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Fig] Đã lưu: {save_path}")
    plt.close(fig)
