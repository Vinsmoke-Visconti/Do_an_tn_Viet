"""
Module: visualizer.py
Mục đích: Cung cấp các hàm trực quan hóa chuyên sâu cho bài toán Time Series Forecasting:
          1. Biểu đồ dự báo đa phân vị (P10 - P50 - P90) với dải bất định (Uncertainty Band).
          2. Biểu đồ Feature Importance từ Variable Selection Network (VSN).
          3. Biểu đồ Heatmap ma trận chú ý thời gian (Temporal Attention Weights).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

# Thiết lập style chuẩn học thuật
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def plot_forecast_with_quantiles(
    dates: np.ndarray,
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    title: str = "TFT Quantile Demand Forecast (Walmart M5)",
    save_path: Optional[str] = "reports/figures/forecast_quantiles.png"
):
    """
    Vẽ biểu đồ so sánh giá trị thực tế và dự báo phân vị (P50 kèm dải P10-P90).
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))

    # Chuyển đổi date nếu cần
    x_axis = np.arange(len(y_true))
    x_labels = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates] if isinstance(dates[0], (pd.Timestamp, str, np.datetime64)) else x_axis

    # 1. Dải bất định [P10, P90]
    ax.fill_between(
        x_axis,
        p10,
        p90,
        color="#3b82f6",
        alpha=0.25,
        label="Khoảng tin cậy 80% (P10 - P90)"
    )

    # 2. Đường dự báo Median P50
    ax.plot(
        x_axis,
        p50,
        color="#1d4ed8",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Dự báo TFT (P50 Median)"
    )

    # 3. Đường thực tế Ground Truth
    ax.plot(
        x_axis,
        y_true,
        color="#ef4444",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Thực tế (Actual Sales)"
    )

    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel("Thời gian (28 ngày Test Horizon)", labelpad=10)
    ax.set_ylabel("Tổng Nhu cầu / Doanh số", labelpad=10)
    
    # Hiển thị tick nhãn ngày cách quãng
    step = max(1, len(x_axis) // 7)
    ax.set_xticks(x_axis[::step])
    ax.set_xticklabels([x_labels[i] for i in x_axis[::step]], rotation=25)

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[✓] Đã lưu biểu đồ Dự báo Phân vị vào: {save_path}")
    plt.close()


def plot_variable_importance(
    past_weights: np.ndarray,
    future_weights: np.ndarray,
    past_cols: List[str],
    future_cols: List[str],
    title: str = "TFT Feature Importance (Variable Selection Network)",
    save_path: Optional[str] = "reports/figures/feature_importance.png"
):
    """
    Vẽ biểu đồ cột thể hiện mức độ quan trọng trung bình của từng biến ngoại sinh và nội sinh từ VSN.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Tính trọng số trung bình theo trục thời gian
    mean_past = np.mean(past_weights, axis=0) if past_weights.ndim > 1 else past_weights
    mean_future = np.mean(future_weights, axis=0) if future_weights.ndim > 1 else future_weights

    # Chuẩn hóa về phần trăm (%)
    mean_past = (mean_past / np.sum(mean_past)) * 100
    mean_future = (mean_future / np.sum(mean_future)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot A: Past Variables ---
    df_past = pd.DataFrame({"Feature": past_cols, "Importance": mean_past}).sort_values("Importance", ascending=True)
    colors_past = sns.color_palette("Blues_r", len(df_past))
    axes[0].barh(df_past["Feature"], df_past["Importance"], color=colors_past, edgecolor="black", alpha=0.85)
    axes[0].set_title("A. Past & Internal Features (Nội sinh & Lịch sử)", fontweight="bold")
    axes[0].set_xlabel("Độ quan trọng (%)")
    for i, v in enumerate(df_past["Importance"]):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=10)

    # --- Plot B: Known-Future Exogenous Variables ---
    df_future = pd.DataFrame({"Feature": future_cols, "Importance": mean_future}).sort_values("Importance", ascending=True)
    colors_future = sns.color_palette("Greens_r", len(df_future))
    axes[1].barh(df_future["Feature"], df_future["Importance"], color=colors_future, edgecolor="black", alpha=0.85)
    axes[1].set_title("B. Known-Future Exogenous Variables (Yếu tố Ngoại sinh)", fontweight="bold")
    axes[1].set_xlabel("Độ quan trọng (%)")
    for i, v in enumerate(df_future["Importance"]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=10)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Đã lưu biểu đồ Feature Importance vào: {save_path}")
    plt.close()


def plot_temporal_attention(
    attention_matrix: np.ndarray,
    seq_len: int = 56,
    horizon: int = 28,
    title: str = "TFT Interpretable Temporal Attention Heatmap",
    save_path: Optional[str] = "reports/figures/temporal_attention.png"
):
    """
    Vẽ Heatmap thể hiện mối quan hệ Attention giữa các bước tương lai (Horizon) và các bước quá khứ (Seq_len).
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # attention_matrix có kích thước [Seq_len + Horizon, Seq_len + Horizon]
    # Ta quan tâm đến việc: Các bước Horizon [seq_len : seq_len + horizon] chú ý vào đâu trong [0 : seq_len]
    future_to_past_attn = attention_matrix[seq_len:, :seq_len] # [Horizon, Seq_len]

    # Chuẩn hóa từng hàng để tổng bằng 1 phục vụ trực quan
    row_sums = future_to_past_attn.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    norm_attn = future_to_past_attn / row_sums

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.heatmap(
        norm_attn,
        cmap="rocket_r",
        cbar_kws={"label": "Mức độ chú ý (Attention Weight)"},
        ax=ax
    )

    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel("Các bước thời gian trong quá khứ (t - 56 đến t)", labelpad=10)
    ax.set_ylabel("Các ngày dự báo trong tương lai (t + 1 đến t + 28)", labelpad=10)

    # Đánh dấu các mốc lag quan trọng trong quá khứ (t-7, t-14, t-28, t-56)
    key_past_lags = [seq_len - 1, seq_len - 7, seq_len - 14, seq_len - 28, 0]
    key_past_labels = ["t-1", "t-7", "t-14", "t-28", "t-56"]
    ax.set_xticks(key_past_lags)
    ax.set_xticklabels(key_past_labels, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[✓] Đã lưu biểu đồ Temporal Attention Heatmap vào: {save_path}")
    plt.close()
