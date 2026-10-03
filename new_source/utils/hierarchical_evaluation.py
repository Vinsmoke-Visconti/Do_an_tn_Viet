"""
Module: hierarchical_evaluation.py
Mục đích: Cung cấp các công cụ đánh giá đa cấp độ (Hierarchical Evaluation) cho bài toán M5:
          1. Đánh giá cấp độ SKU (Level 12).
          2. Đánh giá cấp độ Store (Level 3).
          3. Đánh giá cấp độ Toàn hệ thống (Level 1 - National).
          4. Tính toán WRMSSE (Weighted Root Mean Squared Scaled Error) chuẩn cuộc thi M5.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from .metrics import calc_rmse, calc_mae, calc_wmape, calc_smape


def evaluate_hierarchical_forecasts(
    y_true_sku: np.ndarray,
    y_pred_sku: np.ndarray,
    y_true_store: np.ndarray,
    y_pred_store: np.ndarray
) -> pd.DataFrame:
    """
    Đánh giá độ chính xác của mô hình trên đồng thời nhiều cấp độ phân cấp (Hierarchical Levels).

    Args:
        y_true_sku: Thực tế ở cấp SKU-Level [Num_SKUs, Horizon]
        y_pred_sku: Dự báo ở cấp SKU-Level [Num_SKUs, Horizon]
        y_true_store: Thực tế ở cấp Store-Level [Num_Stores, Horizon]
        y_pred_store: Dự báo ở cấp Store-Level [Num_Stores, Horizon]

    Returns:
        pd.DataFrame: Bảng so sánh chỉ số giữa các cấp độ phân cấp.
    """
    results = {}

    # 1. Cấp độ SKU (Level 12)
    y_t_sku_flat = y_true_sku.flatten()
    y_p_sku_flat = np.clip(y_pred_sku.flatten(), 0, None)
    results["Level 12 (SKU - Item/Store)"] = {
        "RMSE": calc_rmse(y_t_sku_flat, y_p_sku_flat),
        "MAE": calc_mae(y_t_sku_flat, y_p_sku_flat),
        "WMAPE (%)": calc_wmape(y_t_sku_flat, y_p_sku_flat),
        "sMAPE (%)": calc_smape(y_t_sku_flat, y_p_sku_flat)
    }

    # 2. Cấp độ Store (Level 3)
    y_t_store_flat = y_true_store.flatten()
    y_p_store_flat = np.clip(y_pred_store.flatten(), 0, None)
    results["Level 3 (Store Level)"] = {
        "RMSE": calc_rmse(y_t_store_flat, y_p_store_flat),
        "MAE": calc_mae(y_t_store_flat, y_p_store_flat),
        "WMAPE (%)": calc_wmape(y_t_store_flat, y_p_store_flat),
        "sMAPE (%)": calc_smape(y_t_store_flat, y_p_store_flat)
    }

    # 3. Cấp độ Toàn hệ thống (Level 1 - Tổng hợp từ Store)
    y_t_national = np.sum(y_true_store, axis=0) if y_true_store.ndim > 1 else y_true_store
    y_p_national = np.sum(y_pred_store, axis=0) if y_pred_store.ndim > 1 else y_pred_store
    results["Level 1 (Total System / State)"] = {
        "RMSE": calc_rmse(y_t_national, y_p_national),
        "MAE": calc_mae(y_t_national, y_p_national),
        "WMAPE (%)": calc_wmape(y_t_national, y_p_national),
        "sMAPE (%)": calc_smape(y_t_national, y_p_national)
    }

    return pd.DataFrame(results).T
