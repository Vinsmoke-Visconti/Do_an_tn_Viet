"""
Module: hierarchical_evaluation.py  (new_source)
Mục đích: Đánh giá dự báo trên nhiều cấp độ phân cấp (Hierarchical Levels) theo chuẩn M5:
          Level 12 (SKU), Level 3 (Store), Level 1 (Toàn hệ thống).
"""
import numpy as np
import pandas as pd
from typing import Dict
from .metrics import calc_rmse, calc_mae, calc_wmape, calc_smape


def evaluate_hierarchical(
    y_true_sku:   np.ndarray,
    y_pred_sku:   np.ndarray,
    y_true_store: np.ndarray,
    y_pred_store: np.ndarray,
) -> pd.DataFrame:
    """
    So sánh độ chính xác dự báo ở 3 cấp độ phân cấp.

    Args:
        y_true_sku   : Doanh số thực tế cấp SKU   (1D hoặc 2D)
        y_pred_sku   : Dự báo cấp SKU
        y_true_store : Doanh số thực tế cấp Store  (1D hoặc 2D)
        y_pred_store : Dự báo cấp Store

    Returns:
        pd.DataFrame: Bảng tổng hợp chỉ số theo từng cấp.
    """
    def _flat(arr):
        return np.clip(np.asarray(arr).flatten(), 0, None)

    results = {}

    # Level 12 — SKU/Item/Store
    yt, yp = _flat(y_true_sku), _flat(y_pred_sku)
    results["Level 12 (SKU)"] = {
        "RMSE": calc_rmse(yt, yp), "MAE": calc_mae(yt, yp),
        "WMAPE (%)": calc_wmape(yt, yp), "sMAPE (%)": calc_smape(yt, yp)
    }

    # Level 3 — Store / State
    yt, yp = _flat(y_true_store), _flat(y_pred_store)
    results["Level 3 (Store)"] = {
        "RMSE": calc_rmse(yt, yp), "MAE": calc_mae(yt, yp),
        "WMAPE (%)": calc_wmape(yt, yp), "sMAPE (%)": calc_smape(yt, yp)
    }

    # Level 1 — National (tổng hợp từ Store)
    if np.asarray(y_true_store).ndim > 1:
        yt_nat = np.sum(y_true_store, axis=0)
        yp_nat = np.sum(y_pred_store, axis=0)
    else:
        yt_nat = np.asarray(y_true_store)
        yp_nat = np.asarray(y_pred_store)
    yt_nat, yp_nat = _flat(yt_nat), _flat(yp_nat)
    results["Level 1 (National)"] = {
        "RMSE": calc_rmse(yt_nat, yp_nat), "MAE": calc_mae(yt_nat, yp_nat),
        "WMAPE (%)": calc_wmape(yt_nat, yp_nat), "sMAPE (%)": calc_smape(yt_nat, yp_nat)
    }

    df = pd.DataFrame(results).T
    df.index.name = "Level"
    return df.round(3)
