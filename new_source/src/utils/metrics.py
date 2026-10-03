"""
Module: metrics.py  (new_source version)
Mục đích: Cung cấp các hàm đo lường độ chính xác chuỗi thời gian:
          RMSE, MAE, WMAPE, sMAPE, MAPE.
"""
import numpy as np
from typing import Dict


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    y_true_safe = np.where(y_true == 0, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def calc_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted MAPE — chỉ số chính của M5 cấp SKU.
    WMAPE = sum|y-yhat| / sum|y| * 100
    """
    total = np.sum(np.abs(y_true))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def calc_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toàn bộ bộ chỉ số đánh giá và trả về dưới dạng dict."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.clip(np.asarray(y_pred).flatten(), 0, None)
    return {
        "RMSE":     calc_rmse(y_true, y_pred),
        "MAE":      calc_mae(y_true, y_pred),
        "WMAPE (%)": calc_wmape(y_true, y_pred),
        "sMAPE (%)": calc_smape(y_true, y_pred),
    }
