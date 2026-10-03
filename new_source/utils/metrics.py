"""
Module: metrics.py
Mục đích: Cung cấp các độ đo đánh giá độ chính xác chuẩn cho bài toán dự báo chuỗi thời gian bán lẻ:
          RMSE, MAE, MAPE, WMAPE (Weighted Mean Absolute Percentage Error), sMAPE.
"""

import numpy as np
from typing import Dict, Union


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    """Mean Absolute Percentage Error (MAPE) - Thêm epsilon để tránh chia cho 0"""
    y_true_safe = np.where(y_true == 0, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def calc_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE):
    WMAPE = (Tổng |y_true - y_pred|) / (Tổng y_true) * 100%
    Đây là độ đo chuẩn và công bằng nhất trong ngành bán lẻ/tồn kho,
    loại bỏ hoàn toàn vấn đề chia cho 0 khi nhu cầu ngắt quãng (Intermittent Demand).
    """
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / total_actual) * 100)


def calc_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính toán và trả về toàn bộ bộ chỉ số đánh giá.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Clip dự báo không âm cho bài toán nhu cầu tồn kho
    y_pred = np.clip(y_pred, 0, None)

    return {
        "RMSE": calc_rmse(y_true, y_pred),
        "MAE": calc_mae(y_true, y_pred),
        "WMAPE (%)": calc_wmape(y_true, y_pred),
        "sMAPE (%)": calc_smape(y_true, y_pred)
    }
