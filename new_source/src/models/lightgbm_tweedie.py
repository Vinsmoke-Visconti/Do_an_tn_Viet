"""
Module: lightgbm_tweedie.py  (new_source)
Mục đích: Class LightGBM với Tweedie Loss theo phương pháp đội Top 1 M5 (YJ_STU).
          Hỗ trợ:
            - Huấn luyện với Early Stopping
            - Dự báo đơn điểm (Direct)
            - Feature Importance (Split & Gain)
            - Lưu / Load mô hình
"""
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Optional, Dict, Any


class LightGBMTweedie:
    """
    LightGBM với Tweedie Loss — giải pháp cốt lõi của Top 1 M5 Competition.

    Tại sao Tweedie?
        - Dữ liệu bán lẻ có rất nhiều ngày nhu cầu = 0 (Intermittent Demand).
        - Tweedie (thuộc họ phân phối xác suất với tham số variance_power)
          xử lý tốt hơn MSE/MAE khi dữ liệu có phân phối lệch phải nặng.
        - variance_power=1.1 phù hợp cho demand thưa (sparse); =1.5 cho
          demand thưa nhiều hơn.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective":               "tweedie",
        "tweedie_variance_power":  1.1,
        "metric":                  "rmse",
        "boosting_type":           "gbdt",
        "learning_rate":           0.05,
        "num_leaves":              127,
        "min_data_in_leaf":        50,
        "feature_fraction":        0.8,
        "bagging_fraction":        0.8,
        "bagging_freq":            1,
        "seed":                    42,
        "verbose":                 -1,
        "n_jobs":                  -1,
    }

    def __init__(self, params: Optional[Dict] = None, store_id: str = "ALL"):
        self.params   = params if params else dict(self.DEFAULT_PARAMS)
        self.store_id = store_id
        self.model: Optional[lgb.Booster] = None

    # ------------------------------------------------------------------
    # Huấn luyện
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series,
        categorical_features: List[str] = "auto",
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 50,
    ) -> None:
        print(f"[LGB:{self.store_id}] Chuẩn bị dataset ...")
        dtrain = lgb.Dataset(X_train, label=y_train,
                             categorical_feature=categorical_features,
                             free_raw_data=False)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                             categorical_feature=categorical_features,
                             free_raw_data=False)

        print(f"[LGB:{self.store_id}] Bắt đầu train (objective={self.params['objective']}) ...")
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )
        print(f"[LGB:{self.store_id}] Hoàn tất. Best iter: {self.model.best_iteration}")

    # ------------------------------------------------------------------
    # Dự báo trực tiếp (Direct Forecast)
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được huấn luyện!")
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        return np.clip(preds, 0, None)  # Nhu cầu không âm

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được huấn luyện!")
        scores   = self.model.feature_importance(importance_type=importance_type)
        features = self.model.feature_name()
        df = pd.DataFrame({"feature": features, "importance": scores})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Lưu / Load mô hình
    # ------------------------------------------------------------------
    def save(self, output_dir: str = "outputs/models") -> str:
        if self.model is None:
            raise RuntimeError("Không có mô hình để lưu!")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"lgb_tweedie_{self.store_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[LGB:{self.store_id}] Mô hình đã lưu tại: {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "LightGBMTweedie":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[LGB] Đã load mô hình từ: {path}")
        return obj
