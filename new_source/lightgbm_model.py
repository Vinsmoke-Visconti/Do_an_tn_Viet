import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class LightGBMTop1Model:
    def __init__(self, params=None):
        """
        Khởi tạo mô hình LightGBM theo phương pháp của Top 1 (YJ_STU).
        Sử dụng Tweedie loss để xử lý tốt dữ liệu sales có nhiều số 0 (intermittent demand).
        """
        self.default_params = {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1, # 1.1 đến 1.5 thường dùng cho sales
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'seed': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        self.params = params if params else self.default_params
        self.model = None

    def train(self, X_train, y_train, X_val, y_val, categorical_features='auto', num_boost_round=1000, early_stopping_rounds=50):
        """
        Huấn luyện mô hình với early stopping.
        """
        print("[LightGBM] Đang chuẩn bị dữ liệu...")
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features, free_raw_data=False)

        print(f"[LightGBM] Bắt đầu huấn luyện với objective={self.params['objective']}...")
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=50)
            ]
        )
        print(f"[LightGBM] Huấn luyện hoàn tất. Best iteration: {self.model.best_iteration}")

    def predict(self, X_test):
        """Dự báo trên tập test"""
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện!")
        preds = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        # Nhu cầu không thể âm
        return np.clip(preds, a_min=0, a_max=None)

    def plot_feature_importance(self, max_num_features=20, importance_type='gain', output_path=None):
        """
        Vẽ biểu đồ Feature Importance (Split hoặc Gain). Đội Top 1 thường dùng Gain.
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện!")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        lgb.plot_importance(self.model, max_num_features=max_num_features, importance_type=importance_type, ax=ax, title=f'LightGBM Feature Importance ({importance_type})')
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150)
            print(f"[LightGBM] Đã lưu biểu đồ Feature Importance tại: {output_path}")
        else:
            plt.show()
        plt.close(fig)

    def get_feature_importance_df(self, importance_type='gain'):
        """Trả về DataFrame chứa Feature Importance"""
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện!")
        importance = self.model.feature_importance(importance_type=importance_type)
        features = self.model.feature_name()
        df = pd.DataFrame({'feature': features, 'importance': importance})
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        return df
