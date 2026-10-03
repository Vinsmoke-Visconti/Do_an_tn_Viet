import os
import pandas as pd
import numpy as np
from lightgbm_model import LightGBMTop1Model
import sys

from utils.metrics import evaluate_forecast

def prepare_data(data_path="data/processed/m5_sku_level_features.parquet"):
    """
    Load data và chia tập train, val, test theo thời gian (temporal split).
    """
    # Lấy đường dẫn tuyệt đối dựa trên vị trí của file này
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(base_dir, data_path)
    
    print(f"[*] Đang load dữ liệu từ {full_data_path}...")
    df = pd.read_parquet(full_data_path)
    
    # Định nghĩa các cột không phải đặc trưng dự báo
    exclude_cols = ['id', 'date', 'd', 'sales']
    
    # Định nghĩa các cột categorical (hoặc tự động chuyển object sang category)
    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
        if col not in cat_cols:
            cat_cols.append(col)
            
    features = [c for c in df.columns if c not in exclude_cols]
    
    # Chia tập theo d_num (giống như các mô hình trước: horizon = 28)
    max_d = df["d_num"].max()
    test_split = max_d - 28
    val_split = test_split - 28
    
    print(f"[*] Chia tập dữ liệu:")
    print(f"    Train: d_num <= {val_split}")
    print(f"    Val  : {val_split} < d_num <= {test_split}")
    print(f"    Test : d_num > {test_split}")
    
    train_df = df[df["d_num"] <= val_split].copy()
    val_df = df[(df["d_num"] > val_split) & (df["d_num"] <= test_split)].copy()
    test_df = df[df["d_num"] > test_split].copy()
    
    X_train, y_train = train_df[features], train_df['sales']
    X_val, y_val = val_df[features], val_df['sales']
    X_test, y_test = test_df[features], test_df['sales']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, features, cat_cols, test_df

def main():
    print("=================================================================")
    print(" THỰC NGHIỆM: MÔ HÌNH LIGHTGBM (PHƯƠNG PHÁP ĐỘI TOP 1 M5)")
    print("=================================================================")
    
    # 1. Chuẩn bị dữ liệu
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, features, cat_cols, test_df = prepare_data()
    except FileNotFoundError:
        print("[!] Không tìm thấy file dữ liệu processed. Vui lòng chạy pipeline Phase 2 trước.")
        return

    # Filter lại cat_cols thực tế có trong data
    actual_cat_cols = [c for c in cat_cols if c in X_train.columns]
    
    # 2. Khởi tạo và huấn luyện mô hình LightGBM
    lgb_wrapper = LightGBMTop1Model()
    
    print("\n[*] Bắt đầu huấn luyện LightGBM...")
    lgb_wrapper.train(
        X_train, y_train, 
        X_val, y_val, 
        categorical_features=actual_cat_cols,
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    
    # 3. Đánh giá trên tập Test
    print("\n[*] Đang dự báo trên tập Test...")
    preds = lgb_wrapper.predict(X_test)
    test_df['lgb_pred'] = preds
    
    metrics = evaluate_forecast(y_test.values, preds)
    print("\n[KẾT QUẢ ĐÁNH GIÁ CẤP ĐỘ SKU]")
    print(f"  - RMSE  : {metrics['RMSE']:.3f}")
    print(f"  - MAE   : {metrics['MAE']:.3f}")
    print(f"  - WMAPE : {metrics['WMAPE (%)']:.3f}%")
    print(f"  - sMAPE : {metrics['sMAPE (%)']:.3f}%")
    
    # 4. Phân tích Feature Importance (Khả năng diễn giải theo phương pháp Top 1)
    print("\n[*] Trích xuất Feature Importance (Gain)...")
    imp_df = lgb_wrapper.get_feature_importance_df(importance_type='gain')
    
    print("\n--- TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT (THEO GAIN) ---")
    print(imp_df.head(10).to_string(index=False))
    
    # Tính tổng đóng góp %
    total_gain = imp_df['importance'].sum()
    imp_df['importance_percent'] = (imp_df['importance'] / total_gain) * 100
    
    print("\n--- PHÂN TÍCH NHÓM ĐẶC TRƯNG ---")
    lags_gain = imp_df[imp_df['feature'].str.contains('lag|rolling')]['importance_percent'].sum()
    holiday_gain = imp_df[imp_df['feature'].str.contains('event|evt_')]['importance_percent'].sum()
    price_gain = imp_df[imp_df['feature'].str.contains('price|discount')]['importance_percent'].sum()
    calendar_gain = imp_df[imp_df['feature'].str.contains('wday|month|weekend')]['importance_percent'].sum()
    
    print(f"  - Chuỗi trễ (Lags & Rolling): {lags_gain:.1f}%")
    print(f"  - Chu kỳ thời gian (Calendar) : {calendar_gain:.1f}%")
    print(f"  - Giá & Khuyến mãi (Price)    : {price_gain:.1f}%")
    print(f"  - Ngày lễ (Holidays/Events)   : {holiday_gain:.1f}%")
    
    # Lưu biểu đồ
    output_fig = os.path.join(os.path.dirname(__file__), 'lgb_feature_importance.png')
    lgb_wrapper.plot_feature_importance(max_num_features=15, output_path=output_fig)
    
    print("\n[✓] Thực nghiệm LightGBM hoàn tất!")

if __name__ == "__main__":
    main()
