"""
Module: memory_optimizer.py
Mục đích: Tối ưu kiểu dữ liệu Pandas để giảm RAM khi xử lý M5 dataset lớn.
"""
import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Tự động ép kiểu dữ liệu của các cột số sang kiểu nhỏ hơn để tiết kiệm RAM.
    Bỏ qua: object, category, datetime, timedelta.
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

    for col in df.columns:
        col_type = df[col].dtype

        # Chỉ xử lý các cột kiểu số nguyên hoặc số thực thuần túy
        if col_type == object or str(col_type) == "category":
            continue
        if col_type.kind not in ("i", "u", "f"):   # i=int, u=uint, f=float; bỏ 'M'(datetime), 'm'(timedelta)
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if col_type.kind in ("i", "u"):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:  # float
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if verbose:
        saved_pct = (start_mem - end_mem) / start_mem * 100 if start_mem > 0 else 0
        print(f"[Mem Opt] RAM: {start_mem:.2f} MB → {end_mem:.2f} MB ({saved_pct:.1f}% tiết kiệm)")
    return df
