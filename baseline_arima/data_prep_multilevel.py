"""
Module: data_prep_multilevel.py
Purpose: Build M5 hierarchical time-series data for 5 aggregation levels
         from raw CSV files, using the FULL date range d_1 -> d_1913.

Levels:
    L1  : National Total          - 1       series  (long format)
    L2  : State                   - 3       series  (long format)
    L3  : Store                   - 10      series  (long format)
    L9  : Store x Department      - 70      series  (long format)
    L12 : Store x Item (SKU)      - 30,490  series  (WIDE format)

Date Range:
    Full  : d_1  (2011-01-29) -> d_1913 (2016-04-24)
    Train : d_1  -> d_1885  (~5.2 years)
    Test  : d_1886 -> d_1913  (28 days, M5 standard)

Output:
    outputs/data/level_L1.parquet  .. level_L12.parquet
"""

import os
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(BASE_DIR, "..", "new_source", "data", "raw")
DATA_OUT_DIR = os.path.join(BASE_DIR, "outputs", "data")

SALES_CSV    = os.path.join(RAW_DIR, "sales_train_evaluation.csv")
CALENDAR_CSV = os.path.join(RAW_DIR, "calendar.csv")

LAST_D  = 1913
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Aggregation keys per level (L12 is handled separately — wide format)
LEVEL_GROUPS = {
    "L1": [],
    "L2": ["state_id"],
    "L3": ["store_id"],
    "L9": ["store_id", "dept_id"],
}

os.makedirs(DATA_OUT_DIR, exist_ok=True)


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_calendar() -> pd.DataFrame:
    """Load calendar: d_num -> date mapping (d_1 through LAST_D)."""
    cal = pd.read_csv(CALENDAR_CSV, usecols=["d", "date"])
    cal["d_num"] = cal["d"].str[2:].astype(int)
    cal["date"]  = pd.to_datetime(cal["date"])
    cal = cal[cal["d_num"] <= LAST_D].copy()
    return cal[["d_num", "date"]].reset_index(drop=True)


def load_sales_wide() -> pd.DataFrame:
    """
    Load sales_train_evaluation.csv in wide format.
    Columns: id_cols + d_1 .. d_LAST_D
    Returns: DataFrame with 30,490 rows x (6 + LAST_D) cols
    """
    d_cols_needed = set([f"d_{i}" for i in range(1, LAST_D + 1)])
    all_needed    = set(ID_COLS) | d_cols_needed
    df = pd.read_csv(SALES_CSV, usecols=lambda c: c in all_needed)

    # Sort d-columns numerically
    present_d = sorted(
        [c for c in df.columns if c.startswith("d_") and c[2:].isdigit()],
        key=lambda x: int(x[2:]),
    )
    return df[ID_COLS + present_d]


# ── Builder ────────────────────────────────────────────────────────────────────

def build_long_level(
    df_wide: pd.DataFrame,
    cal: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """
    Aggregate sales to the given level, then return long-format DataFrame:
        series_id | d_num | date | sales
    """
    d_cols = [c for c in df_wide.columns if c.startswith("d_") and c[2:].isdigit()]
    grp    = LEVEL_GROUPS[level]

    if not grp:
        # L1: sum all 30490 rows -> 1 series
        totals = df_wide[d_cols].sum(axis=0).to_frame().T
        totals["series_id"] = "NATIONAL_TOTAL"
        agg_wide = totals
    else:
        agg_wide = (
            df_wide.groupby(grp, sort=True)[d_cols]
            .sum()
            .reset_index()
        )
        if len(grp) == 1:
            agg_wide["series_id"] = agg_wide[grp[0]]
        else:
            agg_wide["series_id"] = (
                agg_wide[grp].astype(str).agg("_".join, axis=1)
            )

    # Melt aggregated wide -> long
    df_long = agg_wide[["series_id"] + d_cols].melt(
        id_vars=["series_id"],
        var_name="d",
        value_name="sales",
    )
    df_long["d_num"] = df_long["d"].str[2:].astype(int)
    df_long.drop(columns=["d"], inplace=True)

    # Attach real dates from calendar
    df_long = df_long.merge(cal, on="d_num", how="left")
    df_long.sort_values(["series_id", "d_num"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    return df_long[["series_id", "d_num", "date", "sales"]]


# ── Entry ──────────────────────────────────────────────────────────────────────

def build_all(force: bool = False) -> dict:
    """
    Build and save parquet files for all 5 levels.
    Skips existing files unless force=True.

    Returns:
        dict mapping level -> output parquet path
    """
    cal     = load_calendar()
    df_wide = load_sales_wide()

    d_present = [c for c in df_wide.columns if c.startswith("d_") and c[2:].isdigit()]
    print(
        f"[data_prep] {df_wide.shape[0]:,} SKUs x {len(d_present)} days "
        f"({cal['date'].min().date()} -> {cal['date'].max().date()})"
    )

    paths = {}

    # ── L1, L2, L3, L9 — long format ─────────────────────────────────────────
    for level in ["L1", "L2", "L3", "L9"]:
        out = os.path.join(DATA_OUT_DIR, f"level_{level}.parquet")
        paths[level] = out

        if os.path.exists(out) and not force:
            n = pd.read_parquet(out, columns=["series_id"])["series_id"].nunique()
            print(f"[data_prep] {level}: cached ({n} series) -> {out}")
            continue

        print(f"[data_prep] Building {level} ...", end="", flush=True)
        df_lv = build_long_level(df_wide, cal, level)
        n     = df_lv["series_id"].nunique()
        df_lv.to_parquet(out, index=False)
        print(f" {n} series, {len(df_lv):,} rows saved.")

    # ── L12 — wide format (to avoid 58M-row melt) ────────────────────────────
    out12 = os.path.join(DATA_OUT_DIR, "level_L12.parquet")
    paths["L12"] = out12

    if os.path.exists(out12) and not force:
        n12 = pd.read_parquet(out12, columns=["id"]).shape[0]
        print(f"[data_prep] L12: cached ({n12:,} SKUs) -> {out12}")
    else:
        print(f"[data_prep] Building L12 (wide) ...", end="", flush=True)
        d_cols = [c for c in df_wide.columns if c.startswith("d_") and c[2:].isdigit()]
        df12   = df_wide[ID_COLS + d_cols].copy()
        df12.to_parquet(out12, index=False)
        print(f" {len(df12):,} SKUs x {len(d_cols)} days saved.")

    print("[data_prep] All levels ready.")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="M5 Multi-level data prep")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild all levels even if cached")
    args = parser.parse_args()
    build_all(force=args.force)

