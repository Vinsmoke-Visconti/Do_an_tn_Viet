"""
Module: pipeline_arima_multilevel.py
Purpose: Multi-level ARIMA(p,d,q) baseline pipeline for M5 forecasting.

Supports 5 aggregation levels:
    L1 : National Total        (1       series) -- sequential
    L2 : State                 (3       series) -- sequential
    L3 : Store                 (10      series) -- sequential
    L9 : Store x Department    (70      series) -- sequential
    L12: Store x Item (SKU)    (30,490  series) -- parallel + checkpoint

Design (Approach A -- No Fallback):
    - Fit/predict failure -> raise ARIMAFitError / ARIMAPredictError
    - Never silently substitute with another method

Train/Test split (M5 standard):
    Train : d_1 (2011-01-29) -> d_1885
    Test  : d_1886 -> d_1913 (28 days)
"""

from __future__ import annotations
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

D1_DATE = "2011-01-29"     # d_1 reference date for date computation

# Level config: use parallel (True) or sequential (False)
LEVEL_USE_PARALLEL = {
    "L1":  False,
    "L2":  False,
    "L3":  False,
    "L9":  False,   # 70 series is fast enough sequentially (~35s)
    "L12": True,    # 30490 series requires parallelism
}


# -- Metrics --------------------------------------------------------------------

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=float).flatten()
    p = np.clip(np.asarray(y_pred, dtype=float).flatten(), 0.0, None)
    rmse  = float(np.sqrt(np.mean((y - p) ** 2)))
    mae   = float(np.mean(np.abs(y - p)))
    denom = np.sum(np.abs(y))
    wmape = float(np.sum(np.abs(y - p)) / denom * 100.0) if denom > 0 else 0.0
    d2    = (np.abs(y) + np.abs(p)) / 2.0 + 1e-5
    smape = float(np.mean(np.abs(p - y) / d2) * 100.0)
    return {
        "RMSE":      round(rmse,  4),
        "MAE":       round(mae,   4),
        "WMAPE (%)": round(wmape, 4),
        "sMAPE (%)": round(smape, 4),
    }


# -- Worker (module-level -- required for Windows multiprocessing spawn) ----------

def _fit_predict_one(args: tuple) -> dict:
    """
    Fit ARIMA on a single time series and predict test horizon.

    MUST be a top-level (module-level) function for Windows multiprocessing
    to allow pickling via spawn.

    Args:
        args: (series_id, sales_arr, d_nums, order, test_start_d, test_days)

    Returns:
        dict with keys:
            series_id: str
            metrics  : dict | None
            fc_rows  : dict of lists | None   (raw data for pd.DataFrame)
            error    : str | None
    """
    sid, sales_arr, d_nums, order, test_start_d, test_days = args

    # Ensure model_arima_pure is importable in child process
    _cur = os.path.dirname(os.path.abspath(__file__))
    if _cur not in sys.path:
        sys.path.insert(0, _cur)

    from model_arima_pure import PureARIMAModel  # noqa: import inside worker

    warnings.filterwarnings("ignore")

    d_nums    = np.asarray(d_nums, dtype=int)
    sales_arr = np.asarray(sales_arr, dtype=float)

    train_mask = d_nums < test_start_d
    test_mask  = (d_nums >= test_start_d)

    y_train = sales_arr[train_mask]
    y_test  = sales_arr[test_mask][:test_days]
    d_test  = d_nums[test_mask][:test_days]

    if len(y_test) == 0:
        return {"series_id": sid, "metrics": None, "fc_rows": None,
                "error": "No test data found"}

    # Compute date strings from d_num (d_1 = 2011-01-29)
    D1 = pd.Timestamp(D1_DATE)
    dates_test = [
        (D1 + pd.Timedelta(days=int(d) - 1)).strftime("%d-%m-%Y")
        for d in d_test
    ]

    try:
        model = PureARIMAModel(order=order, name=f"ARIMA_{sid}")
        model.fit(y_train)
        y_pred, ci_lo, ci_hi = model.predict(horizon=len(y_test), alpha=0.05)
    except Exception as e:
        return {"series_id": sid, "metrics": None, "fc_rows": None, "error": str(e)}

    metrics = _evaluate(y_test, y_pred)

    eps     = 1e-5
    err_pct = np.where(y_test > eps, np.abs(y_test - y_pred) / y_test * 100.0, 0.0)

    fc_rows = {
        "series_id":      [sid] * len(y_test),
        "d_num":          d_test.tolist(),
        "date":           dates_test,
        "actual_sales":   y_test.tolist(),
        "forecast_arima": y_pred.tolist(),
        "ci_lower_95":    (ci_lo.tolist() if ci_lo is not None else y_pred.tolist()),
        "ci_upper_95":    (ci_hi.tolist() if ci_hi is not None else y_pred.tolist()),
        "error_abs":      np.abs(y_test - y_pred).tolist(),
        "error_pct":      err_pct.tolist(),
    }

    return {"series_id": sid, "metrics": metrics, "fc_rows": fc_rows, "error": None}


# -- Pipeline -------------------------------------------------------------------

class MultiLevelARIMAPipeline:
    """
    ARIMA(p,d,q) baseline pipeline for 5-level M5 hierarchical forecasting.

    Usage:
        pipe = MultiLevelARIMAPipeline(order=(7,1,1), n_jobs=6)
        pipe.run_all()                     # all 5 levels
        pipe.run_all(levels=["L1","L3"])   # specific levels
    """

    BATCH_SIZE = 500   # Series per checkpoint batch for L12

    def __init__(
        self,
        order: Tuple[int, int, int] = (7, 1, 1),
        test_start_d: int = 1886,
        test_days: int = 28,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        n_jobs: int = 4,
    ):
        self.order        = order
        self.test_start_d = test_start_d
        self.test_days    = test_days
        self.n_jobs       = n_jobs

        self.data_dir    = data_dir    or os.path.join(CURRENT_DIR, "outputs", "data")
        output_dir       = output_dir  or os.path.join(CURRENT_DIR, "outputs")
        self.results_dir = os.path.join(output_dir, "results")
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    # -- Arg builders ----------------------------------------------------------

    def _args_from_long(self, level: str) -> List[tuple]:
        """
        Load long-format parquet (L1/L2/L3/L9) and convert to worker arg list.
        Each arg = (series_id, sales_arr, d_nums, order, test_start_d, test_days)
        """
        path = os.path.join(self.data_dir, f"level_{level}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Level data not found: {path}\n"
                f"Run data_prep_multilevel.py first."
            )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["series_id", "d_num"]).reset_index(drop=True)

        args = []
        for sid, grp in df.groupby("series_id", sort=True):
            grp = grp.sort_values("d_num").reset_index(drop=True)
            args.append((
                sid,
                grp["sales"].values.astype(float),
                grp["d_num"].values.astype(int),
                self.order,
                self.test_start_d,
                self.test_days,
            ))
        return args

    def _l12_batch_args(self, df_wide: pd.DataFrame, d_cols: List[str],
                        d_nums: np.ndarray, start: int, end: int) -> List[tuple]:
        """Build worker args for one batch slice of L12 wide-format data."""
        batch_sales = df_wide[d_cols].iloc[start:end].values.astype(float)
        batch_sids  = df_wide["id"].iloc[start:end].values
        return [
            (batch_sids[i], batch_sales[i], d_nums, self.order,
             self.test_start_d, self.test_days)
            for i in range(len(batch_sids))
        ]

    # -- Sequential runner -----------------------------------------------------

    def _run_sequential(
        self,
        args_list: List[tuple],
        level: str,
        verbose: bool = True,
    ) -> Tuple[List[dict], List[dict]]:
        """Fit ARIMA sequentially and return (successes, errors)."""
        successes, errors = [], []
        for args in args_list:
            res = _fit_predict_one(args)
            if res["error"] is None:
                successes.append(res)
                if verbose:
                    m = res["metrics"]
                    print(f"    >> {res['series_id']:45s} "
                          f"WMAPE={m['WMAPE (%)']:6.2f}%  RMSE={m['RMSE']:.1f}")
            else:
                errors.append({"series_id": res["series_id"], "error": res["error"]})
                if verbose:
                    print(f"    >> {res['series_id']:45s} ERROR: {res['error'][:70]}")
        return successes, errors

    # -- Parallel batched runner (L12) -----------------------------------------

    def _run_l12_parallel(self) -> dict:
        """
        Fit ARIMA for L12 (30,490 series) in parallel batches with checkpoint/resume.

        Resume logic:
            - Reads existing metrics CSV to find already-processed series_ids
            - Skips them in subsequent runs
        """
        path = os.path.join(self.data_dir, "level_L12.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"L12 data not found: {path}\n"
                "Run data_prep_multilevel.py first."
            )

        fc_path = os.path.join(self.results_dir, "arima_l12_forecast.csv")
        m_path  = os.path.join(self.results_dir, "arima_l12_metrics.csv")

        # Resume: find already-done series
        done_sids: set = set()
        if os.path.exists(m_path):
            done_sids = set(pd.read_csv(m_path, usecols=["series_id"])["series_id"].tolist())
            print(f"    [resume] {len(done_sids):,} series already done from checkpoint.")

        # Load L12 wide format
        print("    [L12] Loading wide-format parquet ...", end="", flush=True)
        df_wide = pd.read_parquet(path)
        d_cols  = sorted(
            [c for c in df_wide.columns if c.startswith("d_") and c[2:].isdigit()],
            key=lambda x: int(x[2:]),
        )
        d_nums = np.array([int(c[2:]) for c in d_cols], dtype=int)
        print(f" {len(df_wide):,} SKUs loaded.")

        # Filter out already-done series
        if done_sids:
            df_wide = df_wide[~df_wide["id"].isin(done_sids)].reset_index(drop=True)

        n_total     = len(df_wide) + len(done_sids)
        n_remaining = len(df_wide)
        n_jobs      = min(self.n_jobs, n_remaining, os.cpu_count() or 1)

        if n_remaining == 0:
            print(f"    [L12] All {n_total:,} series already done.")
            m_df = pd.read_csv(m_path)
            return {
                "n_total": n_total, "n_ok": len(m_df),
                "n_error": 0, "avg_wmape": m_df["WMAPE (%)"].mean()
            }

        print(f"    [L12] {n_remaining:,}/{n_total:,} remaining | n_jobs={n_jobs} | batch={self.BATCH_SIZE}")

        t0         = time.time()
        total_done = len(done_sids)
        n_ok       = len(done_sids)
        n_error    = 0

        for batch_start in range(0, n_remaining, self.BATCH_SIZE):
            batch_end  = min(batch_start + self.BATCH_SIZE, n_remaining)
            batch_args = self._l12_batch_args(df_wide, d_cols, d_nums, batch_start, batch_end)

            # Run batch in parallel
            with ProcessPoolExecutor(max_workers=n_jobs) as exc:
                batch_results = list(exc.map(_fit_predict_one, batch_args))

            # Collect results
            batch_fc_rows  = []
            batch_metrics  = []

            for res in batch_results:
                if res["error"] is None:
                    batch_fc_rows.append(pd.DataFrame(res["fc_rows"]))
                    batch_metrics.append({"series_id": res["series_id"], **res["metrics"]})
                    n_ok += 1
                else:
                    n_error += 1
                total_done += 1

            # Checkpoint: append to CSV
            if batch_fc_rows:
                new_fc = pd.concat(batch_fc_rows, ignore_index=True)
                write_header = not os.path.exists(fc_path)
                new_fc.to_csv(fc_path, mode="a", header=write_header,
                              index=False, encoding="utf-8-sig")

            if batch_metrics:
                new_m = pd.DataFrame(batch_metrics)
                write_header = not os.path.exists(m_path)
                new_m.to_csv(m_path, mode="a", header=write_header,
                             index=False, encoding="utf-8-sig")

            # Progress
            elapsed = time.time() - t0
            rate    = (total_done - len(done_sids)) / max(elapsed, 1)
            eta     = (n_total - total_done) / max(rate, 0.001)
            print(
                f"    [L12] {total_done:,}/{n_total:,} | "
                f"elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min | "
                f"ok={n_ok:,} err={n_error:,}"
            )

        elapsed = time.time() - t0
        m_df    = pd.read_csv(m_path)
        return {
            "n_total":   n_total,
            "n_ok":      n_ok,
            "n_error":   n_error,
            "elapsed_s": round(elapsed, 1),
            "avg_wmape": round(float(m_df["WMAPE (%)"].mean()), 4),
            "avg_rmse":  round(float(m_df["RMSE"].mean()), 4),
        }

    # -- Per-level runner ------------------------------------------------------

    def _save_results(
        self,
        level: str,
        successes: List[dict],
        errors: List[dict],
        elapsed: float,
    ) -> dict:
        """Save forecast CSV, metrics CSV, summary JSON for a completed level."""
        if not successes:
            print(f"    [warn] No successful results for {level}.")
            return {}

        fc_df = pd.concat(
            [pd.DataFrame(r["fc_rows"]) for r in successes], ignore_index=True
        )
        m_df  = pd.DataFrame(
            [{"series_id": r["series_id"], **r["metrics"]} for r in successes]
        )

        avg_row = {
            "series_id": "AVERAGE",
            "RMSE":       round(float(m_df["RMSE"].mean()),       4),
            "MAE":        round(float(m_df["MAE"].mean()),        4),
            "WMAPE (%)":  round(float(m_df["WMAPE (%)"].mean()),  4),
            "sMAPE (%)":  round(float(m_df["sMAPE (%)"].mean()),  4),
        }
        m_df_full = pd.concat([m_df, pd.DataFrame([avg_row])], ignore_index=True)

        tag      = level.lower()
        fc_path  = os.path.join(self.results_dir, f"arima_{tag}_forecast.csv")
        m_path   = os.path.join(self.results_dir, f"arima_{tag}_metrics.csv")
        j_path   = os.path.join(self.results_dir, f"arima_{tag}_summary.json")

        fc_df.to_csv(fc_path,    index=False, encoding="utf-8-sig")
        m_df_full.to_csv(m_path, index=False, encoding="utf-8-sig")

        summary = {
            "level":      level,
            "model":      f"ARIMA{list(self.order)}",
            "order":      list(self.order),
            "train_from": f"d_1 (2011-01-29)",
            "train_to":   f"d_{self.test_start_d - 1}",
            "test_from":  f"d_{self.test_start_d}",
            "test_to":    f"d_{self.test_start_d + self.test_days - 1}",
            "test_days":  self.test_days,
            "n_series":   len(successes) + len(errors),
            "n_ok":       len(successes),
            "n_error":    len(errors),
            "elapsed_s":  round(elapsed, 2),
            "errors":     errors[:20],          # store at most 20 errors
            "metrics":    m_df_full.to_dict(orient="records"),
        }
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=True)

        print(f"\n    Forecast CSV : {fc_path}")
        print(f"    Metrics  CSV : {m_path}")
        print(f"    Summary JSON : {j_path}")

        return avg_row

    def run_level(self, level: str) -> dict:
        """Run the ARIMA pipeline for one hierarchy level."""
        sep = "-" * 62
        print(f"\n{sep}")
        print(f"  [Level {level}]  ARIMA{self.order}  |  "
              f"d_1 -> d_{self.test_start_d - 1} (train) | "
              f"d_{self.test_start_d} -> d_{self.test_start_d + self.test_days - 1} (test)")
        print(sep)

        t0 = time.time()

        if level == "L12":
            # -- L12: parallel batched with checkpoint ----------------------
            stats   = self._run_l12_parallel()
            elapsed = time.time() - t0

            # Save JSON summary for L12
            j_path = os.path.join(self.results_dir, "arima_l12_summary.json")
            with open(j_path, "w", encoding="utf-8") as f:
                json.dump({
                    "level":      "L12",
                    "model":      f"ARIMA{list(self.order)}",
                    "order":      list(self.order),
                    "train_from": "d_1 (2011-01-29)",
                    "train_to":   f"d_{self.test_start_d - 1}",
                    "test_from":  f"d_{self.test_start_d}",
                    "test_to":    f"d_{self.test_start_d + self.test_days - 1}",
                    "test_days":  self.test_days,
                    **stats,
                }, f, indent=4)

            print(f"\n    [L12] Done in {elapsed/60:.1f} min | "
                  f"ok={stats.get('n_ok',0):,} | "
                  f"avg_WMAPE={stats.get('avg_wmape', '?')}%")
            return {"level": "L12", "elapsed": elapsed, **stats}

        else:
            # -- L1/L2/L3/L9: sequential -----------------------------------
            args_list          = self._args_from_long(level)
            n                  = len(args_list)
            print(f"  {n} series\n")
            successes, errors  = self._run_sequential(args_list, level, verbose=True)
            elapsed            = time.time() - t0
            avg_row            = self._save_results(level, successes, errors, elapsed)

            wmape = avg_row.get("WMAPE (%)", "?")
            rmse  = avg_row.get("RMSE",     "?")
            print(f"\n    Done in {elapsed:.1f}s  |  "
                  f"WMAPE_avg={wmape}%  RMSE_avg={rmse}  |  "
                  f"ok={len(successes)}/{n}  err={len(errors)}")
            return {"level": level, "elapsed": elapsed, "avg_row": avg_row}

    # -- Full pipeline ---------------------------------------------------------

    def run_all(self, levels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run ARIMA pipeline for all (or specified) hierarchy levels.

        Args:
            levels: list of level strings, e.g. ["L1","L2","L3","L9","L12"].
                    Defaults to all 5 levels.

        Returns:
            Summary dict keyed by level.
        """
        levels = levels or ["L1", "L2", "L3", "L9", "L12"]

        print("=" * 70)
        print("  MULTI-LEVEL ARIMA BASELINE  |  M5 FORECASTING ACCURACY (WALMART)")
        print(f"  Model : Pure ARIMA{self.order}  (No Seasonal, No Exogenous, No Fallback)")
        print(f"  Levels: {levels}")
        print(f"  Train : d_1 (2011-01-29) -> d_{self.test_start_d - 1}")
        print(f"  Test  : d_{self.test_start_d} -> d_{self.test_start_d + self.test_days - 1} ({self.test_days} days)")
        print("=" * 70)

        t_total = time.time()
        results = {}

        for lv in levels:
            try:
                results[lv] = self.run_level(lv)
            except Exception as e:
                print(f"\n[ERROR] Level {lv}: {e}")
                results[lv] = {"level": lv, "error": str(e)}

        total_elapsed = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        print("=" * 70)
        return results
