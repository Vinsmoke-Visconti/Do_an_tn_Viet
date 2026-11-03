"""
Entry point: Multi-level ARIMA(7,1,1) Baseline for M5 Forecasting.

Pipeline:
    Step 1 : Data preparation  — build 5-level parquets from raw CSV
    Step 2 : ARIMA training    — fit ARIMA per series for each level
    Step 3 : Visualization     — plots of forecasts and metrics

Usage:
    # Full run (all 5 levels, 4 parallel workers)
    python run_arima_multilevel.py

    # Specific levels only (fast check: L1 + L3)
    python run_arima_multilevel.py --levels L1 L3

    # More parallel workers for L12 (recommended: 4-8)
    python run_arima_multilevel.py --jobs 8

    # Force rebuild of data files
    python run_arima_multilevel.py --rebuild-data

    # Specify ARIMA order
    python run_arima_multilevel.py --order 7 1 1

Levels & expected runtime (ARIMA(7,1,1), i7 8-core):
    L1  :       1 series   ~  1s
    L2  :       3 series   ~  2s
    L3  :      10 series   ~  6s
    L9  :      70 series   ~ 40s
    L12 : 30,490 series   ~ 30min (8 workers)
"""

import os
import sys
import argparse
import time

# Ensure current directory is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def main():
    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Multi-level ARIMA baseline for M5 Forecasting (Walmart)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--levels", nargs="+",
        default=["L1", "L2", "L3", "L9", "L12"],
        choices=["L1", "L2", "L3", "L9", "L12"],
        help="Levels to run (default: all 5 levels)",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Parallel workers for L12 (default: 4)",
    )
    parser.add_argument(
        "--order", nargs=3, type=int, default=[7, 1, 1],
        metavar=("p", "d", "q"),
        help="ARIMA order (default: 7 1 1)",
    )
    parser.add_argument(
        "--rebuild-data", action="store_true",
        help="Force rebuild level parquet files even if cached",
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip visualization step",
    )
    parser.add_argument(
        "--no-l12", action="store_true",
        help="Exclude L12 from run (useful for quick testing)",
    )
    args = parser.parse_args()

    # Apply --no-l12 flag
    levels = [lv for lv in args.levels if not (args.no_l12 and lv == "L12")]

    order = tuple(args.order)

    print("=" * 70)
    print("  BASELINE: Multi-Level Pure ARIMA")
    print(f"  Model  : ARIMA{list(order)}  (No Seasonal, No Exogenous)")
    print(f"  Levels : {levels}")
    print(f"  Workers: {args.jobs} (for L12)")
    print("=" * 70)

    t_start = time.time()

    # ── Step 1: Data prep ─────────────────────────────────────────────────────
    print("\n[Step 1/3] Data preparation ...")
    from data_prep_multilevel import build_all
    build_all(force=args.rebuild_data)

    # ── Step 2: ARIMA pipeline ────────────────────────────────────────────────
    print("\n[Step 2/3] Running ARIMA pipeline ...")
    from pipeline_arima_multilevel import MultiLevelARIMAPipeline
    pipeline = MultiLevelARIMAPipeline(
        order=order,
        n_jobs=args.jobs,
    )
    results = pipeline.run_all(levels=levels)

    # ── Step 3: Visualization ─────────────────────────────────────────────────
    if not args.skip_viz:
        print("\n[Step 3/3] Generating visualizations ...")
        from visualize_arima_multilevel import generate_all
        generate_all()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE  |  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Outputs: {os.path.join(CURRENT_DIR, 'outputs')}")
    print("=" * 70)


# Windows multiprocessing REQUIRES this guard.
# ProcessPoolExecutor uses 'spawn' on Windows, which re-imports this module.
# Without the guard, child processes would re-enter main() -> infinite loop.
if __name__ == "__main__":
    main()

