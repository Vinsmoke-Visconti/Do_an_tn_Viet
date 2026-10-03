"""
Script: run_all.py
Mục đích: Chạy toàn bộ pipeline 3 bước liên tiếp.

Sử dụng:
    python run_all.py            # Chạy đầy đủ 3 bước
    python run_all.py --skip 1   # Bỏ qua bước 1 (đã có data processed)
"""

import os, sys, argparse, subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_name: str, step_no: int):
    print(f"\n{'='*70}")
    print(f"  CHẠY BƯỚC {step_no}: {script_name}")
    print(f"{'='*70}")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script_name)],
        cwd=BASE_DIR, env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    if result.returncode != 0:
        print(f"\n[!] Bước {step_no} thất bại (exit code {result.returncode}). Dừng pipeline.")
        sys.exit(result.returncode)
    print(f"\n[✓] Bước {step_no} hoàn tất.")


def main():
    parser = argparse.ArgumentParser(description="LightGBM Top-1 M5 Full Pipeline")
    parser.add_argument("--skip", type=int, default=0,
                        help="Bỏ qua các bước <= giá trị này (1=bỏ preprocessing).")
    args = parser.parse_args()

    steps = [
        (1, "01_preprocessing.py"),
        (2, "02_train_and_forecast.py"),
        (3, "03_evaluation_report.py"),
    ]

    for step_no, script in steps:
        if step_no <= args.skip:
            print(f"[SKIP] Bỏ qua Bước {step_no}: {script}")
            continue
        run_step(script, step_no)

    print(f"\n{'='*70}")
    print("  TOÀN BỘ PIPELINE HOÀN TẤT THÀNH CÔNG!")
    print(f"  Kết quả nằm trong: {os.path.join(BASE_DIR, 'outputs')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
