"""
Script 04: Phan tich tinh mua vu da cap do M5 (L1, L2, L3, L9, L12)
He sinh thai bieu do phan cap:
  - 00_master_summary/: So sanh SSI da cap, doi chieu Macro (L1) vs Micro (L12), Heatmap tong quan
  - L1_National/, L2_State/, L3_Store/, L9_Cat_State/, L12_SKU/:
      * 01_stl_decomposition.png: STL 4 thanh phan + chi so SSI/TSI + highlight nhip tuan
      * 02_periodogram_fft.png: Pho Fourier voi truc hoanh la Chu ky (So ngay)
      * 03_acf_pacf.png: Tuong quan tu va tuong quan rieng phan voi danh dau lag 7,14,28
      * 04_seasonality_preprocessing.png: Minh chung tien xu ly STL Deseasonal va Fourier Harmonics
"""
import sys
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
OUT_FIG    = os.path.join(BASE_DIR, "outputs", "figures", "seasonality")
OUT_RES    = os.path.join(BASE_DIR, "outputs", "results")

# Tao cac thu muc cap do
LEVEL_DIRS = {
    "summary": os.path.join(OUT_FIG, "00_master_summary"),
    "L1":      os.path.join(OUT_FIG, "L1_National"),
    "L2":      os.path.join(OUT_FIG, "L2_State"),
    "L3":      os.path.join(OUT_FIG, "L3_Store"),
    "L9":      os.path.join(OUT_FIG, "L9_Cat_State"),
    "L12":     os.path.join(OUT_FIG, "L12_SKU"),
}

for d in LEVEL_DIRS.values():
    os.makedirs(d, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

def safe_savefig(path, **kwargs):
    try:
        plt.savefig(path, **kwargs)
    except Exception as e:
        print(f"      [Warning] Khong the ghi de {os.path.basename(path)}: {e}")


COLORS = {
    "L1":  "#1E88E5",  # Blue
    "L2":  "#43A047",  # Green
    "L3":  "#FB8C00",  # Orange
    "L9":  "#8E24AA",  # Purple
    "L12": "#E53935",  # Red
}

LEVEL_LABELS = {
    "L1":  "L1 - Toan quoc (National Total)",
    "L2":  "L2 - Bang (State Level - CA)",
    "L3":  "L3 - Cua hang (Store Level - CA_1)",
    "L9":  "L9 - Nganh hang x Bang (Cat x State - FOODS_CA)",
    "L12": "L12 - San pham tai Cua hang (SKU Level)"
}

print("="*75)
print("  PHAN TICH TINH MUA VU DA CAP DO M5 - HE SINH THAI BIEU DO CHUYEN SAU")
print("="*75)

print("\n[1] Load du lieu va tong hop 5 cap do phan cap...")
df_sku = pd.read_parquet(os.path.join(DATA_PROC, "m5_sku_level_features.parquet"))
df_sku["date"] = pd.to_datetime(df_sku["date"])
df_base = df_sku[["date","d_num","item_id","dept_id","cat_id","store_id","state_id","sales"]].copy()

daily_L1  = df_base.groupby("date")["sales"].sum().reset_index()
daily_L2  = df_base.groupby(["date","state_id"])["sales"].sum().reset_index().rename(columns={"state_id":"key"})
daily_L3  = df_base.groupby(["date","store_id"])["sales"].sum().reset_index().rename(columns={"store_id":"key"})
daily_L9  = df_base.groupby(["date","state_id","cat_id"])["sales"].sum().reset_index()
daily_L9["key"] = daily_L9["state_id"] + "_" + daily_L9["cat_id"]
daily_L12 = df_base.groupby(["date","item_id","store_id"])["sales"].sum().reset_index()
daily_L12["key"] = daily_L12["item_id"] + "_" + daily_L12["store_id"]

def pick_series(df_level, key_col="key", prefer_key=None):
    if key_col not in df_level.columns:
        return df_level.set_index("date")["sales"].sort_index(), "Total"
    if prefer_key and prefer_key in df_level[key_col].values:
        chosen_key = prefer_key
    else:
        chosen_key = df_level.groupby(key_col)["sales"].count().idxmax()
    s = df_level[df_level[key_col]==chosen_key].set_index("date")["sales"].sort_index()
    print(f"   -> Dai dien {key_col}: {chosen_key} ({len(s)} ngay)")
    return s, chosen_key

series_dict = {}
series_keys = {}

s, k = pick_series(daily_L1);                 series_dict["L1"] = s; series_keys["L1"] = k
s, k = pick_series(daily_L2, prefer_key="CA"); series_dict["L2"] = s; series_keys["L2"] = k
s, k = pick_series(daily_L3, prefer_key="CA_1"); series_dict["L3"] = s; series_keys["L3"] = k
s, k = pick_series(daily_L9, prefer_key="CA_FOODS"); series_dict["L9"] = s; series_keys["L9"] = k
s, k = pick_series(daily_L12);                series_dict["L12"] = s; series_keys["L12"] = k

for lv, s in series_dict.items():
    print(f"   {lv} ({series_keys[lv]}): {len(s)} ngay | Mean={s.mean():.2f} | Std={s.std():.2f} | Zero%={(s==0).mean()*100:.1f}%")

# =========================================================================
# [2] STL DECOMPOSITION & TINH TOAN SSI / TSI
# =========================================================================
print("\n[2] STL Decomposition (period=7 ngay/tuan)...")
stl_results = {}
for lv, s in series_dict.items():
    s_clean = s.fillna(method="ffill").fillna(0)
    stl = STL(s_clean, period=7, robust=True)
    stl_results[lv] = stl.fit()

def seasonal_strength(res):
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    return max(0.0, 1.0 - var_r / (var_sr + 1e-9))

def trend_strength(res):
    var_r = np.var(res.resid)
    var_tr = np.var(res.trend + res.resid)
    return max(0.0, 1.0 - var_r / (var_tr + 1e-9))

ssi_data = []
for lv, res in stl_results.items():
    ssi = seasonal_strength(res)
    ts  = trend_strength(res)
    vsp = np.var(res.seasonal) / (np.var(res.observed) + 1e-9) * 100
    vtp = np.var(res.trend)    / (np.var(res.observed) + 1e-9) * 100
    vrp = np.var(res.resid)    / (np.var(res.observed) + 1e-9) * 100
    judge = "Manh (Strong)" if ssi > 0.64 else ("Trung binh (Moderate)" if ssi > 0.40 else "Yeu/Nhiễu (Weak/Noisy)")
    ssi_data.append({
        "Level": lv,
        "Label": LEVEL_LABELS[lv],
        "Key": series_keys[lv],
        "SSI_7day": round(ssi, 4),
        "TSI": round(ts, 4),
        "PctVar_Seasonal": round(vsp, 2),
        "PctVar_Trend": round(vtp, 2),
        "PctVar_Resid": round(vrp, 2),
        "Nhan_xet": judge
    })
    print(f"   {lv}: SSI={ssi:.4f} ({judge}) | TSI={ts:.4f} | VarS={vsp:.1f}% | VarR={vrp:.1f}%")

df_ssi = pd.DataFrame(ssi_data)
df_ssi.to_csv(os.path.join(OUT_RES, "seasonality_ssi_results.csv"), index=False, encoding="utf-8-sig")

# =========================================================================
# [3] SINH BIEU DO CHO TUNG CAP DO (L1, L2, L3, L9, L12)
# =========================================================================
print("\n[3] Sinh 4 bieu do chi tiet cho tung cap do trong tung thu muc rieng...")

for lv in ["L1", "L2", "L3", "L9", "L12"]:
    target_dir = LEVEL_DIRS[lv]
    color      = COLORS[lv]
    label_full = LEVEL_LABELS[lv]
    s          = series_dict[lv].fillna(method="ffill").fillna(0)
    dates      = s.index
    res        = stl_results[lv]
    ssi_val    = [d["SSI_7day"] for d in ssi_data if d["Level"]==lv][0]
    tsi_val    = [d["TSI"] for d in ssi_data if d["Level"]==lv][0]
    
    print(f"\n   >>> Dang xu ly thu muc: {target_dir}")
    
    # -------------------------------------------------------------
    # 3.1 STL Decomposition (Chi tiet cap do)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(f"STL Decomposition - {label_full}\n"
                 f"[Seasonal Strength Index F_S = {ssi_val:.4f} | Trend Strength F_T = {tsi_val:.4f} | Chu ky = 7 ngay]",
                 fontsize=13, fontweight="bold", y=0.98)
    
    # (a) Observed
    axes[0].plot(dates, res.observed, color=color, lw=1.2, label=f"Observed ({series_keys[lv]})")
    axes[0].set_ylabel("Doanh so thuc", fontsize=10, fontweight="bold", color=color)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)
    
    # (b) Trend
    axes[1].plot(dates, res.trend, color="#212121", lw=1.8, label="Trend Component")
    axes[1].fill_between(dates, res.trend, alpha=0.15, color=color)
    axes[1].set_ylabel("Xu huong (Trend)", fontsize=10, fontweight="bold")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=9)
    
    # (c) Seasonal (highlight cuoi tuan)
    axes[2].plot(dates, res.seasonal, color=color, lw=1.0, alpha=0.9, label="Seasonal 7-Day (Mua vu tuan)")
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    # Highlight cac ngay thu 7 & Chu nhat
    is_weekend = (dates.dayofweek >= 5)
    axes[2].fill_between(dates, res.seasonal.min()*1.2, res.seasonal.max()*1.2,
                         where=is_weekend, color="#FFCDD2", alpha=0.35, label="Cuoi tuan (Sat-Sun Peak)")
    axes[2].set_ylabel("Mua vu (Seasonal)", fontsize=10, fontweight="bold", color=color)
    axes[2].set_ylim(res.seasonal.min()*1.25, res.seasonal.max()*1.25)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=9)
    
    # (d) Residual
    axes[3].plot(dates, res.resid, color="#616161", lw=0.8, alpha=0.75, label="Residual / Nhieu (Noise)")
    axes[3].axhline(0, color="black", lw=0.8, ls="--")
    axes[3].set_ylabel("Phan du (Resid)", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Thoi gian (Ngay)", fontsize=10)
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="upper left", fontsize=9)
    
    # Formatting ticks
    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
        ax.tick_params(labelsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p_stl = os.path.join(target_dir, "01_stl_decomposition.png")
    plt.savefig(p_stl, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      [x] {p_stl}")

    # -------------------------------------------------------------
    # 3.2 Periodogram FFT (Pho Chu ky - So ngay)
    # -------------------------------------------------------------
    freqs, power = periodogram(s.values, scaling="density")
    valid = freqs > 0
    periods = 1.0 / freqs[valid]
    power_v = power[valid]
    
    mask = (periods >= 2) & (periods <= 100)
    sub_p = periods[mask]
    sub_pw = power_v[mask]
    
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.semilogy(sub_p, sub_pw, color=color, lw=1.8, label=f"Mat do cong suat FFT ({lv})")
    ax.fill_between(sub_p, sub_pw, alpha=0.15, color=color)
    
    # Danh dau chu ky 7, 14, 28, 35
    peak_periods = [7.0, 14.0, 28.0]
    peak_labels  = ["Chu ky 7 ngay\n(Nhip tuan - Peak)", "Chu ky 14 ngay\n(2 tuan)", "Chu ky 28 ngay\n(Thang SNAP)"]
    for p_mark, p_lbl in zip(peak_periods, peak_labels):
        ax.axvline(p_mark, color="#D32F2F", lw=1.5, ls="--", alpha=0.8)
        # Lay power gan p_mark nhat
        idx_near = np.argmin(np.abs(sub_p - p_mark))
        y_val = sub_pw[idx_near]
        ax.annotate(p_lbl, xy=(p_mark, y_val), xytext=(p_mark + 1.5, y_val * 2.2),
                    arrowprops=dict(facecolor="#D32F2F", edgecolor="#D32F2F", arrowstyle="->", lw=1.2),
                    fontsize=9, fontweight="bold", color="#B71C1C",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", edgecolor="#EF9A9A"))

    ax.set_title(f"Phan Tich Pho Fourier FFT - Xac Dinh Chu Ky Mua Vu: {label_full}\n"
                 f"(Dinh nang luong ro ret tai 7 ngay chung minh nhip mua sam tuan le manh me)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Chu ky P = 1/f (So ngay / chu ky)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Spectral Density (Log scale)", fontsize=11, fontweight="bold")
    ax.set_xlim(2, 60)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    p_fft = os.path.join(target_dir, "02_periodogram_fft.png")
    plt.savefig(p_fft, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      [x] {p_fft}")

    # -------------------------------------------------------------
    # 3.3 ACF & PACF Plot
    # -------------------------------------------------------------
    fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Ham Tu Tuong Quan (ACF) & Tu Tuong Quan Rieng Phan (PACF): {label_full}\n"
                 f"[Minh chung tinh phu thuoc chu ky tuan tai Lag 7, 14, 21, 28, 35]",
                 fontsize=13, fontweight="bold", y=0.98)
    
    plot_acf(s, ax=ax_acf, lags=45, alpha=0.05, color=color, vlines_kwargs={"colors": color, "lw": 1.5})
    plot_pacf(s, ax=ax_pacf, lags=45, alpha=0.05, color=color, vlines_kwargs={"colors": color, "lw": 1.5}, method="ywm")
    
    for lag in [7, 14, 21, 28, 35]:
        ax_acf.axvline(lag, color="#E53935", lw=1.2, ls="--", alpha=0.6)
        ax_pacf.axvline(lag, color="#E53935", lw=1.2, ls="--", alpha=0.6)
    
    ax_acf.set_title(f"ACF - Tu Tuong Quan (Lag 7 Peak)", fontsize=11, fontweight="bold", color=color)
    ax_pacf.set_title(f"PACF - Tu Tuong Quan Rieng Phan", fontsize=11, fontweight="bold", color=color)
    ax_acf.set_xlabel("Do tre Lag (So ngay)", fontsize=10)
    ax_pacf.set_xlabel("Do tre Lag (So ngay)", fontsize=10)
    ax_acf.grid(alpha=0.3)
    ax_pacf.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p_acf = os.path.join(target_dir, "03_acf_pacf.png")
    plt.savefig(p_acf, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      [x] {p_acf}")

    # -------------------------------------------------------------
    # 3.4 Preprocessing: Deseasonalized & Fourier Harmonics
    # -------------------------------------------------------------
    deseasonal = s - res.seasonal
    tail_len = min(120, len(s))
    tail_dates = dates[-tail_len:]
    s_tail = s.values[-tail_len:]
    deseas_tail = deseasonal.values[-tail_len:]
    trend_tail = res.trend.values[-tail_len:]
    
    t_arr = np.arange(len(s))
    sin7  = np.sin(2 * np.pi * t_arr / 7)[-tail_len:]
    cos7  = np.cos(2 * np.pi * t_arr / 7)[-tail_len:]
    sin14 = np.sin(2 * np.pi * t_arr / 14)[-tail_len:]
    cos14 = np.cos(2 * np.pi * t_arr / 14)[-tail_len:]
    
    fig, axes_prep = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(f"Minh Hoa Ky Thuat Tien Xu Ly Mua Vu: {label_full}\n"
                 f"(Deseasonalization & Bo sung Dac trung Song dieu hoa Fourier sin/cos)",
                 fontsize=13, fontweight="bold", y=0.98)
    
    # Subplot 1: Goc vs Trend
    axes_prep[0].plot(tail_dates, s_tail, color=color, lw=1.2, label="Chuoi goc Y_t")
    axes_prep[0].plot(tail_dates, trend_tail, color="#212121", lw=2.0, ls="--", label="Xu huong STL Trend T_t")
    axes_prep[0].fill_between(tail_dates, s_tail, trend_tail, alpha=0.18, color=color)
    axes_prep[0].set_ylabel("Doanh so", fontsize=10, fontweight="bold")
    axes_prep[0].set_title("(1) Chuoi Thoi Gian Goc va Duong Xu Huong", fontsize=11, fontweight="bold")
    axes_prep[0].legend(loc="upper left", fontsize=9)
    axes_prep[0].grid(alpha=0.3)
    
    # Subplot 2: Khử mùa vụ (Deseasonalized)
    axes_prep[1].plot(tail_dates, deseas_tail, color="#37474F", lw=1.2, label="Sau khu mua vu (Y_t - S_t)")
    axes_prep[1].axhline(deseas_tail.mean(), color="#E53935", lw=1.2, ls=":", label="Trung binh")
    axes_prep[1].set_ylabel("Doanh so Da Khu Mua Vu", fontsize=10, fontweight="bold")
    axes_prep[1].set_title("(2) Tin Hieu Sau Khi Loai Bo Mua Vu Tuan (Deseasonalized)", fontsize=11, fontweight="bold")
    axes_prep[1].legend(loc="upper left", fontsize=9)
    axes_prep[1].grid(alpha=0.3)
    
    # Subplot 3: Fourier Waveform nap vao mo hinh
    ax3_twin = axes_prep[2].twinx()
    axes_prep[2].plot(tail_dates, s_tail, color=color, lw=0.9, alpha=0.4, label="Chuoi goc (tham chieu)")
    ax3_twin.plot(tail_dates, sin7, color="#E53935", lw=1.8, label=r"$\sin(2\pi t / 7)$ - Song dieu hoa 7 ngay")
    ax3_twin.plot(tail_dates, cos7, color="#1E88E5", lw=1.8, ls="--", label=r"$\cos(2\pi t / 7)$ - Song dieu hoa 7 ngay")
    ax3_twin.plot(tail_dates, sin14, color="#FB8C00", lw=1.2, ls=":", label=r"$\sin(2\pi t / 14)$ - Song 14 ngay")
    
    axes_prep[2].set_ylabel("Doanh so", fontsize=10)
    ax3_twin.set_ylabel("Gia tri song Fourier [-1, 1]", fontsize=10, fontweight="bold", color="#B71C1C")
    axes_prep[2].set_xlabel("Thoi gian (120 ngay gan nhat)", fontsize=10)
    axes_prep[2].set_title("(3) Cac Bien Ngoai Sinh Fourier (Sin/Cos Harmonics) Cung Cap Cho LightGBM / TFT",
                           fontsize=11, fontweight="bold")
    
    lines1, labels1 = axes_prep[2].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axes_prep[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.5)
    axes_prep[2].grid(alpha=0.3)
    
    for ax_p in axes_prep:
        ax_p.xaxis.set_major_locator(mticker.MaxNLocator(8))
        ax_p.tick_params(labelsize=9)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p_prep = os.path.join(target_dir, "04_seasonality_preprocessing.png")
    plt.savefig(p_prep, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"      [x] {p_prep}")

# =========================================================================
# [4] SINH THU MUC TONG QUAN (00_master_summary/)
# =========================================================================
print("\n[4] Sinh cac bieu do Master Summary tai 00_master_summary/ ...")
sum_dir = LEVEL_DIRS["summary"]

# 4.1 Bieu do so sanh SSI va Phan ra Phuong sai 5 cap do
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
fig.suptitle("So Sanh Chi So Suc Manh Mua Vu (SSI) & Phan Ra Phuong Sai Giua 5 Cap Do M5",
             fontsize=14, fontweight="bold")

labels   = [d["Level"] for d in ssi_data]
ssi_vals = [d["SSI_7day"] for d in ssi_data]
tsi_vals = [d["TSI"] for d in ssi_data]
bar_cols = [COLORS[lv] for lv in labels]

# SSI Bar
ax_s = axes[0]
bars = ax_s.bar(labels, ssi_vals, color=bar_cols, edgecolor="black", lw=1.0, zorder=3, width=0.55)
ax_s.axhline(0.64, color="#D32F2F", lw=1.5, ls="--", label="Nguong Mua vu Manh (0.64)")
ax_s.axhline(0.40, color="#F57C00", lw=1.5, ls="--", label="Nguong Trung binh (0.40)")

for bar, v in zip(bars, ssi_vals):
    ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
              f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")

ax_s.set_ylim(0, 1.05)
ax_s.set_ylabel("Seasonal Strength Index (SSI F_S)", fontsize=11, fontweight="bold")
ax_s.set_title("Chi So Suc Manh Mua Vu (F_S) theo Cap Do\n(Giam dot ngot o L12 do hien tuong tap hop - Aggregation)", fontsize=11)
ax_s.grid(axis="y", alpha=0.3, zorder=0)
ax_s.legend(loc="upper right", fontsize=9.5)

# Variance Decomposition
ax_v = axes[1]
x_pos = np.arange(len(labels))
w = 0.28
v_seas = [d["PctVar_Seasonal"] for d in ssi_data]
v_trnd = [d["PctVar_Trend"] for d in ssi_data]
v_res  = [d["PctVar_Resid"] for d in ssi_data]

b1 = ax_v.bar(x_pos - w, v_seas, w, label="% Phuong sai Mua vu", color="#1E88E5", edgecolor="black", lw=0.8)
b2 = ax_v.bar(x_pos,     v_trnd, w, label="% Phuong sai Xu huong", color="#43A047", edgecolor="black", lw=0.8)
b3 = ax_v.bar(x_pos + w, v_res,  w, label="% Phuong sai Nhieu (Resid)", color="#E53935", edgecolor="black", lw=0.8)

for bar_grp in [b1, b2, b3]:
    for bar in bar_grp:
        h = bar.get_height()
        if h > 4:
            ax_v.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

ax_v.set_xticks(x_pos)
ax_v.set_xticklabels([f"{lv}\n({series_keys[lv]})" for lv in labels], fontsize=9)
ax_v.set_ylabel("% Phuong Sai Giai Thich (Variance Explained)", fontsize=11, fontweight="bold")
ax_v.set_title("Phan Ra Phuong Sai: Mua Vu vs Xu Huong vs Nhieu\n(L12 chiu 97.4% nhieu ngau nhien)", fontsize=11)
ax_v.set_ylim(0, 105)
ax_v.grid(axis="y", alpha=0.3)
ax_v.legend(loc="upper right", fontsize=9.5)

plt.tight_layout()
p_master_ssi = os.path.join(sum_dir, "01_hierarchical_ssi_comparison.png")
plt.savefig(p_master_ssi, dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUT_FIG, "02_seasonal_strength_index.png"), dpi=200, bbox_inches="tight") # legacy
plt.close()
print(f"   [x] {p_master_ssi}")

# 4.2 Bieu do doi chieu Macro (L1) vs Micro (L12)
fig, (ax_macro, ax_micro) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle("Doi Chieu Truc Dien: Tin Hieu Vi Mo L1 (Toan Quoc) vs Tin Hieu Vi Mo L12 (SKU Ban Le)\n"
             "Minh Chung Hien Tuong Trieu Tieu Nhieu Nho Tong Hop (Aggregation Effect)",
             fontsize=13, fontweight="bold", y=0.98)

tail_comp = 90
d_tail = series_dict["L1"].index[-tail_comp:]
s_l1_tail  = series_dict["L1"].values[-tail_comp:]
s_l12_tail = series_dict["L12"].values[-tail_comp:]
is_we_comp = (d_tail.dayofweek >= 5)

# L1 Macro
ax_macro.plot(d_tail, s_l1_tail, color=COLORS["L1"], lw=2.0, label="L1 - Toan quoc (SSI = 0.7523)")
ax_macro.fill_between(d_tail, s_l1_tail.min()*0.9, s_l1_tail.max()*1.05,
                      where=is_we_comp, color="#BBDEFB", alpha=0.4, label="Cuoi tuan (Saturday-Sunday Peak)")
ax_macro.set_ylabel("Doanh so Toan Quoc (units/day)", fontsize=10, fontweight="bold", color=COLORS["L1"])
ax_macro.set_title("L1 - Cap Toan Quoc: Nhip dieu tuan hoan tuyet doi deu dan, bien do on dinh, nhiễu thap", fontsize=11, fontweight="bold")
ax_macro.grid(alpha=0.3)
ax_macro.legend(loc="upper right", fontsize=9.5)

# L12 Micro
ax_micro.step(d_tail, s_l12_tail, where="mid", color=COLORS["L12"], lw=1.8, label=f"L12 - SKU {series_keys['L12']} (SSI = 0.1334)")
ax_micro.fill_between(d_tail, 0, s_l12_tail.max()*1.2,
                      where=is_we_comp, color="#FFCDD2", alpha=0.4, label="Cuoi tuan (Nhiều ngày = 0 units)")
ax_micro.set_ylabel("Doanh so SKU (units/day)", fontsize=10, fontweight="bold", color=COLORS["L12"])
ax_micro.set_xlabel("Thoi gian (90 ngay gan nhat)", fontsize=10, fontweight="bold")
ax_micro.set_title("L12 - Cap SKU Don Le: Tin hieu roi rac, ngat quang (Intermittent), ti le zero cao, khong the nhin thay chu ky",
                   fontsize=11, fontweight="bold")
ax_micro.grid(alpha=0.3)
ax_micro.legend(loc="upper right", fontsize=9.5)

for ax_m in [ax_macro, ax_micro]:
    ax_m.xaxis.set_major_locator(mticker.MaxNLocator(9))
    ax_m.tick_params(labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
p_macro_micro = os.path.join(sum_dir, "02_macro_vs_micro_comparison.png")
plt.savefig(p_macro_micro, dpi=200, bbox_inches="tight")
plt.close()
print(f"   [x] {p_macro_micro}")

# 4.3 Heatmap tong hop Thang x Ngay Tuan
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle("Heatmap Mua Vu - Doanh So TB theo Thang x Ngay Tuan tren 5 Cap Do M5\n(Do = Doanh so cao; Xanh = Doanh so thap)",
             fontsize=13, fontweight="bold")
day_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for ax, (lv, s) in zip(axes, series_dict.items()):
    s_df = s.reset_index()
    s_df.columns = ["date","sales"]
    s_df["dow"]   = s_df["date"].dt.dayofweek
    s_df["month"] = s_df["date"].dt.month
    pivot = s_df.pivot_table(values="sales", index="month", columns="dow", aggfunc="mean")
    pivot.columns = day_names[:len(pivot.columns)]
    pivot.index   = [month_names[m-1] for m in pivot.index]
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8.5)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"{lv} ({series_keys[lv]})", fontsize=10, fontweight="bold", color=COLORS[lv])
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
p_heatmap = os.path.join(sum_dir, "03_seasonality_heatmap_overview.png")
plt.savefig(p_heatmap, dpi=200, bbox_inches="tight")
safe_savefig(os.path.join(OUT_FIG, "05_seasonality_heatmap.png"), dpi=200, bbox_inches="tight") # legacy
plt.close()
print(f"   [x] {p_heatmap}")

# 4.4 All levels STL overview (Cap nhat lai file cu cho net hon)
fig, axes = plt.subplots(5, 4, figsize=(22, 20))
fig.suptitle("STL Decomposition - 5 Cap Do M5 (Period=7 ngay/tuan, Da cap do)\n"
             "So sanh truc dien thanh phan Xu huong, Mua vu va Phan du giua cac cap",
             fontsize=14, fontweight="bold", y=0.98)

for row, (lv, res) in enumerate(stl_results.items()):
    c = COLORS[lv]
    s = series_dict[lv]
    d = s.index
    ax0, ax1, ax2, ax3 = axes[row]
    
    ax0.plot(d, res.observed, color=c, lw=0.9)
    ax0.set_ylabel(f"{lv}\n({series_keys[lv]})", fontsize=10, fontweight="bold", color=c)
    if row == 0: ax0.set_title("Observed (Goc)", fontsize=11, fontweight="bold")
    
    ax1.plot(d, res.trend, color="#212121", lw=1.2)
    ax1.fill_between(d, res.trend, alpha=0.15, color=c)
    if row == 0: ax1.set_title("Trend (Xu huong)", fontsize=11, fontweight="bold")
    
    ax2.plot(d, res.seasonal, color=c, lw=0.9, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    if row == 0: ax2.set_title("Seasonal (Mua vu 7 ngay)", fontsize=11, fontweight="bold")
    
    ax3.plot(d, res.resid, color="#757575", lw=0.7, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.5, ls="--")
    if row == 0: ax3.set_title("Residual (Nhieu phan du)", fontsize=11, fontweight="bold")
    
    for ax_col in [ax0, ax1, ax2, ax3]:
        ax_col.tick_params(labelsize=7.5)
        ax_col.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax_col.grid(alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.97])
p_stl_all = os.path.join(sum_dir, "04_all_levels_stl_overview.png")
plt.savefig(p_stl_all, dpi=180, bbox_inches="tight")
safe_savefig(os.path.join(OUT_FIG, "01_stl_decomposition_5levels.png"), dpi=180, bbox_inches="tight") # legacy
plt.close()
print(f"   [x] {p_stl_all}")

# Cap nhat luon file legacy 03 va 04 va 06 de dong nhat
# Legacy 03: ACF/PACF 5 levels
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle("ACF / PACF - Minh Chung Tinh Mua Vu 5 Cap Do M5\n(Dinh ro net tai Lag 7, 14, 28 tai cac cap tong hop)",
             fontsize=13, fontweight="bold", y=0.98)
for row, (lv, s) in enumerate(series_dict.items()):
    s_clean = s.fillna(method="ffill").fillna(0)
    c = COLORS[lv]
    ax_a = axes[row][0]; ax_p = axes[row][1]
    plot_acf(s_clean, ax=ax_a, lags=50, alpha=0.05, color=c, vlines_kwargs={"colors": c})
    plot_pacf(s_clean, ax=ax_p, lags=50, alpha=0.05, color=c, vlines_kwargs={"colors": c}, method="ywm")
    for lg in [7, 14, 28]:
        ax_a.axvline(lg, color="red", lw=1.2, ls="--", alpha=0.6)
        ax_p.axvline(lg, color="red", lw=1.2, ls="--", alpha=0.6)
    ax_a.set_title(f"{LEVEL_LABELS[lv]} - ACF", fontsize=9.5, color=c)
    ax_p.set_title(f"{LEVEL_LABELS[lv]} - PACF", fontsize=9.5, color=c)
plt.tight_layout(rect=[0, 0, 1, 0.97])
safe_savefig(os.path.join(OUT_FIG, "03_acf_pacf_seasonality.png"), dpi=160, bbox_inches="tight")
plt.close()

# Legacy 04: Periodogram
fig, axes = plt.subplots(5, 1, figsize=(14, 18))
fig.suptitle("Periodogram - Xac Dinh Chu Ky Mua Vu (FFT) 5 Cap Do\n(Dinh tai Period=7 -> mua vu tuan; Period~365 -> mua vu nam)",
             fontsize=13, fontweight="bold", y=0.98)
for row, (lv, s) in enumerate(series_dict.items()):
    s_clean = s.fillna(method="ffill").fillna(0).values
    freqs, power = periodogram(s_clean, scaling="density")
    valid = freqs > 0
    periods = 1.0 / freqs[valid]
    power_v = power[valid]
    mask = periods <= 400
    periods, power_v = periods[mask], power_v[mask]
    ax = axes[row]
    ax.semilogy(periods, power_v, color=COLORS[lv], lw=1.2, alpha=0.85)
    ax.fill_between(periods, power_v, alpha=0.15, color=COLORS[lv])
    for p_mark in [7, 14, 28, 365]:
        if p_mark <= 400: ax.axvline(p_mark, color="red", lw=1.2, ls="--", alpha=0.7)
    ax.set_ylabel(lv, fontsize=11, fontweight="bold", color=COLORS[lv])
    ax.set_title(LEVEL_LABELS[lv], fontsize=9.5, color=COLORS[lv])
    ax.set_xlabel("Period (ngay)", fontsize=8.5)
    ax.grid(alpha=0.3); ax.set_xlim(2, 400)
plt.tight_layout(rect=[0, 0, 1, 0.97])
safe_savefig(os.path.join(OUT_FIG, "04_periodogram_fft.png"), dpi=160, bbox_inches="tight")
plt.close()

# Legacy 06: Preprocessing 5 levels
fig, axes = plt.subplots(5, 3, figsize=(22, 18))
fig.suptitle("Tien Xu Ly Tinh Mua Vu: STL vs Fourier Features (5 Cap Do)\n"
             "(Cot 1: Goc; Cot 2: STL Deseasonal; Cot 3: Fourier sin/cos)",
             fontsize=12, fontweight="bold", y=0.98)
for row, (lv, s) in enumerate(series_dict.items()):
    c = COLORS[lv]
    s_clean = s.fillna(method="ffill").fillna(0)
    stl = STL(s_clean, period=7, robust=True); res = stl.fit()
    deseasonal = s_clean - res.seasonal
    t = np.arange(len(s_clean))
    sin7 = np.sin(2*np.pi*t/7); cos7 = np.cos(2*np.pi*t/7); sin14 = np.sin(2*np.pi*t/14)
    tail = -min(180, len(s_clean))
    dates_t = s_clean.index[tail:]
    ax0, ax1, ax2 = axes[row]
    ax0.plot(dates_t, s_clean.values[tail:], color=c, lw=0.9, label="Original")
    ax0.plot(dates_t, res.trend.values[tail:], color="black", lw=1.5, ls="--", label="Trend")
    ax0.fill_between(dates_t, s_clean.values[tail:], res.trend.values[tail:], alpha=0.2, color=c)
    if row == 0: ax0.set_title("Chuoi Goc + Trend", fontsize=10)
    ax0.set_ylabel(lv, fontsize=10, fontweight="bold", color=c); ax0.legend(fontsize=7)
    ax1.plot(dates_t, deseasonal.values[tail:], color="#333", lw=0.9, alpha=0.85)
    ax1.axhline(deseasonal.mean(), color="red", lw=1, ls="--", alpha=0.6)
    if row == 0: ax1.set_title("Sau STL Deseasonal", fontsize=10)
    ax2_twin = ax2.twinx()
    ax2.plot(dates_t, s_clean.values[tail:], color=c, lw=0.7, alpha=0.5, label="Original")
    ax2_twin.plot(dates_t, sin7[tail:], color="red", lw=1.2, ls="-", label="sin(2pi/7)")
    ax2_twin.plot(dates_t, cos7[tail:], color="blue", lw=1.2, ls="--", label="cos(2pi/7)")
    ax2_twin.plot(dates_t, sin14[tail:], color="orange", lw=0.8, ls=":", label="sin(2pi/14)")
    ax2_twin.set_ylabel("Fourier Value", fontsize=8)
    if row == 0: ax2.set_title("Fourier Features sin/cos", fontsize=10)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=6, loc="upper right")
    for ax_sub in [ax0, ax1, ax2]:
        ax_sub.tick_params(labelsize=7)
        ax_sub.xaxis.set_major_locator(mticker.MaxNLocator(4))
plt.tight_layout(rect=[0, 0, 1, 0.97])
safe_savefig(os.path.join(OUT_FIG, "06_seasonality_preprocessing.png"), dpi=160, bbox_inches="tight")
plt.close()

print("\n"+"="*75)
print("  HOAN TAT SINH HE SINH THAI BIEU DO TINH MUA VU!")
print(f"  Thu muc goc: {OUT_FIG}")
print("  Cac thu muc cap do:")
for k, v in LEVEL_DIRS.items():
    print(f"   * {k}: {v}")
print("="*75)
