"""
Script 04: Phan tich tinh mua vu da cap do M5 (L1, L2, L3, L9, L12)
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
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
OUT_FIG    = os.path.join(BASE_DIR, "outputs", "figures", "seasonality")
OUT_RES    = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

COLORS = {"L1":"#2196F3","L2":"#4CAF50","L3":"#FF9800","L9":"#9C27B0","L12":"#F44336"}
LEVEL_LABELS = {
    "L1":"L1 - Toan quoc","L2":"L2 - Bang","L3":"L3 - Cua hang",
    "L9":"L9 - DanhMuc x Bang","L12":"L12 - SKU"
}

print("="*60)
print("  PHAN TICH TINH MUA VU DA CAP DO M5")
print("="*60)

print("\n[1] Load du lieu va tong hop 5 cap do...")
df_sku = pd.read_parquet(os.path.join(DATA_PROC, "m5_sku_level_features.parquet"))
df_sku["date"] = pd.to_datetime(df_sku["date"])
df_base = df_sku[["date","d_num","item_id","dept_id","cat_id","store_id","state_id","sales"]].copy()

daily_L1  = df_base.groupby("date")["sales"].sum().reset_index()
daily_L2  = df_base.groupby(["date","state_id"])["sales"].sum().reset_index().rename(columns={"state_id":"key"})
daily_L3  = df_base.groupby(["date","store_id"])["sales"].sum().reset_index().rename(columns={"store_id":"key"})
daily_L9  = df_base.groupby(["date","state_id","cat_id"])["sales"].sum().reset_index()
daily_L9["key"] = daily_L9["state_id"]+"_"+daily_L9["cat_id"]
daily_L12 = df_base.groupby(["date","item_id","store_id"])["sales"].sum().reset_index()
daily_L12["key"] = daily_L12["item_id"]+"_"+daily_L12["store_id"]

def pick_series(df_level, key_col="key"):
    if key_col not in df_level.columns:
        return df_level.set_index("date")["sales"].sort_index()
    best_key = df_level.groupby(key_col)["sales"].count().idxmax()
    s = df_level[df_level[key_col]==best_key].set_index("date")["sales"].sort_index()
    print(f"   -> Key dai dien: {best_key} ({len(s)} ngay)")
    return s

series_dict = {
    "L1": pick_series(daily_L1),
    "L2": pick_series(daily_L2),
    "L3": pick_series(daily_L3),
    "L9": pick_series(daily_L9),
    "L12":pick_series(daily_L12),
}

for k, s in series_dict.items():
    print(f"   {k}: {len(s)} ngay | mean={s.mean():.2f} | std={s.std():.2f}")

# --- STL Decomposition ---
print("\n[2] STL Decomposition (period=7)...")
stl_results = {}
for lv, s in series_dict.items():
    s_clean = s.fillna(method="ffill").fillna(0)
    stl = STL(s_clean, period=7, robust=True)
    stl_results[lv] = stl.fit()

fig, axes = plt.subplots(5, 4, figsize=(22, 20))
fig.suptitle("STL Decomposition - 5 Cap Do M5 (Seasonal period=7 ngay/tuan)",
             fontsize=14, fontweight="bold", y=0.98)
for row, (lv, res) in enumerate(stl_results.items()):
    color = COLORS[lv]
    s = series_dict[lv]
    dates = s.index
    ax0,ax1,ax2,ax3 = axes[row]
    ax0.plot(dates, res.observed, color=color, lw=0.8)
    ax0.set_ylabel(lv, fontsize=11, fontweight="bold", color=color)
    if row==0: ax0.set_title("Observed", fontsize=11)
    ax1.plot(dates, res.trend, color="#333", lw=1.2)
    ax1.fill_between(dates, res.trend, alpha=0.15, color=color)
    if row==0: ax1.set_title("Trend", fontsize=11)
    ax2.plot(dates, res.seasonal, color=color, lw=0.8, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    if row==0: ax2.set_title("Seasonal (7-day)", fontsize=11)
    ax3.plot(dates, res.resid, color="#888", lw=0.6, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.5, ls="--")
    if row==0: ax3.set_title("Residual", fontsize=11)
    for ax in [ax0,ax1,ax2,ax3]:
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(OUT_FIG,"01_stl_decomposition_5levels.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 01_stl_decomposition_5levels.png")

# --- Seasonal Strength Index ---
print("\n[3] Tinh Seasonal Strength Index (SSI)...")
def seasonal_strength(res):
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    return max(0, 1 - var_r/var_sr)
def trend_strength(res):
    var_r = np.var(res.resid)
    var_tr = np.var(res.trend + res.resid)
    return max(0, 1 - var_r/var_tr)

ssi_data = []
for lv, res in stl_results.items():
    ssi = seasonal_strength(res)
    ts  = trend_strength(res)
    vsp = np.var(res.seasonal)/np.var(res.observed)*100
    vtp = np.var(res.trend)/np.var(res.observed)*100
    judge = "Manh" if ssi>0.64 else ("Trung binh" if ssi>0.40 else "Yeu")
    ssi_data.append({"Level":lv,"SSI_7day":round(ssi,4),"TSI":round(ts,4),
                     "PctVar_Seasonal":round(vsp,2),"PctVar_Trend":round(vtp,2),"Nhan_xet":judge})
    print(f"   {lv}: SSI={ssi:.4f} ({judge}), TSI={ts:.4f}")

df_ssi = pd.DataFrame(ssi_data)

# SSI Bar Chart
fig, axes = plt.subplots(1,2, figsize=(15,6))
fig.suptitle("Chi So Suc Manh Mua Vu (SSI) - 5 Cap Do M5", fontsize=14, fontweight="bold")
labels = list(COLORS.keys())
ssi_vals = [d["SSI_7day"] for d in ssi_data]
tsi_vals = [d["TSI"] for d in ssi_data]
colors   = [COLORS[lv] for lv in labels]
ax = axes[0]
bars = ax.bar(labels, ssi_vals, color=colors, edgecolor="white", lw=1.5, zorder=3)
ax.axhline(0.64, color="red",   lw=1.5, ls="--", label="Nguong Manh (0.64)")
ax.axhline(0.40, color="orange",lw=1.5, ls="--", label="Nguong TB (0.40)")
for bar, v in zip(bars, ssi_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0,1.1); ax.set_ylabel("SSI"); ax.legend(fontsize=9)
ax.set_title("SSI theo Cap Do (7-day seasonality)", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax2 = axes[1]
x = np.arange(5); w = 0.35
ax2.bar(x-w/2, [d["PctVar_Seasonal"] for d in ssi_data], w, label="% Seasonal", color=colors, alpha=0.85, edgecolor="white")
ax2.bar(x+w/2, [d["PctVar_Trend"]    for d in ssi_data], w, label="% Trend",    color=colors, alpha=0.45, edgecolor="white", hatch="//")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel("% Variance Explained"); ax2.legend(fontsize=9)
ax2.set_title("Phan Ra Phuong Sai theo Thanh Phan", fontsize=11)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG,"02_seasonal_strength_index.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 02_seasonal_strength_index.png")

# --- ACF/PACF ---
print("\n[4] Ve ACF/PACF...")
fig, axes = plt.subplots(5,2, figsize=(16,20))
fig.suptitle("ACF / PACF - Minh Chung Tinh Mua Vu 5 Cap Do M5\n(Dinh tai lag 7,14,28 -> mua vu tuan)",
             fontsize=13, fontweight="bold", y=0.98)
for row,(lv,s) in enumerate(series_dict.items()):
    s_clean = s.fillna(method="ffill").fillna(0)
    color = COLORS[lv]
    ax_acf  = axes[row][0]
    ax_pacf = axes[row][1]
    plot_acf( s_clean, ax=ax_acf,  lags=60, alpha=0.05, color=color, vlines_kwargs={"colors":color})
    plot_pacf(s_clean, ax=ax_pacf, lags=60, alpha=0.05, color=color, vlines_kwargs={"colors":color}, method="ywm")
    for lag in [7,14,28,35]:
        ax_acf.axvline( lag, color="red", lw=1.2, ls="--", alpha=0.6)
        ax_pacf.axvline(lag, color="red", lw=1.2, ls="--", alpha=0.6)
    ax_acf.set_title(f"{LEVEL_LABELS[lv]} - ACF",  fontsize=9.5, color=color)
    ax_pacf.set_title(f"{LEVEL_LABELS[lv]} - PACF", fontsize=9.5, color=color)
    ax_acf.set_xlabel("Lag (ngay)", fontsize=8)
    ax_pacf.set_xlabel("Lag (ngay)", fontsize=8)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(OUT_FIG,"03_acf_pacf_seasonality.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 03_acf_pacf_seasonality.png")

# --- Periodogram ---
print("\n[5] Ve Periodogram FFT...")
fig, axes = plt.subplots(5,1, figsize=(14,18), sharex=False)
fig.suptitle("Periodogram - Xac Dinh Chu Ky Mua Vu (FFT)\n(Dinh tai period=7 -> mua vu tuan; period~365 -> mua vu nam)",
             fontsize=13, fontweight="bold", y=0.98)
for row,(lv,s) in enumerate(series_dict.items()):
    s_clean = s.fillna(method="ffill").fillna(0).values
    freqs, power = periodogram(s_clean, scaling="density")
    valid = freqs > 0
    periods = 1/freqs[valid]
    power_v = power[valid]
    mask = periods <= 400
    periods, power_v = periods[mask], power_v[mask]
    ax = axes[row]
    ax.semilogy(periods, power_v, color=COLORS[lv], lw=1.2, alpha=0.85)
    ax.fill_between(periods, power_v, alpha=0.15, color=COLORS[lv])
    for p_mark in [7,14,28,365]:
        if p_mark<=400:
            ax.axvline(p_mark, color="red", lw=1.2, ls="--", alpha=0.7)
    ax.set_ylabel(lv, fontsize=11, fontweight="bold", color=COLORS[lv])
    ax.set_title(LEVEL_LABELS[lv], fontsize=9, color=COLORS[lv])
    ax.set_xlabel("Period (ngay)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(2,400)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(OUT_FIG,"04_periodogram_fft.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 04_periodogram_fft.png")

# --- Heatmap Mua vu ---
print("\n[6] Ve Heatmap mua vu...")
fig, axes = plt.subplots(1,5, figsize=(20,5))
fig.suptitle("Heatmap Mua Vu - Doanh So TB theo Thang x Ngay Tuan\n(Do = cao; Xanh = thap)",
             fontsize=13, fontweight="bold")
day_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for ax,(lv,s) in zip(axes, series_dict.items()):
    s_df = s.reset_index()
    s_df.columns = ["date","sales"]
    s_df["dow"]   = s_df["date"].dt.dayofweek
    s_df["month"] = s_df["date"].dt.month
    pivot = s_df.pivot_table(values="sales", index="month", columns="dow", aggfunc="mean")
    pivot.columns = day_names[:len(pivot.columns)]
    pivot.index   = [month_names[m-1] for m in pivot.index]
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(f"{lv}", fontsize=10, fontweight="bold", color=COLORS[lv])
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG,"05_seasonality_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 05_seasonality_heatmap.png")

# --- Tien xu ly ---
print("\n[7] Demo tien xu ly mua vu (STL deseasonal + Fourier)...")
fig, axes = plt.subplots(5,3, figsize=(22,18))
fig.suptitle("Tien Xu Ly Tinh Mua Vu: STL vs Fourier Features\n"
             "(Cot 1: Goc; Cot 2: STL Deseasonal; Cot 3: Fourier sin/cos)",
             fontsize=12, fontweight="bold", y=0.98)
for row,(lv,s) in enumerate(series_dict.items()):
    color = COLORS[lv]
    s_clean = s.fillna(method="ffill").fillna(0)
    stl = STL(s_clean, period=7, robust=True); res = stl.fit()
    deseasonal = s_clean - res.seasonal
    t = np.arange(len(s_clean))
    sin7  = np.sin(2*np.pi*t/7)
    cos7  = np.cos(2*np.pi*t/7)
    sin14 = np.sin(2*np.pi*t/14)
    tail  = -min(200,len(s_clean))
    dates = s_clean.index[tail:]
    ax0,ax1,ax2 = axes[row]
    ax0.plot(dates, s_clean.values[tail:], color=color, lw=0.9, label="Original")
    ax0.plot(dates, res.trend.values[tail:], color="black", lw=1.5, ls="--", label="Trend")
    ax0.fill_between(dates, s_clean.values[tail:], res.trend.values[tail:], alpha=0.2, color=color)
    if row==0: ax0.set_title("Chuoi Goc + Trend", fontsize=10)
    ax0.set_ylabel(lv, fontsize=10, fontweight="bold", color=color)
    ax0.legend(fontsize=7)
    ax1.plot(dates, deseasonal.values[tail:], color="#333", lw=0.9, alpha=0.85)
    ax1.axhline(deseasonal.mean(), color="red", lw=1, ls="--", alpha=0.6)
    if row==0: ax1.set_title("Sau STL Deseasonal", fontsize=10)
    ax2_twin = ax2.twinx()
    ax2.plot(dates, s_clean.values[tail:], color=color, lw=0.7, alpha=0.5, label="Original")
    ax2_twin.plot(dates, sin7[tail:],  color="red",    lw=1.2, ls="-",  label="sin(2pi/7)")
    ax2_twin.plot(dates, cos7[tail:],  color="blue",   lw=1.2, ls="--", label="cos(2pi/7)")
    ax2_twin.plot(dates, sin14[tail:], color="orange", lw=0.8, ls=":",  label="sin(2pi/14)")
    ax2_twin.set_ylabel("Fourier Value", fontsize=8)
    if row==0: ax2.set_title("Fourier Features sin/cos", fontsize=10)
    lines1,labels1 = ax2.get_legend_handles_labels()
    lines2,labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=6, loc="upper right")
    for ax in [ax0,ax1,ax2]:
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(OUT_FIG,"06_seasonality_preprocessing.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 06_seasonality_preprocessing.png")

df_ssi.to_csv(os.path.join(OUT_RES,"seasonality_ssi_results.csv"), index=False, encoding="utf-8-sig")
print("\n[DONE] Script 04 hoan tat!")
print(f"  Bieu do luu tai: {OUT_FIG}")
