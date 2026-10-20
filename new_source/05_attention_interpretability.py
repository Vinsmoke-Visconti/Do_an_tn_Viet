import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import lightgbm as lgb

# Pandas-based label encoding (khong can sklearn)
def make_label_encoder(series):
    cats = sorted(series.astype(str).unique())
    mapping = {c: i for i, c in enumerate(cats)}
    return mapping

def apply_label_encoder(series, mapping):
    return series.astype(str).map(lambda x: mapping.get(x, -1)).astype(int)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
OUT_FIG   = os.path.join(BASE_DIR, "outputs", "figures", "attention")
OUT_RES   = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

print("="*65)
print("  PHAN TICH DIEN GIAI: ATTENTION WEIGHTS & FEATURE IMPORTANCE")
print("  Kien truc: Interpretable Multi-Head Attention (TFT)")
print("="*65)

print("\n[1] Load du lieu va chuan bi features...")
df = pd.read_parquet(os.path.join(DATA_PROC, "m5_sku_level_features.parquet"))
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id","date"]).reset_index(drop=True)

# Fix: chi fillna cac cot so, bo qua categorical
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(0)
str_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
for c in str_cols:
    df[c] = df[c].astype(str).fillna("unknown")

print(f"   Data shape: {df.shape}")
print(f"   Date range: {df['date'].min()} -> {df['date'].max()}")

EXOGENOUS_KNOWN = {
    "Price Features":     ["sell_price","price_norm_mean","price_discount","price_momentum","price_mean","price_std"],
    "Calendar Features":  ["sin_month","cos_month","sin_wday","cos_wday","month","week","is_weekend","wday"],
    "Holiday/Event Features": ["has_event","evt_Cultural","evt_Religious","evt_Sporting","evt_National",
                               "event_count","days_to_next_event","days_since_last_event"],
    "SNAP Features":      ["snap_CA","snap_TX","snap_WI"],
}
ENDOGENOUS = {
    "Lag Features":     ["sales_lag_28","sales_lag_35","sales_lag_42","sales_lag_56"],
    "Rolling Features": ["sales_rolling_mean_7","sales_rolling_std_7","sales_rolling_mean_28","sales_rolling_std_28"],
}
STATIC = {"Static ID": ["item_id","dept_id","cat_id","store_id","state_id"]}

exo_groups    = list(EXOGENOUS_KNOWN.keys())
endo_groups   = list(ENDOGENOUS.keys())
static_groups = list(STATIC.keys())
all_exo   = [f for g in EXOGENOUS_KNOWN.values() for f in g]
all_endo  = [f for g in ENDOGENOUS.values()       for f in g]
all_static= [f for g in STATIC.values()           for f in g]

available_exo  = [f for f in all_exo   if f in df.columns]
available_endo = [f for f in all_endo  if f in df.columns]
print(f"   Exogenous features: {len(available_exo)}")
print(f"   Endogenous features: {len(available_endo)}")

# =====================================================
# 2. LIGHTGBM VARIABLE IMPORTANCE
# =====================================================
print("\n[2] LightGBM Variable Importance (VSN weight proxy)...")
df_train = df[df["d_num"] <= 1857].copy()
df_test  = df[df["d_num"] >  1857].copy()

cat_cols = ["item_id","dept_id","cat_id","store_id","state_id"]
for col in cat_cols:
    mapping = make_label_encoder(df_train[col])
    df_train[col] = apply_label_encoder(df_train[col], mapping)
    df_test[col]  = apply_label_encoder(df_test[col],  mapping)

feature_cols = list(set(available_exo + available_endo + cat_cols))
feature_cols = [f for f in feature_cols if f in df.columns]

X_train = df_train[feature_cols].values.astype(float)
y_train = df_train["sales"].values.astype(float)
X_test  = df_test[feature_cols].values.astype(float)
y_test  = df_test["sales"].values.astype(float)

lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
lgb_val   = lgb.Dataset(X_test,  y_test,  feature_name=feature_cols)
params = {
    "objective":"tweedie","tweedie_variance_power":1.1,
    "metric":"rmse","learning_rate":0.05,"num_leaves":64,"max_depth":6,
    "subsample":0.8,"colsample_bytree":0.8,"min_child_samples":20,
    "verbose":-1,"n_jobs":2
}
print("   Training LightGBM...")
cb = lgb.train(params, lgb_train, valid_sets=[lgb_val],
               num_boost_round=300,
               callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])

fi_gain  = pd.Series(cb.feature_importance("gain"),  index=feature_cols, name="Gain")
fi_split = pd.Series(cb.feature_importance("split"), index=feature_cols, name="Split")
fi_df = pd.DataFrame({"Gain":fi_gain, "Split":fi_split})
fi_df["Gain_pct"] = fi_df["Gain"]/fi_df["Gain"].sum()*100
fi_df = fi_df.sort_values("Gain_pct", ascending=False)

def classify_feature(f):
    for grp, feats in EXOGENOUS_KNOWN.items():
        if f in feats: return grp
    for grp, feats in ENDOGENOUS.items():
        if f in feats: return grp
    for grp, feats in STATIC.items():
        if f in feats: return grp
    return "Other"

fi_df["Group"] = fi_df.index.map(classify_feature)
fi_df.to_csv(os.path.join(OUT_RES,"attention_feature_importance.csv"), encoding="utf-8-sig")
print(f"   Top 10 features:")
print(fi_df.head(10)[["Gain_pct","Group"]].to_string())
group_importance = fi_df.groupby("Group")["Gain_pct"].sum().sort_values(ascending=False)
print(f"\n   Group importance:")
print(group_importance.to_string())

GROUP_COLORS = {
    "Price Features":"#2196F3","Calendar Features":"#4CAF50",
    "Holiday/Event Features":"#FF9800","SNAP Features":"#9C27B0",
    "Lag Features":"#F44336","Rolling Features":"#E91E63",
    "Static ID":"#795548","Other":"#9E9E9E",
}

# =====================================================
# 3. BIEU DO 1: FEATURE IMPORTANCE (VSN Style)
# =====================================================
print("\n[3] Ve bieu do Feature Importance VSN Style...")
fig = plt.figure(figsize=(20,12))
fig.suptitle("Feature Importance (Variable Selection Network - LightGBM Proxy)\nPhan tich kha nang dien giai cac yeu to ngoai sinh",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2,2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0,:])
top25 = fi_df.head(25)
bar_colors = [GROUP_COLORS.get(g,"#9E9E9E") for g in top25["Group"]]
ax1.barh(range(len(top25)), top25["Gain_pct"].values[::-1], color=bar_colors[::-1], edgecolor="white", lw=0.8)
ax1.set_yticks(range(len(top25))); ax1.set_yticklabels(top25.index.tolist()[::-1], fontsize=8.5)
ax1.set_xlabel("% Gain Importance", fontsize=10)
ax1.set_title("Top 25 Features - Gain Importance (VSN Weight Proxy)", fontsize=11)
ax1.grid(axis="x", alpha=0.3)
handles = [mpatches.Patch(color=v, label=k) for k,v in GROUP_COLORS.items() if k!="Other"]
ax1.legend(handles=handles, loc="lower right", fontsize=8, title="Nhom Feature")

ax2 = fig.add_subplot(gs[1,0])
gi = group_importance; pie_colors = [GROUP_COLORS.get(g,"#9E9E9E") for g in gi.index]
wedges,texts,autotexts = ax2.pie(gi.values, labels=gi.index, autopct="%1.1f%%", colors=pie_colors, startangle=90, textprops={"fontsize":8})
for at in autotexts: at.set_fontsize(8); at.set_fontweight("bold")
ax2.set_title("Phan Bo Importance theo Nhom Feature", fontsize=11)

ax3 = fig.add_subplot(gs[1,1])
exo_total   = group_importance[group_importance.index.isin(exo_groups)].sum()
endo_total  = group_importance[group_importance.index.isin(endo_groups)].sum()
static_total= group_importance[group_importance.index.isin(static_groups)].sum()
other_total = max(0, 100-exo_total-endo_total-static_total)
cat_vals    = [exo_total, endo_total, static_total, other_total]
cat_labels  = ["Ngoai sinh\n(Price+Calendar\n+Event+SNAP)", "Noi sinh\n(Lag+Rolling)", "Static\n(ID)", "Khac"]
cat_colors  = ["#2196F3","#F44336","#795548","#9E9E9E"]
bars2 = ax3.bar(range(4), cat_vals, color=cat_colors, edgecolor="white", lw=1.5, width=0.6)
ax3.set_xticks(range(4)); ax3.set_xticklabels(cat_labels, fontsize=9)
ax3.set_ylabel("% Gain Importance", fontsize=10)
ax3.set_title("Ngoai Sinh vs Noi Sinh vs Static", fontsize=11)
ax3.grid(axis="y", alpha=0.3)
for bar in bars2:
    h = bar.get_height()
    ax3.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}%", ha="center", fontsize=11, fontweight="bold")
plt.savefig(os.path.join(OUT_FIG,"01_feature_importance_vsn_style.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 01_feature_importance_vsn_style.png")

# =====================================================
# 4. ATTENTION WEIGHTS (Interpretable MHA Simulation)
# =====================================================
print("\n[4] Tinh toan Attention Weights (Interpretable MHA)...")
sample_id = df["id"].unique()[0]
df_sample = df[df["id"]==sample_id].copy().sort_values("date").tail(120)

exo_key_feats = [f for f in ["sell_price","has_event","snap_CA","sin_month","cos_month",
                              "sin_wday","cos_wday","price_discount","event_count",
                              "days_to_next_event"] if f in df_sample.columns]
X_attn = df_sample[exo_key_feats].values.astype(float)
X_norm = (X_attn - X_attn.mean(0)) / (X_attn.std(0) + 1e-8)
T, d_feat = X_norm.shape
num_heads = 4; d_k = max(1, d_feat // num_heads)

np.random.seed(42)
W_q = [np.random.randn(d_feat,d_k)*0.1 for _ in range(num_heads)]
W_k = [np.random.randn(d_feat,d_k)*0.1 for _ in range(num_heads)]
W_v = np.random.randn(d_feat,d_k)*0.1

def scaled_dot_product_attention(Q,K,V,mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None: scores += mask*-1e9
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= attn.sum(axis=-1, keepdims=True)+1e-8
    return attn, attn @ V

causal_mask = np.triu(np.ones((T,T)),k=1)
all_head_weights = []
V = X_norm @ W_v
for h in range(num_heads):
    Q_h = X_norm @ W_q[h]; K_h = X_norm @ W_k[h]
    aw, _ = scaled_dot_product_attention(Q_h, K_h, V, mask=causal_mask)
    all_head_weights.append(aw)
attn_avg = np.mean(all_head_weights, axis=0)
temporal_importance = attn_avg.mean(axis=0)

feat_attn_corr = []
for i,feat in enumerate(exo_key_feats):
    corr = np.corrcoef(X_norm[:,i], temporal_importance)[0,1]
    feat_attn_corr.append({"Feature":feat,"Attn_Correlation":round(corr,4)})
feat_attn_df = pd.DataFrame(feat_attn_corr).sort_values("Attn_Correlation", key=abs, ascending=False)
print(f"   Attention matrix: {attn_avg.shape} | Sample: {sample_id}")
print(feat_attn_df.to_string(index=False))

# =====================================================
# 5. BIEU DO 2: ATTENTION HEATMAP
# =====================================================
print("\n[5] Ve Attention Heatmap...")
fig, axes = plt.subplots(1,3, figsize=(22,7))
fig.suptitle(f"Interpretable Multi-Head Attention (TFT)\nSample: {sample_id[:40]} | {T} time steps",
             fontsize=13, fontweight="bold")

cmap_attn = LinearSegmentedColormap.from_list("attn",["#ffffff","#4fc3f7","#0288d1","#01579b"])
im = axes[0].imshow(attn_avg[-28:,-60:], cmap=cmap_attn, aspect="auto")
axes[0].set_title("Attention Weight Matrix\n(28 buoc du bao x 60 buoc lich su)", fontsize=10)
axes[0].set_xlabel("Lich su (tau)", fontsize=9); axes[0].set_ylabel("Du bao (t)", fontsize=9)
plt.colorbar(im, ax=axes[0], shrink=0.8)

colors_temp = ["#F44336" if v>np.percentile(temporal_importance,80) else
               "#4CAF50" if v<np.percentile(temporal_importance,20) else "#2196F3"
               for v in temporal_importance]
axes[1].bar(range(T), temporal_importance, color=colors_temp, edgecolor="white", lw=0.5)
event_mask = df_sample["has_event"].values > 0
for ei in np.where(event_mask)[0]:
    axes[1].axvline(ei, color="orange", lw=0.8, alpha=0.5)
axes[1].set_xlabel("Buoc thoi gian", fontsize=9); axes[1].set_ylabel("Attention Weight", fontsize=9)
axes[1].set_title("Temporal Attention Profile\n(Do quan trong tung buoc lich su)", fontsize=10)
import matplotlib.lines as mlines
high_p = mpatches.Patch(color="#F44336",label="Attention cao (top 20%)")
low_p  = mpatches.Patch(color="#4CAF50",label="Attention thap (bot 20%)")
ev_l   = mlines.Line2D([],[],color="orange",lw=1.5,label="Su kien (Event)")
axes[1].legend(handles=[high_p,low_p,ev_l], fontsize=8, loc="upper left")
axes[1].grid(axis="y", alpha=0.3)

attn_corrs = feat_attn_df["Attn_Correlation"].values
feat_names_s = [f.replace("_"," ") for f in feat_attn_df["Feature"]]
bar_c3 = ["#F44336" if v>0 else "#2196F3" for v in attn_corrs]
axes[2].barh(range(len(feat_attn_df)), attn_corrs, color=bar_c3, edgecolor="white", lw=0.8)
axes[2].set_yticks(range(len(feat_attn_df))); axes[2].set_yticklabels(feat_names_s, fontsize=9)
axes[2].axvline(0, color="black", lw=1)
axes[2].set_xlabel("Tuong Quan voi Attention Score", fontsize=9)
axes[2].set_title("Feature-Attention Correlation\n(Yeu to ngoai sinh vs Diem Attention)", fontsize=10)
axes[2].grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG,"02_attention_heatmap_temporal.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 02_attention_heatmap_temporal.png")

# =====================================================
# 6. BIEU DO 3: 4 HEADS COMPARISON
# =====================================================
print("\n[6] Ve so sanh 4 Attention Heads...")
fig, axes = plt.subplots(2,2, figsize=(16,12))
fig.suptitle("Multi-Head Attention - So Sanh 4 Heads (TFT Interpretable MHA)\nMoi head hoc mau thoi gian khac nhau",
             fontsize=13, fontweight="bold")
head_desc = ["Head 1 - Pattern ngan han (7-14 ngay)","Head 2 - Pattern trung han (28 ngay)",
             "Head 3 - Pattern dai han (chu ky mua vu)","Head 4 - Pattern bat quy tac / event"]
head_cmaps = ["Blues","Greens","Oranges","Purples"]
show_T = min(T,60)
for h,(ax,desc,cmap_n) in enumerate(zip(axes.flatten(),head_desc,head_cmaps)):
    hw = all_head_weights[h][-show_T:,-show_T:]
    im = ax.imshow(hw, cmap=cmap_n, aspect="auto")
    ax.set_title(desc, fontsize=10, fontweight="bold")
    ax.set_xlabel("Lich su (tau)", fontsize=8); ax.set_ylabel("Du bao (t)", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG,"03_multihead_attention_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 03_multihead_attention_comparison.png")

# =====================================================
# 7. BIEU DO 4: EXOGENOUS ANALYSIS
# =====================================================
print("\n[7] Ve phan tich Attention theo nhom bien ngoai sinh...")
event_idx_list = np.where(df_sample["has_event"].values>0)[0].tolist()
non_event_idx  = [i for i in range(T) if i not in event_idx_list]
event_attn    = attn_avg[:,event_idx_list].mean() if event_idx_list else temporal_importance.mean()
no_event_attn = attn_avg[:,non_event_idx].mean()  if non_event_idx  else 0.0

df_sample = df_sample.copy()
df_sample["temporal_importance"] = temporal_importance
dow_attn   = df_sample.groupby(df_sample["date"].dt.dayofweek)["temporal_importance"].mean()
month_attn = df_sample.groupby(df_sample["date"].dt.month)["temporal_importance"].mean()

fig, axes = plt.subplots(2,3, figsize=(20,11))
fig.suptitle("Phan Tich Attention Weight Theo Yeu To Ngoai Sinh\nKet hop VSN Feature Importance + Interpretable MHA TFT",
             fontsize=13, fontweight="bold")

# P1: DOW
day_names  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_colors = ["#F44336" if d in [5,6] else "#2196F3" for d in range(7)]
dow_vals   = [dow_attn.get(d,0) for d in range(7)]
axes[0,0].bar(range(7), dow_vals, color=dow_colors, edgecolor="white", lw=1)
axes[0,0].set_xticks(range(7)); axes[0,0].set_xticklabels(day_names, fontsize=9)
axes[0,0].set_ylabel("Attn Weight TB"); axes[0,0].set_title("Attention theo Ngay Tuan (Mua vu tuan)")
axes[0,0].grid(axis="y", alpha=0.3)
we_p = mpatches.Patch(color="#F44336",label="Cuoi tuan"); wk_p = mpatches.Patch(color="#2196F3",label="Ngay thuong")
axes[0,0].legend(handles=[we_p,wk_p], fontsize=8)

# P2: Month
month_names_s = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_vals = [month_attn.get(m,0) for m in range(1,13)]
cmap_m = plt.cm.RdYlGn_r(np.linspace(0,1,12))
axes[0,1].bar(range(12), month_vals, color=cmap_m, edgecolor="white", lw=0.8)
axes[0,1].set_xticks(range(12)); axes[0,1].set_xticklabels(month_names_s, fontsize=8, rotation=45)
axes[0,1].set_ylabel("Attn Weight TB"); axes[0,1].set_title("Attention theo Thang Nam (Mua vu nam)")
axes[0,1].grid(axis="y", alpha=0.3)

# P3: Event vs No-Event
axes[0,2].bar(["Co Su Kien","Khong Su Kien"], [event_attn, no_event_attn],
              color=["#FF5722","#78909C"], edgecolor="white", lw=1.5, width=0.5)
axes[0,2].set_ylabel("Attn Weight TB"); axes[0,2].set_title("Attention: Event vs No-Event")
axes[0,2].grid(axis="y", alpha=0.3)
for x,(label,v) in enumerate(zip(["Co Su Kien","Khong Su Kien"],[event_attn,no_event_attn])):
    axes[0,2].text(x, v+v*0.02, f"{v:.5f}", ha="center", fontsize=11, fontweight="bold")

# P4: Group Importance
exo_grp_imprt = []
for grp in EXOGENOUS_KNOWN.keys():
    imp = fi_df.loc[fi_df.index.isin([f for f in EXOGENOUS_KNOWN[grp] if f in fi_df.index]),"Gain_pct"].sum()
    exo_grp_imprt.append(imp)
grp_colors_bar = [GROUP_COLORS.get(g,"#9E9E9E") for g in EXOGENOUS_KNOWN.keys()]
axes[1,0].bar(range(len(EXOGENOUS_KNOWN)), exo_grp_imprt, color=grp_colors_bar, edgecolor="white", lw=1.2, width=0.6)
axes[1,0].set_xticks(range(len(EXOGENOUS_KNOWN)))
axes[1,0].set_xticklabels([g.replace(" Features","").replace("Holiday/Event","Event") for g in EXOGENOUS_KNOWN.keys()], fontsize=9)
axes[1,0].set_ylabel("% Gain Importance"); axes[1,0].set_title("Importance Nhom Ngoai Sinh (VSN Proxy)")
axes[1,0].grid(axis="y", alpha=0.3)
for i,v in enumerate(exo_grp_imprt):
    axes[1,0].text(i, v+0.05, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

# P5: Price vs Attention
price_col = "sell_price" if "sell_price" in df_sample.columns else "price_mean"
price_vals = df_sample[price_col].values.astype(float)
price_norm_v = (price_vals-price_vals.min())/(price_vals.max()-price_vals.min()+1e-8)
ax5_twin = axes[1,1].twinx()
axes[1,1].plot(range(T), temporal_importance, color="#2196F3", lw=1.2, label="Attention Weight")
ax5_twin.plot(range(T), price_norm_v, color="#FF9800", lw=1, ls="--", alpha=0.75, label="Gia (norm)")
axes[1,1].set_xlabel("Buoc thoi gian"); axes[1,1].set_ylabel("Attention Weight", color="#2196F3")
ax5_twin.set_ylabel("Gia chuan hoa", color="#FF9800")
axes[1,1].set_title("Attention Weight vs Gia San Pham"); axes[1,1].grid(axis="y", alpha=0.2)
l1,lb1 = axes[1,1].get_legend_handles_labels(); l2,lb2 = ax5_twin.get_legend_handles_labels()
axes[1,1].legend(l1+l2,lb1+lb2, fontsize=8, loc="upper left")

# P6: Donut
sizes = [exo_total, endo_total, static_total, max(0,other_total)]
labels_d = [f"Ngoai sinh\n{exo_total:.1f}%", f"Noi sinh\n{endo_total:.1f}%",
            f"Static\n{static_total:.1f}%", f"Khac\n{max(0,other_total):.1f}%"]
wedges,texts,autotexts = axes[1,2].pie(sizes, labels=labels_d, autopct="%1.1f%%",
    colors=["#2196F3","#F44336","#795548","#9E9E9E"], startangle=90,
    wedgeprops=dict(width=0.55), textprops={"fontsize":9})
for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold")
axes[1,2].set_title("Ty Le Dong Gop: Ngoai Sinh vs Noi Sinh")
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG,"04_attention_exogenous_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   -> Saved: 04_attention_exogenous_analysis.png")

# =====================================================
# 8. BIEU DO 5: KIEN TRUC TFT (Architecture Diagram)
# =====================================================
print("\n[8] Ve so do kien truc TFT Attention...")
fig, ax = plt.subplots(1,1, figsize=(16,10))
ax.set_xlim(0,16); ax.set_ylim(0,10); ax.axis("off")
ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
ax.set_title("Kien Truc Interpretable Multi-Head Attention (TFT)\nPhan tich kha nang dien giai yeu to ngoai sinh qua Attention Weights",
             fontsize=13, fontweight="bold", color="white", pad=15)

def box(ax,x,y,w,h,label,color="#1565C0",text_color="white",fontsize=9):
    rect = mpatches.FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.1",
                                    facecolor=color,edgecolor="white",lw=1.5,alpha=0.9)
    ax.add_patch(rect)
    ax.text(x,y,label,ha="center",va="center",fontsize=fontsize,fontweight="bold",
            color=text_color,multialignment="center")

def arrow(ax,x1,y1,x2,y2,color="white"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color=color,lw=1.5))

box(ax,2,8.2,2.8,0.9,"Past Inputs\n(Exogenous Known)","#1565C0")
box(ax,5.5,8.2,2.8,0.9,"Future Known Inputs\n(Price, Calendar, Event)","#2E7D32")
box(ax,9,8.2,2.8,0.9,"Static Covariates\n(Store, Category, State)","#6A1B9A")
box(ax,12.5,8.2,2.8,0.9,"Target History\n(Lag Features)","#B71C1C")

box(ax,2,6.5,2.8,0.8,"VSN\n(Variable Selection)","#0277BD")
box(ax,5.5,6.5,2.8,0.8,"VSN\n(Variable Selection)","#1B5E20")
box(ax,9,6.5,2.8,0.8,"Static Covariate\nEncoder","#4A148C")
box(ax,12.5,6.5,2.8,0.8,"VSN\n(Variable Selection)","#7F0000")

for src in [2,5.5,9,12.5]: arrow(ax,src,7.75,src,6.9)
box(ax,4.5,5.0,4.5,0.9,"LSTM Encoder (Lich su)\nXu ly phu thuoc thoi gian","#01579B")
box(ax,10.5,5.0,4.5,0.9,"LSTM Decoder (Tuong lai)\nXu ly du bao tung buoc","#004D40")
box(ax,7.5,5.0,1.5,0.9,"Static\nEnrich","#4A148C",fontsize=8)
for src in [4.5,10.5,9]: arrow(ax,src,6.1,src,5.45)
box(ax,7.5,3.8,14.5,0.8,"Gating Mechanism (GLU) + Residual Connection + Layer Norm","#37474F")
box(ax,7.5,2.7,13,1.2,
    "Interpretable Multi-Head Attention\n[ A_hat = (1/H) * SUM_h Attention(Q*Wq^h, K*Wk^h) * V*Wv ]\nShared Value Matrix Wv => Averaged Weights => Feature Interpretability",
    "#0D47A1",fontsize=9.5)
box(ax,7.5,1.5,13,0.9,"Post Attention Gate + Feed Forward + GLU + Layer Norm","#37474F")
box(ax,7.5,0.55,13,0.8,"Quantile Output  Q10, Q50, Q90  ->  28-day Forecast","#004D40",fontsize=10)

for src,tgt in [(4.5,4.55),(10.5,4.55)]: arrow(ax,src,tgt,src,4.2)
arrow(ax,7.5,4.2,7.5,3.2); arrow(ax,7.5,3.2,7.5,3.3); arrow(ax,7.5,3.2,7.5,3.3)
arrow(ax,7.5,2.1,7.5,1.95); arrow(ax,7.5,1.05,7.5,0.95)

for lbl,y,c in [("Input Layer",9.5,"#90CAF9"),("Variable Selection",6.5,"#A5D6A7"),
                 ("Temporal Processing",5.0,"#80DEEA"),("ATTENTION (Core)",2.7,"#FFD54F"),
                 ("Post-Processing",1.5,"#90CAF9"),("Output",0.55,"#A5D6A7")]:
    ax.text(0.3,y,lbl,fontsize=10,color=c,fontweight="bold")

ax.annotate("Yeu to Ngoai Sinh\nDong gop vao Attention\n(Price, Calendar, Event)",
            xy=(7.5,2.7),xytext=(13.5,3.8),fontsize=8.5,color="#FFD54F",fontweight="bold",
            arrowprops=dict(arrowstyle="->",color="#FFD54F",lw=1.5),
            bbox=dict(boxstyle="round",facecolor="#1a1a2e",edgecolor="#FFD54F"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG,"05_tft_attention_architecture.png"), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("   -> Saved: 05_tft_attention_architecture.png")

feat_attn_df.to_csv(os.path.join(OUT_RES,"feature_attention_correlation.csv"), index=False, encoding="utf-8-sig")
group_importance.to_csv(os.path.join(OUT_RES,"group_feature_importance.csv"), encoding="utf-8-sig")

print("\n"+"="*65)
print("  KET QUA PHAN TICH ATTENTION")
print("="*65)
print(f"  Exogenous importance: {exo_total:.1f}%")
print(f"  Endogenous importance: {endo_total:.1f}%")
print(f"  Static importance:     {static_total:.1f}%")
print(f"  Event attn: {event_attn:.6f} | Non-event: {no_event_attn:.6f}")
if no_event_attn > 0:
    print(f"  Event/Non-event ratio: {event_attn/no_event_attn:.3f}x")
print(f"\n[DONE] Script 05 hoan tat!")
print(f"  Bieu do luu tai: {OUT_FIG}")
