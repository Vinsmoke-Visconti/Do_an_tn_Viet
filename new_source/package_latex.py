import os
import shutil
import zipfile

base_dir = r"d:\NAM_IV (2026-27)\DO_AN_TN\do_an\new_Do_An\new_source\outputs"
pkg_dir = os.path.join(base_dir, "latex_overleaf_package")
zip_path = os.path.join(base_dir, "latex_overleaf_package.zip")

# Re-create package dir
if os.path.exists(pkg_dir):
    shutil.rmtree(pkg_dir)
os.makedirs(os.path.join(pkg_dir, "figures", "seasonality"), exist_ok=True)
os.makedirs(os.path.join(pkg_dir, "figures", "attention"), exist_ok=True)

# Copy .tex and .bib
tex_src = os.path.join(base_dir, "reports", "bao_cao_tong_hop_thuc_nghiem_tft.tex")
bib_src = os.path.join(base_dir, "reports", "references.bib")

shutil.copy(tex_src, os.path.join(pkg_dir, "main.tex"))
shutil.copy(tex_src, os.path.join(pkg_dir, "bao_cao_tong_hop_thuc_nghiem_tft.tex"))
tex_thesis_src = os.path.join(base_dir, "reports", "tong_hop_luan_van_tft_kielhofer.tex")
if os.path.exists(tex_thesis_src):
    shutil.copy(tex_thesis_src, os.path.join(pkg_dir, "tong_hop_luan_van_tft_kielhofer.tex"))
if os.path.exists(bib_src):
    shutil.copy(bib_src, os.path.join(pkg_dir, "references.bib"))

# Copy seasonality figures
fig_season = os.path.join(base_dir, "figures", "seasonality")
dst_season = os.path.join(pkg_dir, "figures", "seasonality")
if os.path.exists(fig_season):
    for f in os.listdir(fig_season):
        shutil.copy(os.path.join(fig_season, f), os.path.join(dst_season, f))

# Copy attention figures
fig_attn = os.path.join(base_dir, "figures", "attention")
dst_attn = os.path.join(pkg_dir, "figures", "attention")
if os.path.exists(fig_attn):
    for f in os.listdir(fig_attn):
        shutil.copy(os.path.join(fig_attn, f), os.path.join(dst_attn, f))

# Zip the package
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(pkg_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, pkg_dir)
            zipf.write(full_path, rel_path)

print(f"SUCCESS: Package folder created at {pkg_dir}")
print(f"SUCCESS: Zip file created at {zip_path}")
print("Contents:")
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        print(" -", os.path.relpath(os.path.join(root, f), pkg_dir))
