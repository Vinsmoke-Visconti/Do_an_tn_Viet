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

# Recursive copy for seasonality figures (supports subdirectories L1, L2, etc.)
fig_season = os.path.join(base_dir, "figures", "seasonality")
dst_season = os.path.join(pkg_dir, "figures", "seasonality")
if os.path.exists(fig_season):
    for root, dirs, files in os.walk(fig_season):
        rel = os.path.relpath(root, fig_season)
        target_sub = os.path.join(dst_season, rel) if rel != "." else dst_season
        os.makedirs(target_sub, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(root, f), os.path.join(target_sub, f))

# Recursive copy for attention figures
fig_attn = os.path.join(base_dir, "figures", "attention")
dst_attn = os.path.join(pkg_dir, "figures", "attention")
if os.path.exists(fig_attn):
    for root, dirs, files in os.walk(fig_attn):
        rel = os.path.relpath(root, fig_attn)
        target_sub = os.path.join(dst_attn, rel) if rel != "." else dst_attn
        os.makedirs(target_sub, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(root, f), os.path.join(target_sub, f))

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
print("Contents summary:")
file_count = 0
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        file_count += 1
print(f"Total packaged files: {file_count}")
