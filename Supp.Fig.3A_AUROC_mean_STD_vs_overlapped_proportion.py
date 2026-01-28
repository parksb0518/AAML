#%%
# Supp.Fig.3A_AUROC mean vs overlapped score
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

# === Load data ===
file_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/20_79/All_Unique_Window_AUROC_Summary.xlsx"
df = pd.read_excel(file_path)

# === Reference windows ===
ref_windows = [(29, 38), (40, 61), (69, 78)]
ref_dict = {"29-38", "40-61", "69-78"}

ref_colors = {
    "29-38": "#F5798F",
    "40-61": "#EE2D3F",
    "69-78": "#872126"
}

# === Compute calibrated overlap ===
def calc_weighted_overlap(window_str):
    try:
        start, end = map(int, window_str.split('-'))
    except:
        return None
    length_w = end - start + 1
    total_weighted = 0
    for r_start, r_end in ref_windows:
        overlap_start = max(start, r_start)
        overlap_end = min(end, r_end)
        if overlap_start <= overlap_end:
            overlap = overlap_end - overlap_start + 1
            ref_len = r_end - r_start + 1
            weighted = overlap / (length_w + ref_len - overlap)
            total_weighted = max(total_weighted, weighted)
    return total_weighted

df["Calibrated_Overlap"] = df["Window"].apply(calc_weighted_overlap)

# === Valid data ===
valid_df = df.dropna(subset=["Calibrated_Overlap", "AUROC_Mean"])

# === Remove reference windows for regression ===
nonref_df = valid_df[~valid_df["Window"].isin(ref_dict)].copy()

x = nonref_df["Calibrated_Overlap"].values
y = nonref_df["AUROC_Mean"].values

# === Linear regression using NON-REFERENCE windows only ===
slope, intercept = np.polyfit(x, y, 1)
line_x = np.linspace(min(x), max(x), 100)
line_y = slope * line_x + intercept

y_pred = slope * x + intercept
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
r, p = pearsonr(x, y)

# === Compute variance by overlap bins ===
bins = np.linspace(valid_df["Calibrated_Overlap"].min(), valid_df["Calibrated_Overlap"].max(), 7)
valid_df["OverlapBin"] = pd.cut(valid_df["Calibrated_Overlap"], bins=bins, include_lowest=True)
var_df = valid_df.groupby("OverlapBin")["AUROC_Mean"].var().reset_index()
var_df["bin_center"] = var_df["OverlapBin"].apply(lambda b: (b.left + b.right) / 2)

# === Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})

# --- Top: Scatter Plot ---
ax1.scatter(valid_df["Calibrated_Overlap"], valid_df["AUROC_Mean"],
            c="lightgray", edgecolors="black", s=55, alpha=0.7)

# Regression line (non-ref only)
ax1.plot(line_x, line_y, color="black", linestyle="--", linewidth=1.6)

# Highlight reference windows
for ref, color in ref_colors.items():
    row = valid_df[valid_df["Window"] == ref]
    if not row.empty:
        ax1.scatter(row["Calibrated_Overlap"], row["AUROC_Mean"],
                    s=140, c=color, edgecolors="black", label=f"{ref}")
        ax1.text(float(row["Calibrated_Overlap"]) + 0.005,
                 float(row["AUROC_Mean"]),
                 ref, fontsize=10, color=color, fontweight="bold")

ax1.set_ylabel("AUROC Mean", fontsize=12)
ax1.set_title("Calibrated Overlap vs AUROC\n(Regression excludes reference windows)", fontsize=14)

textstr = f"$r$ = {r:.2f}\n$R^2$ = {r2:.2f}\nMAE = {mae:.2f}\n$p$ = {p:.2e}"
ax1.text(0.02, min(valid_df["AUROC_Mean"])+0.006, textstr,
         fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"))

# --- Bottom: Variance Barplot ---
ax2.bar(var_df["bin_center"], var_df["AUROC_Mean"],
        width=(bins[1]-bins[0])*0.8, alpha=0.7, edgecolor="black", color="gray")

ax2.set_ylabel("Variance", fontsize=12)
ax2.set_xlabel("Calibrated Overlap Score", fontsize=12)

# === Save ===
save_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/plots/scatter_overlapped_proportion/scatter_weighted_overlap_auroc_with_variance_subplot_ref_window_eliminate.pdf"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved: {save_path}")