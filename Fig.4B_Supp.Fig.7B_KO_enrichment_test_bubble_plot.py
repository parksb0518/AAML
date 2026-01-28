# %%

# Fig4B: Bubble plot of C.propinquum

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
file_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/PICRUSt2/ko_annotation_coryne_40to61_skin_condition.tsv"
output_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/PICRUSt2"
output_base = os.path.join(output_dir, "Top4_pathways_bubble_blue_yellow")

# Load data
df = pd.read_csv(file_path, sep='\t')
df = df.dropna(subset=['pathway_name', 'log2_fold_change', 'p_adjust'])
df['pathway_name'] = df['pathway_name'].astype(str)

# Get top 4 positive and top 4 negative by p_adjust (tie-breaker: log2_fold_change)
top_pos = df[df['log2_fold_change'] > 0].sort_values(['p_adjust', 'log2_fold_change'], ascending=[True, False]).head(4)
top_neg = df[df['log2_fold_change'] < 0].sort_values(['p_adjust', 'log2_fold_change'], ascending=[True, True]).head(4)
plot_df = pd.concat([top_pos, top_neg])

# Calculate bubble size from significance
plot_df['neg_log10_p'] = -np.log10(plot_df['p_adjust'])
sizes = plot_df['neg_log10_p'] * 100  # scale factor for visibility

# Custom colormap from yellow to white to blue
custom_cmap = LinearSegmentedColormap.from_list(
    "yellow_white_blue",
    ["#EAA72F", "white", "#0000E6"],
    N=256
)

# Sort pathways so negatives are on top for better visual separation
plot_df = plot_df.sort_values('log2_fold_change')

# Plot
plt.figure(figsize=(7.5, 3.5))
scatter = plt.scatter(
    plot_df['log2_fold_change'],
    plot_df['pathway_name'],
    s=sizes,
    c=plot_df['log2_fold_change'],
    cmap=custom_cmap,
    alpha=0.8,
    edgecolor='black'
)

# Colorbar for log2FC
cbar = plt.colorbar(scatter)
cbar.set_label('log2(Fold Change)')

# === Add bubble size legend for p-values ===
legend_pvals = [0.05, 0.01, 0.001]  # example cutoffs
legend_sizes = [-np.log10(legend_pvals) * 100]  # match scale factor
for p in legend_pvals:
    plt.scatter([], [], s=-np.log10(p) * 100, color='gray', alpha=0.6, edgecolor='black',
                label=f"p={p}")

plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='p-value', loc='lower right')

# Labels & formatting
plt.xlabel('log2(Fold Change)')
plt.ylabel('Pathway')
plt.tight_layout()

# Save
plt.savefig(f"{output_base}.pdf", format='pdf')
plt.savefig(f"{output_base}.png", format='png', dpi=300)
plt.close()

# %%

# Fig4B: Bubble plot of S.gordonii

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# === Config ===
file_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/PICRUSt2/Streptococcus_gordonii_SkinGroup_ko_annotation.tsv"
output_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/PICRUSt2"
output_base = os.path.join(
    output_dir,
    "age_skingroup_Top4_pathways_bubble_blue_yellow"
)

# === Load data ===
df = pd.read_csv(file_path, sep="\t")
df = df.dropna(subset=["pathway_name", "log2_fold_change", "p_adjust"])
df["pathway_name"] = df["pathway_name"].astype(str)

# === Handle p_adjust == 0 ===
min_nonzero_p = df.loc[df["p_adjust"] > 0, "p_adjust"].min()
df.loc[df["p_adjust"] == 0, "p_adjust"] = min_nonzero_p

print(f"✔ Replaced p_adjust=0 with min non-zero value: {min_nonzero_p:.2e}")

# === Select top pathways ===
top_pos = (
    df[df["log2_fold_change"] > 0]
    .sort_values(["p_adjust", "log2_fold_change"], ascending=[True, False])
    .head(4)
)

top_neg = (
    df[df["log2_fold_change"] < 0]
    .sort_values(["p_adjust", "log2_fold_change"], ascending=[True, True])
    .head(4)
)

plot_df = pd.concat([top_pos, top_neg])

# === Bubble size from significance ===
plot_df["neg_log10_p"] = -np.log10(plot_df["p_adjust"])
sizes = plot_df["neg_log10_p"] * 20  # scale factor

# === Custom colormap ===
custom_cmap = LinearSegmentedColormap.from_list(
    "yellow_white_blue",
    ["#EAA72F", "white", "#0000E6"], # change order#0000E6
    N=256
)

# Sort so negative log2FC on top
plot_df = plot_df.sort_values("log2_fold_change")

# === Plot ===
plt.figure(figsize=(7.5, 3.5))
scatter = plt.scatter(
    plot_df["log2_fold_change"],
    plot_df["pathway_name"],
    s=sizes,
    c=plot_df["log2_fold_change"],
    cmap=custom_cmap,
    alpha=0.8,
    edgecolor="black"
)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("log2(Fold Change)")

# === Bubble size legend (p-values) ===
legend_pvals = [0.05, 0.01, 0.001]
for p in legend_pvals:
    plt.scatter(
        [],
        [],
        s=-np.log10(p) * 20,
        color="gray",
        alpha=0.6,
        edgecolor="black",
        label=f"p={p}"
    )

plt.legend(
    scatterpoints=1,
    frameon=True,
    labelspacing=1,
    title="Adjusted p-value",
    loc="lower right"
)

# Labels
plt.xlabel("log2(Fold Change)")
plt.ylabel("Pathway")
plt.tight_layout()
plt.show()

# Save
plt.savefig(f"{output_base}.pdf", format="pdf")
plt.savefig(f"{output_base}.png", format="png", dpi=300)

plt.close()
# %%
