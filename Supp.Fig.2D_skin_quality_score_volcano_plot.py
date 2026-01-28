# %%
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# === Load data ===
file_path = "/home/parksb0518/SBI_Lab/COSMAX/Data/clinical_data_OM_TE.xlsx"
df = pd.read_excel(file_path)

# === Drop missing OM_TE_score ===
df = df.dropna(subset=["OM_TE_score"])

# === Percentile-based group using rank ===
df["OM_TE_rank"] = df["OM_TE_score"].rank(pct=True)
top_40 = df[df["OM_TE_rank"] >= 0.6]
bottom_40 = df[df["OM_TE_rank"] <= 0.4]

# === Select numeric features ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ["OM_TE_score", "OM_TE_rank"]]

# === Mann–Whitney test between Top and Bottom 40% ===
results = []
for feat in numeric_cols:
    group1 = top_40[feat].dropna()
    group2 = bottom_40[feat].dropna()
    if len(group1) > 2 and len(group2) > 2:
        stat, pval = mannwhitneyu(group1, group2, alternative="two-sided")
        mean_diff = group1.mean() - group2.mean()
        results.append({'feature': feat, 'mean_diff': mean_diff, 'pval': pval})

# === Convert results to DataFrame ===
result_df = pd.DataFrame(results)

if result_df.empty:
    print("⚠️ No features passed the test.")
else:
    result_df["log_pval"] = -np.log10(result_df["pval"])
    result_df["significant"] = result_df["pval"] < 0.05
    result_df["direction"] = result_df["mean_diff"].apply(lambda x: "Up" if x > 0 else "Down")

    # Color based on significance + direction
    def get_color(row):
        if not row["significant"]:
            return "gray"
        return "#D5221E" if row["mean_diff"] > 0 else "#3677AD"

    result_df["color"] = result_df.apply(get_color, axis=1)
    
    result_df.to_csv("/home/parksb0518/SBI_Lab/COSMAX/plots/OM_TE_volcano_result_df.csv", index=False)

    # === Volcano Plot ===
    plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(
        data=result_df,
        x="mean_diff",
        y="log_pval",
        hue="color",
        palette={"#D5221E": "#D5221E", "#3677AD": "#3677AD", "gray": "gray"},
        style="significant",
        markers={True: "o", False: "o"},
        edgecolor="black",
        s=80,
        legend=False
    )

    # Threshold lines
    plt.axhline(-np.log10(0.05), linestyle='--', color='black', linewidth=1)
    plt.axvline(0, linestyle='--', color='black', linewidth=1)
    plt.xlim(-100, 50)

    # Annotate top 10 most significant features
    top_features = result_df.sort_values("pval").head(10)
    for i, (_, row) in enumerate(top_features.iterrows()):
        # Alternate left and right positions
        x_offset = 15 if i % 2 == 0 else -15
        y_offset = 0.5 if i % 2 == 0 else 0.8

        ax.annotate(
            row["feature"],
            xy=(row["mean_diff"], row["log_pval"]),
            xytext=(row["mean_diff"] + x_offset, row["log_pval"] + y_offset),
            textcoords='data',
            fontsize=9,
            arrowprops=dict(
                arrowstyle="-",
                color='black',
                lw=0.7,
                shrinkA=0,
                shrinkB=3
                #connectionstyle="arc3,rad=0.2"
            ),
            horizontalalignment='right' if x_offset < 0 else 'left',
            verticalalignment='bottom'
    )

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Significant Up',
               markerfacecolor='#D5221E', markersize=6,),
        Line2D([0], [0], marker='o', color='w', label='Significant Down',
               markerfacecolor='#3677AD', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Not Significant',
               markerfacecolor='gray', markersize=6)
    ]
    ax.legend(handles=legend_elements, title="Feature Change", loc='upper left', fontsize=6, title_fontsize = 6)

    # Labels and layout
    plt.xlabel("Mean Difference (OM_TE_Top40% - OM_TE_Bottom40%)")
    plt.ylabel("-log10(p-value)")
    #plt.title("Volcano Plot (Mann–Whitney U): OM_TE_score-Based Feature Differences")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    save_path = "/home/parksb0518/SBI_Lab/COSMAX/plots/volcano_plot_OM_TE.png"
    save_path_pdf = "/home/parksb0518/SBI_Lab/COSMAX/plots/volcano_plot_OM_TE.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    plt.show()
    
    