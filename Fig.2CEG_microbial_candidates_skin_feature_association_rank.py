

# %%
import os
import pandas as pd
import numpy as np
# === Paths ===
data_dir = "/home/parksb0518/SBI_Lab/COSMAX/Data"
merged_data_file = os.path.join(data_dir, "merged_data.csv")
species_data_file = os.path.join(data_dir, "species_data.csv")

# === Read merged_data and species_data ===
merged_data = pd.read_csv(merged_data_file)
species_data = pd.read_csv(species_data_file)
# %%
from sklearn.preprocessing import MinMaxScaler

# Step 1: Identify where the species data starts
start_col = "Edaphobacter_sp."
start_idx = merged_data.columns.get_loc(start_col)

# Step 2: Slice species abundance data
species_cols = merged_data.columns[start_idx:]
species_abund = merged_data[species_cols]

# Step 3: Apply MinMaxScaler column-wise
scaler = MinMaxScaler()
scaled_species = pd.DataFrame(
    scaler.fit_transform(species_abund),
    columns=species_cols,
    index=merged_data.index  # maintain original row index
)

# Step 4: Replace original values with scaled ones
merged_data.loc[:, species_cols] = scaled_species

# ✅ Confirm
print("✅ Scaled species range:")
print(scaled_species.describe().T[["min", "max"]].head())

# %%
import pandas as pd

# Load clinical data from Excel
om_te_data = pd.read_excel("/home/parksb0518/SBI_Lab/COSMAX/Data/clinical_data_OM_TE.xlsx")

# Drop rows with missing OM_TE_score
om_te_data = om_te_data.dropna(subset=["OM_TE_score"])

# Ensure mb.selected_sample.id is string type
om_te_data["mb.selected_sample-id"] = om_te_data["mb.selected_sample-id"].astype(str)
merged_data["mb.selected_sample-id"] = merged_data["mb.selected_sample-id"].astype(str)

# Merge OM_TE_score into merged_data
merged_data = merged_data.merge(
    om_te_data[["mb.selected_sample-id", "OM_TE_score"]],
    on="mb.selected_sample-id",
    how="left"
)

# Check if the merge worked
print(merged_data["OM_TE_score"].describe())


# %%
##AAML_rank_generate

# === Add Rank, Total_number, and Direction for ML Z-score files ===

import pandas as pd
import os

# === Input file paths ===
input_files = [
    "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/OM_TE_score/all_skin_resident_flora_zscore_5CV_29-38.xlsx",
    "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/OM_TE_score/all_skin_resident_flora_zscore_5CV_40-61.xlsx",
    "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/OM_TE_score/all_skin_resident_flora_zscore_5CV_69-78.xlsx",
]

# === Output directory ===
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/rank_generation"
os.makedirs(save_dir, exist_ok=True)

for file_path in input_files:
    print(f"\n📂 Processing: {file_path}")
    df = pd.read_excel(file_path)

    # === Sanity check ===
    if "Z-score Mean" not in df.columns or "Abs Z-score Mean" not in df.columns:
        raise ValueError(f"⚠️ Missing required columns in {file_path}")

    # === Add Direction ===
    df["Direction"] = df["Z-score Mean"].apply(
        lambda x: "Improved" if x > 0 else ("Deteriorated" if x < 0 else "Neutral")
    )

    # === Add Rank and Total_number ===
    df["Rank"] = range(1, len(df) + 1)
    df["Total_number"] = len(df)

    # === Define save path ===
    filename = os.path.basename(file_path).replace(".xlsx", "_ranked.xlsx")
    output_path = os.path.join(save_dir, filename)

    # === Save result ===
    df.to_excel(output_path, index=False)
    print(f"✅ Saved ranked file to:\n{output_path}")

    # === Preview ===
    print(df[["Species", "Z-score Mean", "Abs Z-score Mean", "Direction", "Rank", "Total_number"]].head(5))

# %%
#Pearson_rank_generate

# === Pearson_R2_rank_add_with_direction ===

import pandas as pd
import os

# === Input & Output paths ===
input_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/Pearson_correlation_all_age_R2/pearson_R2_all_age_presence_only_sorted.xlsx"
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/rank_generation"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "pearson_R2_all_age_presence_only_ranked.xlsx")

# === Load data ===
df = pd.read_excel(input_path)
print(f"✅ Loaded: {df.shape[0]} rows")

# === Add Rank and Total_number ===
df["Rank"] = range(1, len(df) + 1)
df["Total_number"] = len(df)

# === Add Direction based on Pearson_r sign ===
if "Pearson_r" not in df.columns:
    raise ValueError("⚠️ Column 'Pearson_r' not found in the Excel file.")
df["Direction"] = df["Pearson_r"].apply(lambda x: "Improved" if x > 0 else ("Deteriorated" if x < 0 else "Neutral"))

# === Save result ===
df.to_excel(output_path, index=False)
print(f"✅ Ranked Pearson_r file saved to:\n{output_path}")

# === Preview top results ===
print(df[["Microbe", "Pearson_r", "Direction", "Rank", "Total_number"]].head(10))


# %%

# all resident flora fisher exact test in all age (not distance-based)

import numpy as np
import pandas as pd
import os
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

# === Config ===
target_genera = ["Cutibacterium", "Streptococcus", "Staphylococcus", "Corynebacterium", "Neisseria"]
skin_features = ["OM_TE_score", "Oil.ss", "Moist.ss", "ITA.ss", "R7.ss"]
save_root = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_species_all_resident_flora_all_age_Genus"

# === Find all microbial species columns in selected genera ===
species_columns = [
    col for col in merged_data.columns
    if any(col.startswith(genus + "_") for genus in target_genera)
]
print(f"✅ Found {len(species_columns)} target species under selected genera.")

# === Loop through each skin feature ===
for feature in skin_features:
    print(f"\n🔍 Processing all-age Fisher test (non-distance): {feature}")

    # save folder for this feature
    save_dir = os.path.join(save_root, feature)
    os.makedirs(save_dir, exist_ok=True)

    # === Filter for age 20-79 and keep necessary columns ===
    df = merged_data[["Age", feature] + species_columns].dropna(subset=[feature])
    df = df[(df["Age"] >= 20) & (df["Age"] <= 79)]

    # === 1️⃣ Divide into Top 40% (Improved) and Bottom 40% (Deteriorated) ===
    df["Rank"] = df[feature].rank(ascending=False, method="average")
    q_top = df["Rank"].quantile(0.40)
    q_bottom = df["Rank"].quantile(0.60)

    improved = df[df["Rank"] <= q_top].copy()
    deteriorated = df[df["Rank"] >= q_bottom].copy()

    improved["SkinGroup"] = "Improved"
    deteriorated["SkinGroup"] = "Deteriorated"

    skin_df = pd.concat([improved, deteriorated], axis=0)

    # === 2️⃣ Run Fisher exact test per microbe ===
    results = []

    for microbe in species_columns:

        if microbe not in skin_df.columns:
            continue

        # presence = abundance > 0
        skin_df["MicrobePresence"] = (skin_df[microbe] > 0).astype(int)

        contingency = pd.crosstab(skin_df["SkinGroup"], skin_df["MicrobePresence"])

        # ensure 2x2 shape
        if 0 not in contingency.columns:
            contingency[0] = 0
        if 1 not in contingency.columns:
            contingency[1] = 0

        contingency = contingency[[0, 1]]
        contingency.columns = ["Absent", "Present"]

        if contingency.shape != (2, 2):
            print(f"⚠️ Skipped (not 2x2): {microbe}")
            continue

        # Step 1: two-sided OR
        odds, _ = fisher_exact(contingency, alternative="two-sided")

        # Step 2: choose one-sided direction
        if odds > 1:
            alt_used = "greater"
        elif odds < 1:
            alt_used = "less"
        else:
            alt_used = "two-sided"

        # Step 3: compute p-value using chosen alternative
        _, pval = fisher_exact(contingency, alternative=alt_used)

        # Step 4: confidence interval
        try:
            table = np.array([
                [contingency.loc["Improved", "Present"], contingency.loc["Improved", "Absent"]],
                [contingency.loc["Deteriorated", "Present"], contingency.loc["Deteriorated", "Absent"]]
            ])
            ct = Table2x2(table)
            ci_low, ci_upp = ct.oddsratio_confint()
        except Exception as e:
            print(f"⚠️ CI failed for {microbe}: {e}")
            ci_low, ci_upp = np.nan, np.nan

        # record both rows (Improved / Deteriorated)
        for group in contingency.index:
            results.append({
                "Feature": feature,
                "AgeWindow": "20-79",
                "Microbe": microbe,
                "SkinGroup": group,
                "Absent": contingency.loc[group, "Absent"],
                "Present": contingency.loc[group, "Present"],
                "p-value": pval,
                "odds_ratio": odds,
                "CI_lower": ci_low,
                "CI_upper": ci_upp,
                "Fisher_alternative": alt_used
            })

    # === 3️⃣ Save results ===
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="odds_ratio", ascending=False)

    out_path = os.path.join(save_dir, f"fisher_results_all_age_Top40_Bottom40_{feature}.xlsx")
    results_df.to_excel(out_path, index=False)

    print(f"✅ Saved: {out_path}")
# %%
## Pearson correlation (R²-based) across all ages (20-79) for all resident flora species (presence-only)

import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import os

# === Config ===
skin_features = ["OM_TE_score"]
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/Pearson_correlation_all_age_R2"
os.makedirs(save_dir, exist_ok=True)

# === Define resident flora genera ===
resident_genera = ["Cutibacterium", "Streptococcus", "Staphylococcus", "Rothia", "Corynebacterium", "Neisseria"]

# === Identify resident flora columns ===
species_cols = [
    c for c in merged_data.columns
    if any(c.startswith(genus + "_") for genus in resident_genera)
]
print(f"✅ Found {len(species_cols)} resident flora species columns")

# === Filter age range ===
df = merged_data[(merged_data["Age"] >= 20) & (merged_data["Age"] <= 79)].copy()

# === Run correlation analysis ===
results = []

for microbe in species_cols:
    for feature in skin_features:
        if feature not in df.columns:
            continue

        # Presence-only samples
        sub = df[df[microbe] > 0][[microbe, feature]].dropna()
        if len(sub) < 3:
            continue

        try:
            r, p = pearsonr(sub[microbe], sub[feature])
            r2 = r ** 2  # compute R-squared
        except Exception as e:
            print(f"⚠️ Error for {microbe} × {feature}: {e}")
            continue

        results.append({
            "Microbe": microbe,
            "SkinFeature": feature,
            "N_samples": len(sub),
            "Pearson_r": r,
            "R_squared": r2,
            "p-value": p
        })

# === Sort by feature then descending R² ===
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by=["SkinFeature", "R_squared"], ascending=[True, False])

# === Save to Excel ===
out_path = os.path.join(save_dir, "pearson_R2_all_age_presence_only_sorted.xlsx")
res_df.to_excel(out_path, index=False)

print(f"✅ R² correlation analysis done!")
print(f"📁 Saved at: {out_path}")
print(f"📊 Total tested pairs: {len(res_df)}")

# %%
# Fisher_test_rank_generate (top40 bottom 40)

import pandas as pd
import numpy as np
import os

# === Input & Output paths ===
input_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_species_all_resident_flora_all_age_Genus/OM_TE_score/fisher_results_all_age_Top40_Bottom40_OM_TE_score.xlsx"
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/rank_generation"
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, "fisher_rank_top40_bottom40_OM_TE_score_ln.xlsx")

# === Load data ===
df = pd.read_excel(input_path)
print(f"✅ Loaded: {df.shape[0]} rows")

# === Clean up ===
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["odds_ratio"])
df = df[df["odds_ratio"] != 0]

print(f"🧹 After cleaning: {df.shape[0]} rows remain")

# === Remove redundant rows (keep one per microbe) ===
df = df.sort_values(by="odds_ratio", ascending=False)
df = df.drop_duplicates(subset=["Microbe"], keep="first")

# === Add natural log (ln) of odds ratio ===
df["ln_OR"] = np.log(df["odds_ratio"])

# === Direction classification ===
df["Direction"] = np.where(df["ln_OR"] > 0, "Improved", "Deteriorated")

# === Sort by absolute ln(OR) descending ===
df["abs_ln_OR"] = df["ln_OR"].abs()
df = df.sort_values(by="abs_ln_OR", ascending=False).reset_index(drop=True)

# === Add Rank and Total_number ===
df["Rank"] = df.index + 1
df["Total_number"] = len(df)

# === Save result ===
df.to_excel(output_path, index=False)
print(f"✅ Natural log rank file saved to:\n{output_path}")

# === Preview top results ===
print(df[["Rank", "Microbe", "odds_ratio", "ln_OR", "Direction", "Total_number"]].head(10))

# %%
# Fisher_test_rank_generate_by_agewindow

import pandas as pd
import numpy as np
import os

# === Input & Output paths ===
input_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_one_sided_species_all_resident_flora/OM_TE_score/fisher_results_one_sided_species_under_Genus_sorted_OM_TE_score.xlsx"
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_one_sided_species_all_resident_flora/OM_TE_score"
os.makedirs(save_dir, exist_ok=True)

# === Load data ===
df = pd.read_excel(input_path)
print(f"✅ Loaded: {df.shape[0]} rows")

# === Clean up ===
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["odds_ratio"])
df = df[df["odds_ratio"] != 0]
print(f"🧹 After cleaning: {df.shape[0]} rows remain")

# === Check unique AgeWindow values ===
print("📊 AgeWindow values:", df["AgeWindow"].unique())

# === Define target AgeWindows ===
target_windows = ["29-38", "40-61", "69-78"]

# === Loop over each AgeWindow ===
for window in target_windows:
    sub_df = df[df["AgeWindow"] == window].copy()
    if sub_df.empty:
        print(f"⚠️ No data for AgeWindow {window}, skipping.")
        continue

    print(f"\nProcessing AgeWindow: {window} ({sub_df.shape[0]} rows)")

    # --- Remove redundant microbes ---
    sub_df = sub_df.sort_values(by="odds_ratio", ascending=False)
    sub_df = sub_df.drop_duplicates(subset=["Species"], keep="first")

    # --- Add ln(OR) ---
    sub_df["ln_OR"] = np.log(sub_df["odds_ratio"])

    # --- Add Direction ---
    sub_df["Direction"] = np.where(sub_df["ln_OR"] > 0, "Improved", "Deteriorated")

    # --- Sort by absolute ln(OR) ---
    sub_df["abs_ln_OR"] = sub_df["ln_OR"].abs()
    sub_df = sub_df.sort_values(by="abs_ln_OR", ascending=False).reset_index(drop=True)

    # --- Add Rank and Total_number ---
    sub_df["Rank"] = sub_df.index + 1
    sub_df["Total_number"] = len(sub_df)

    # --- Reorder columns for clarity ---
    columns_order = (
        ["Rank", "Species", "odds_ratio", "ln_OR", "abs_ln_OR", "Direction", "p-value", "AgeWindow", "Total_number"]
        + [c for c in sub_df.columns if c not in ["Rank", "Species", "odds_ratio", "ln_OR", "abs_ln_OR", "Direction", "p-value", "AgeWindow", "Total_number"]]
    )
    sub_df = sub_df[columns_order]

    # --- Save each AgeWindow separately ---
    output_path = os.path.join(save_dir, f"fisher_rank_OM_TE_score_ln_{window}.xlsx")
    sub_df.to_excel(output_path, index=False)
    print(f"✅ Saved ranked file for AgeWindow {window} to:\n{output_path}")

# === Done ===
print("\n🎯 All AgeWindows processed successfully!")

# %%
#rank barplot generator (Fig.2CEG)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Paths ===
rank_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/rank_generation"

# === Input files ===
fisher_file = f"{rank_dir}/fisher_rank_OM_TE_score_ln.xlsx"
fisher_file_top40_bottom40 = f"{rank_dir}/fisher_rank_top40_bottom40_OM_TE_score_ln.xlsx"
pearson_file = f"{rank_dir}/pearson_R2_all_age_presence_only_ranked.xlsx"
aaml_young = f"{rank_dir}/fisher_rank_OM_TE_score_ln_29-38.xlsx"
aaml_middle = f"{rank_dir}/fisher_rank_OM_TE_score_ln_40-61.xlsx"
aaml_old = f"{rank_dir}/fisher_rank_OM_TE_score_ln_69-78.xlsx"

# === Load data ===
fisher = pd.read_excel(fisher_file)
fisher_top40_bottom40 = pd.read_excel(fisher_file_top40_bottom40)
pearson = pd.read_excel(pearson_file)
young = pd.read_excel(aaml_young)
middle = pd.read_excel(aaml_middle)
old = pd.read_excel(aaml_old)

# === Target species ===
target_species = [
    "Corynebacterium_propinquum",
    "Cutibacterium_acnes",
    "Cutibacterium_granulosum"
]

# === Helper: extract rank info ===
def get_rank_info(df, species_name):
    if "Species" in df.columns:
        name_col = "Species"
    elif "Microbe" in df.columns:
        name_col = "Microbe"
    else:
        raise ValueError("No Species/Microbe column found")
    row = df[df[name_col] == species_name]
    if row.empty:
        return None, None
    return row["Rank"].values[0], row["Total_number"].values[0]

# === Plotting setup ===
color_map = {
    "AAML Young (29-38)": "#F5798F",
    "AAML Middle (40-61)": "#EE2D3F",
    "AAML Old (69-78)": "#872126",
    "Pearson": "#A7A9AC",
    "Fisher": "#58585A",
    "Fisher_top40_bottom40": "#58585A"
}

plot_order = [
    "AAML Young (29-38)",
    "AAML Middle (40-61)",
    "AAML Old (69-78)",
    "Pearson",
    "Fisher",
    "Fisher_top40_bottom40"
]

max_rank = 58  # top rank range

# === Plot per species ===
for sp in target_species:
    rank_info = {
        "AAML Young (29-38)": get_rank_info(young, sp),
        "AAML Middle (40-61)": get_rank_info(middle, sp),
        "AAML Old (69-78)": get_rank_info(old, sp),
        "Pearson": get_rank_info(pearson, sp),
        "Fisher": get_rank_info(fisher, sp),
        "Fisher_top40_bottom40": get_rank_info(fisher_top40_bottom40, sp)
    }

    # Keep only valid ones
    rank_info = {k: v for k, v in rank_info.items() if v[0] is not None}
    if not rank_info:
        print(f"⚠️ No data for {sp}")
        continue

    # Ensure consistent order
    labels = [lbl for lbl in plot_order if lbl in rank_info.keys()]
    ranks = [rank_info[lbl][0] for lbl in labels]
    totals = [rank_info[lbl][1] for lbl in labels]
    colors = [color_map[lbl] for lbl in labels]

    # Inverted height (Rank 1 top → Rank 58 bottom)
    bar_heights = [max_rank + 1 - r for r in ranks]

    # === Plot ===
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    bars = ax.bar(labels, bar_heights, color=colors, width=0.6, edgecolor='k', linewidth=0.5)

    # Add rank text
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{ranks[i]}/{totals[i]}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # === Y-axis: Rank scale (1 top → 58 bottom) ===
    yticks = np.arange(1, max_rank + 1, 7)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks[::-1]])
    ax.set_ylabel("Rank (1 = Top, 58 = Bottom)", fontsize=11)
    ax.set_ylim(0, max_rank + 5)

    # === Title & Style ===
    ax.set_title(sp.replace("_", " "), fontsize=14, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=10)

    # Vertical lines between groups
    ax.vlines(2.5, 0, max_rank, colors="black", linestyles="--", lw=0.8, alpha=0.6)
    ax.vlines(3.5, 0, max_rank, colors="black", linestyles="--", lw=0.8, alpha=0.6)

    # Legend
    handles = [plt.Line2D([0], [0], color=color_map[k], lw=6, label=k) for k in plot_order if k in rank_info.keys()]
    #ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()

    # Save output
    save_base = os.path.join(rank_dir, f"top40_bottom40_Addition_Rank_barplot_{sp}_final_fisher_age58")
    plt.savefig(f"{save_base}.png", dpi=300)
    plt.savefig(f"{save_base}.pdf")
    plt.close()

    print(f"✅ Saved {sp}: {save_base}")

print("🎯 All species plots completed successfully.")



# %%
