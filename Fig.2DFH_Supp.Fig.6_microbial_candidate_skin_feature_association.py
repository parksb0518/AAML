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

# greater or less one sided p-value test based on Odds ratio in age group

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
import os

# === Microbes to test in each age group ===
targets = {
    (29, 38): ["Cutibacterium_acnes", "Cutibacterium_granulosum", "Corynebacterium_propinquum", "Streptococcus_gordonii"],
    (40, 61): ["Cutibacterium_acnes", "Cutibacterium_granulosum", "Corynebacterium_propinquum", "Streptococcus_gordonii"],
    (69, 78): ["Cutibacterium_acnes", "Cutibacterium_granulosum", "Corynebacterium_propinquum", "Streptococcus_gordonii"],
    (20, 79): ["Cutibacterium_acnes", "Cutibacterium_granulosum", "Corynebacterium_propinquum", "Streptococcus_gordonii"]
}

# === Skin features to loop through ===
skin_features = ["OM_TE_score", "Oil.ss", "Moist.ss", "ITA.ss", "R7.ss"]

# === Base save directory ===
base_save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_one_sided_modified"

for feature in skin_features:
    print(f"\n🔍 Odds ratio–based Fisher Test - Feature: {feature}")
    save_dir = os.path.join(base_save_dir, feature)
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for (age_min, age_max), microbe_list in targets.items():
        age_window = f"{age_min}-{age_max}"
        age_group = merged_data[(merged_data["Age"] >= age_min) & (merged_data["Age"] <= age_max)].copy()

        # Remove missing values for current feature
        age_group = age_group.dropna(subset=[feature])

        # Quantile thresholds (top 40% / bottom 40%)
        q_high = age_group[feature].quantile(0.60)
        q_low = age_group[feature].quantile(0.40)

        # Label group
        age_group["SkinGroup"] = age_group[feature].apply(
            lambda x: "Improved" if x >= q_high else ("Deteriorated" if x <= q_low else "Drop")
        )
        age_group = age_group[age_group["SkinGroup"] != "Drop"]

        for microbe in microbe_list:
            if microbe not in age_group.columns:
                print(f"❌ Microbe {microbe} not found in data.")
                continue

            age_group["MicrobePresence"] = (age_group[microbe] > 0).astype(int)

            contingency = pd.crosstab(age_group["SkinGroup"], age_group["MicrobePresence"])
            contingency.columns = ["Absent", "Present"]

            odds, pval, ci_low, ci_upp, alt_used = np.nan, np.nan, np.nan, np.nan, None

            if contingency.shape == (2, 2):
                # Step 1: Get OR from two-sided test
                odds, _ = fisher_exact(contingency, alternative="two-sided")

                # Step 2: Decide alternative for ALL microbes (including Cutibacterium spp.)
                if odds > 1:
                    alt_used = "greater"
                elif odds < 1:
                    alt_used = "less"
                else:
                    alt_used = "two-sided"

                # Step 3: Run directional test
                _, pval = fisher_exact(contingency, alternative=alt_used)

                # Step 4: Confidence interval
                try:
                    table = np.array([
                        [contingency.loc["Improved", "Present"], contingency.loc["Improved", "Absent"]],
                        [contingency.loc["Deteriorated", "Present"], contingency.loc["Deteriorated", "Absent"]]
                    ])
                    ct = Table2x2(table)
                    ci_low, ci_upp = ct.oddsratio_confint()
                except Exception as e:
                    print(f"⚠️ Error computing CI for {microbe} in {age_window} ({feature}): {e}")
            else:
                print(f"⚠️ Skipped {microbe} in {age_window} ({feature}), not a 2x2 table.")

            for group in contingency.index:
                results.append({
                    "Feature": feature,
                    "AgeWindow": age_window,
                    "Microbe": microbe,
                    "Alternative": alt_used,
                    "SkinGroup": group,
                    "Absent": contingency.loc[group, "Absent"],
                    "Present": contingency.loc[group, "Present"],
                    "p-value": pval,
                    "odds_ratio": odds,
                    "CI_lower": ci_low,
                    "CI_upper": ci_upp
                })

    # Save results to Excel
    result_df = pd.DataFrame(results)
    out_path = os.path.join(save_dir, f"fisher_results_one_sided_with_CI_by_age_group_{feature}.xlsx")
    result_df.to_excel(out_path, index=False)
    print(f"✅ OR-based one-sided test saved: {out_path}")


# %%

#Fisher test one-sided based on odds ratio 

import numpy as np
import pandas as pd
import os
from scipy.stats import fisher_exact
from sklearn.linear_model import LinearRegression
from statsmodels.stats.contingency_tables import Table2x2

# === Config ===
target_species = [
    "Cutibacterium_acnes",
    "Corynebacterium_propinquum",
    "Cutibacterium_granulosum"
]

skin_features = ["OM_TE_score", "Oil.ss", "Moist.ss", "ITA.ss", "R7.ss"]
save_root = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_modified_all_age"

# === Loop through each feature ===
for feature in skin_features:
    print(f"\n🔍 Processing: {feature}")

    # Directory
    save_dir = os.path.join(save_root, feature)
    os.makedirs(save_dir, exist_ok=True)

    # Filter data
    df = merged_data[["Age", feature] + target_species].dropna()
    df = df[(df["Age"] >= 20) & (df["Age"] <= 79)]

    # Linear regression: feature ~ Age
    X = df["Age"].values.reshape(-1, 1)
    y = df[feature].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    df["Distance"] = y - y_pred

    # Define Improved / Deteriorated
    above = df[df["Distance"] > 0].copy()
    below = df[df["Distance"] < 0].copy()
    n_above = int(len(above) * 0.4)
    n_below = int(len(below) * 0.4)
    above = above.sort_values("Distance", ascending=False).iloc[:n_above]
    below = below.sort_values("Distance", ascending=True).iloc[:n_below]
    above["SkinGroup"] = "Improved"
    below["SkinGroup"] = "Deteriorated"
    skin_df = pd.concat([above, below], axis=0)

    # Fisher's test per species
    results = []
    for microbe in target_species:
        if microbe not in skin_df.columns:
            print(f"❌ Microbe {microbe} not found.")
            continue

        skin_df["MicrobePresence"] = (skin_df[microbe] > 0).astype(int)
        contingency = pd.crosstab(skin_df["SkinGroup"], skin_df["MicrobePresence"])
        contingency.columns = ["Absent", "Present"]

        odds, pval, ci_low, ci_upp, alt_used = np.nan, np.nan, np.nan, np.nan, None
        if contingency.shape == (2, 2):
            # Compute initial odds ratio from two-sided test
            odds, _ = fisher_exact(contingency, alternative="two-sided")

            # Decide alternative based on odds ratio
            if odds > 1:
                alt_used = "greater"
            elif odds < 1:
                alt_used = "less"
            else:
                alt_used = "two-sided"  # rare case: odds ratio exactly 1

            # Run directional test
            _, pval = fisher_exact(contingency, alternative=alt_used)

            # Confidence interval
            try:
                table = np.array([
                    [contingency.loc["Improved", "Present"], contingency.loc["Improved", "Absent"]],
                    [contingency.loc["Deteriorated", "Present"], contingency.loc["Deteriorated", "Absent"]]
                ])
                ct = Table2x2(table)
                ci_low, ci_upp = ct.oddsratio_confint()
            except Exception as e:
                print(f"⚠️ Error for {microbe}: {e}")
        else:
            print(f"⚠️ Skipped {microbe}, not a 2x2 table.")

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

    # Save results
    results_df = pd.DataFrame(results)
    out_path = os.path.join(save_dir, f"fisher_results_all_age_{feature}.xlsx")
    results_df.to_excel(out_path, index=False)
    print(f"✅ Saved: {out_path}")

# %%

#forest plot of each microbe (Fig.2DFH)

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config ===
base_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_one_sided"
modified_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/Fisher_exact_test_modified_all_age"
output_dir = os.path.join(modified_dir, "forest_plots_by_microbe_feature")
os.makedirs(output_dir, exist_ok=True)

features = ["OM_TE_score", "Oil.ss", "Moist.ss", "ITA.ss", "R7.ss"]
age_order = ["20-79", "69-78", "40-61", "29-38"]
color_map = {
    "20-79": "gray",
    "29-38": "#D5221E",
    "40-61": "#4EA74A",
    "69-78": "#EF7B19"
}
feature_name_map = {
    "OM_TE_score": "OM_TE Score",
    "Oil.ss": "Oil",
    "Moist.ss": "Moist",
    "ITA.ss": "Skin Tone",
    "R7.ss": "Elasticity"
}

def get_asterisks(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

for feature in features:
    feature_dir_orig = os.path.join(base_dir, feature)
    feature_dir_mod = os.path.join(modified_dir, feature)

    age_file = os.path.join(feature_dir_orig, f"fisher_results_one_sided_with_CI_by_age_group_{feature}.xlsx")
    all_file = os.path.join(feature_dir_mod, f"fisher_results_all_age_{feature}.xlsx")

    # Load data
    df_age = pd.read_excel(age_file)
    df_all = pd.read_excel(all_file)

    df_age = df_age[df_age["SkinGroup"] == "Improved"].copy()
    df_all = df_all[df_all["SkinGroup"] == "Improved"].copy()

    # Fix odds_ratio == 0
    for df in [df_age, df_all]:
        zero_mask = df["odds_ratio"] == 0
        df.loc[zero_mask, "odds_ratio"] = df.loc[zero_mask, "CI_lower"]

    df_combined = pd.concat([df_age, df_all], ignore_index=True)
    microbes = df_combined["Microbe"].unique()

    for microbe in microbes:
        sub_df = df_combined[df_combined["Microbe"] == microbe].copy()
        sub_df["AgeWindow"] = pd.Categorical(sub_df["AgeWindow"], categories=age_order, ordered=True)
        sub_df = sub_df.sort_values("AgeWindow")

        fig, ax = plt.subplots(figsize=(6, 3))
        for _, row in sub_df.iterrows():
            odds = row["odds_ratio"]
            ci_low = row["CI_lower"]
            ci_upp = row["CI_upper"]
            age = row["AgeWindow"]
            pval = row["p-value"]

            if pd.notna(odds) and pd.notna(ci_low) and pd.notna(ci_upp):
                color = color_map.get(age, "black")
                ypos = age_order.index(age)
                ax.errorbar(odds, ypos,
                            xerr=[[odds - ci_low], [ci_upp - odds]],
                            fmt='o', color=color, capsize=4)
                stars = get_asterisks(pval)
                if stars:
                    ax.text(odds, ypos + 0.2, stars, ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.axvline(1, linestyle="--", color="black", linewidth=1)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_yticks(range(len(age_order)))
        ax.set_yticklabels(age_order)
        ax.set_xlabel("Odds Ratio")

        display_feature = feature_name_map.get(feature, feature)
        ax.set_title(f"{microbe} - {display_feature}")

        plt.tight_layout()

        safe_microbe = microbe.replace(" ", "_").replace("/", "_")
        pdf_path = os.path.join(output_dir, f"forestplot_{safe_microbe}_{feature}.pdf")
        png_path = os.path.join(output_dir, f"forestplot_{safe_microbe}_{feature}.png")

        plt.savefig(pdf_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {pdf_path} and {png_path}")