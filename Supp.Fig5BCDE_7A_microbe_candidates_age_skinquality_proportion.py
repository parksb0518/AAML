# %%
import pandas as pd
import os

# === Paths ===
data_dir = "/home/parksb0518/SBI_Lab/COSMAX/Data"
merged_data_file = os.path.join(data_dir, "merged_data.csv")
species_data_file = os.path.join(data_dir, "species_data.csv")

# === Save species_data ===
#species_data.to_csv(species_data_file, index=False)
#print(f"✅ species_data saved to {species_data_file}")
# %%
# === Read merged_data and species_data ===
merged_data = pd.read_csv(merged_data_file)
species_data = pd.read_csv(species_data_file)

print("✅ merged_data and species_data loaded")
print("merged_data shape:", merged_data.shape)
print("species_data shape:", species_data.shape)
# Check if the merge worked
print(merged_data["OM_TE_score"].describe())

# %%
# greater or less one sided p-value test based on Odds ratio in age group
#skin quality based

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
skin_features = ["OM_TE_score"]

# === Base save directory ===
base_save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/skinquality_Fisher_exact_test_one_sided"

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
    out_path = os.path.join(save_dir, f"streptococcus_added_skinquality_fisher_results_one_sided_with_CI_by_age_group_{feature}.xlsx")
    result_df.to_excel(out_path, index=False)
    print(f"✅ OR-based one-sided test saved: {out_path}")
#%%
#age_based_fisher_Test

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
import os

# ===============================
# CONFIG
# ===============================
AGE_COL = "Age"

MICROBES = [
    "Cutibacterium_acnes",
    "Cutibacterium_granulosum",
    "Corynebacterium_propinquum",
    "Streptococcus_gordonii"
]

YOUNG_RANGE = (20, 49)
OLD_RANGE = (50, 79)

SAVE_DIR = "/home/parksb0518/SBI_Lab/COSMAX/Result/Age_fisher_exact_test_one_sided"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# PREPARE DATA
# ===============================
df = merged_data.copy()

# Enforce numeric
df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")

# Assign AgeGroup STRICTLY
def assign_agegroup(age):
    if YOUNG_RANGE[0] <= age <= YOUNG_RANGE[1]:
        return "Young"
    elif OLD_RANGE[0] <= age <= OLD_RANGE[1]:
        return "Old"
    else:
        return np.nan

df["AgeGroup"] = df[AGE_COL].apply(assign_agegroup)
df = df.dropna(subset=["AgeGroup"])

# ===============================
# FISHER TEST
# ===============================
results = []

for microbe in MICROBES:

    if microbe not in df.columns:
        print(f"❌ {microbe} not found, skipping")
        continue

    # Define presence
    df["MicrobePresence"] = (df[microbe] > 0).astype(int)

    # Contingency table
    contingency = pd.crosstab(
        df["MicrobePresence"],
        df["AgeGroup"]
    )

    # Ensure full 2x2
    contingency = contingency.reindex(
        index=[0, 1],
        columns=["Young", "Old"],
        fill_value=0
    )

    # Table format:
    #        Young   Old
    # Absent   a      b
    # Present  c      d
    table = np.array([
        [contingency.loc[0, "Young"], contingency.loc[0, "Old"]],
        [contingency.loc[1, "Young"], contingency.loc[1, "Old"]]
    ])

    # Skip invalid tables
    if np.any(table == 0):
        odds_ratio = np.nan
        p_value = np.nan
        ci_low = np.nan
        ci_up = np.nan
        alternative = None
    else:
        # Two-sided OR
        odds_ratio, _ = fisher_exact(table, alternative="two-sided")

        # Decide direction
        if odds_ratio > 1:
            alternative = "greater"
        elif odds_ratio < 1:
            alternative = "less"
        else:
            alternative = "two-sided"

        _, p_value = fisher_exact(table, alternative=alternative)

        # CI
        ct = Table2x2(table)
        ci_low, ci_up = ct.oddsratio_confint()

    # Save BOTH rows (Absent / Present)
    for presence_label, presence_code in [("Absent", 0), ("Present", 1)]:
        results.append({
            "Microbe": microbe,
            "Presence": presence_label,
            "Young": contingency.loc[presence_code, "Young"],
            "Old": contingency.loc[presence_code, "Old"],
            "Alternative": alternative,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
            "CI_lower": ci_low,
            "CI_upper": ci_up
        })

# ===============================
# SAVE RESULTS
# ===============================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values("p_value")

out_path = os.path.join(
    SAVE_DIR,
    "AgeGroup_Young_vs_Old_fisher_one_sided.xlsx"
)

result_df.to_excel(out_path, index=False)
print(f"✅ Fisher test saved: {out_path}")


# %%
#young_old_presence_proportionplot
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
input_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "Age_fisher_exact_test_one_sided/"
    "AgeGroup_Young_vs_Old_fisher_one_sided.xlsx"
)

output_dir = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "plots/proportion_bar_plot"
)

os.makedirs(output_dir, exist_ok=True)

# ===============================
# COLORS
# ===============================
COLOR_PRESENT = "#35703E"
COLOR_ABSENT = "#949598"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel(input_file)

microbes = df["Microbe"].unique()

# ===============================
# PLOTTING
# ===============================
for microbe in microbes:

    sub = df[df["Microbe"] == microbe].copy()

    # Counts
    young_present = sub[sub["Presence"] == "Present"]["Young"].values[0]
    young_absent  = sub[sub["Presence"] == "Absent"]["Young"].values[0]
    old_present   = sub[sub["Presence"] == "Present"]["Old"].values[0]
    old_absent    = sub[sub["Presence"] == "Absent"]["Old"].values[0]

    # Proportions
    young_total = young_present + young_absent
    old_total = old_present + old_absent

    young_present_p = young_present / young_total
    young_absent_p  = young_absent / young_total
    old_present_p   = old_present / old_total
    old_absent_p    = old_absent / old_total

    # ===============================
    # Plot
    # ===============================
    fig, ax = plt.subplots(figsize=(4.5, 5))

    ax.bar(
        ["Young", "Old"],
        [young_present_p, old_present_p],
        label="Present",
        color=COLOR_PRESENT
    )

    ax.bar(
        ["Young", "Old"],
        [young_absent_p, old_absent_p],
        bottom=[young_present_p, old_present_p],
        label="Absent",
        color=COLOR_ABSENT
    )

    # Formatting
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(microbe.replace("_", " "))
    ax.legend(frameon=False)

    plt.tight_layout()

    # ===============================
    # Save
    # ===============================
    png_path = os.path.join(
        output_dir, f"{microbe}_Young_vs_Old_proportion.png"
    )
    pdf_path = os.path.join(
        output_dir, f"{microbe}_Young_vs_Old_proportion.pdf"
    )

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"✅ Saved: {microbe}")

print("\n🎉 All proportion bar plots generated with correct colors!")


# %%

import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
input_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "skinquality_Fisher_exact_test_one_sided/OM_TE_score/"
    "streptococcus_added_skinquality_fisher_results_one_sided_with_CI_by_age_group_OM_TE_score.xlsx"
)

output_dir = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "plots/proportion_bar_plot"
)

os.makedirs(output_dir, exist_ok=True)

# ===============================
# COLORS
# ===============================
COLOR_PRESENT = "#35703E"
COLOR_ABSENT = "#949598"

# ===============================
# MICROBE → AGE WINDOW
# ===============================
microbe_age_map = {
    "Cutibacterium_acnes": "29-38",
    "Corynebacterium_propinquum": "40-61",
    "Cutibacterium_granulosum": "69-78",
    "Streptococcus_gordonii": "20-79"
}

# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel(input_file)

# ===============================
# PLOTTING
# ===============================
for microbe, age_window in microbe_age_map.items():

    sub = df[
        (df["Microbe"] == microbe) &
        (df["AgeWindow"] == age_window)
    ].copy()

    if sub.empty:
        print(f"⚠️ No data for {microbe} ({age_window}), skipping.")
        continue

    # Extract counts
    imp = sub[sub["SkinGroup"] == "Improved"].iloc[0]
    det = sub[sub["SkinGroup"] == "Deteriorated"].iloc[0]

    imp_present = imp["Present"]
    imp_absent  = imp["Absent"]
    det_present = det["Present"]
    det_absent  = det["Absent"]

    # Proportions
    imp_total = imp_present + imp_absent
    det_total = det_present + det_absent

    imp_present_p = imp_present / imp_total
    imp_absent_p  = imp_absent / imp_total
    det_present_p = det_present / det_total
    det_absent_p  = det_absent / det_total

    # ===============================
    # Plot
    # ===============================
    fig, ax = plt.subplots(figsize=(4.5, 5))

    ax.bar(
        ["Improved", "Deteriorated"],
        [imp_present_p, det_present_p],
        label="Present",
        color=COLOR_PRESENT
    )

    ax.bar(
        ["Improved", "Deteriorated"],
        [imp_absent_p, det_absent_p],
        bottom=[imp_present_p, det_present_p],
        label="Absent",
        color=COLOR_ABSENT
    )

    # Formatting
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(f"{microbe.replace('_', ' ')}\n(Age {age_window})")
    ax.legend(frameon=False)

    plt.tight_layout()

    # ===============================
    # Save
    # ===============================
    png_path = os.path.join(
        output_dir,
        f"{microbe}_{age_window}_Improved_vs_Deteriorated_proportion.png"
    )
    pdf_path = os.path.join(
        output_dir,
        f"{microbe}_{age_window}_Improved_vs_Deteriorated_proportion.pdf"
    )

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"✅ Saved: {microbe} ({age_window})")

print("\n🎉 All skin-quality proportion bar plots generated!")
# %%
# all age young, old, imp, det fisher test

# skin_quality_fisher_test
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
import os

# ===============================
# CONFIG
# ===============================
AGE_COL = "Age"
FEATURE = "OM_TE_score"

MICROBES = [
    "Cutibacterium_acnes",
    "Cutibacterium_granulosum",
    "Corynebacterium_propinquum",
    "Streptococcus_gordonii"
]

AGE_RANGE = (20, 79)

SAVE_DIR = "/home/parksb0518/SBI_Lab/COSMAX/Result/SkinQuality_fisher_exact_test_one_sided"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# PREPARE DATA
# ===============================
df = merged_data.copy()

# Enforce numeric
df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")

# Age filter
df = df[(df[AGE_COL] >= AGE_RANGE[0]) & (df[AGE_COL] <= AGE_RANGE[1])]
df = df.dropna(subset=[FEATURE])

# Define skin quality thresholds (GLOBAL)
q_high = df[FEATURE].quantile(0.60)
q_low = df[FEATURE].quantile(0.40)

def assign_skingroup(x):
    if x >= q_high:
        return "Improved"
    elif x <= q_low:
        return "Deteriorated"
    else:
        return np.nan

df["SkinGroup"] = df[FEATURE].apply(assign_skingroup)
df = df.dropna(subset=["SkinGroup"])

# ===============================
# FISHER TEST
# ===============================
results = []

for microbe in MICROBES:

    if microbe not in df.columns:
        print(f"❌ {microbe} not found, skipping")
        continue

    # Presence definition
    df["MicrobePresence"] = (df[microbe] > 0).astype(int)

    # Contingency table
    contingency = pd.crosstab(
        df["MicrobePresence"],
        df["SkinGroup"]
    )

    # Enforce strict 2x2
    contingency = contingency.reindex(
        index=[0, 1],
        columns=["Improved", "Deteriorated"],
        fill_value=0
    )

    # Table format:
    #                Improved  Deteriorated
    # Absent (0)         a          b
    # Present (1)        c          d
    table = np.array([
        [contingency.loc[0, "Improved"], contingency.loc[0, "Deteriorated"]],
        [contingency.loc[1, "Improved"], contingency.loc[1, "Deteriorated"]]
    ])

    # Skip invalid tables
    if np.any(table == 0):
        odds_ratio = np.nan
        p_value = np.nan
        ci_low = np.nan
        ci_up = np.nan
        alternative = None
    else:
        # Two-sided OR
        odds_ratio, _ = fisher_exact(table, alternative="two-sided")

        # Decide direction
        if odds_ratio > 1:
            alternative = "greater"
        elif odds_ratio < 1:
            alternative = "less"
        else:
            alternative = "two-sided"

        _, p_value = fisher_exact(table, alternative=alternative)

        # CI
        ct = Table2x2(table)
        ci_low, ci_up = ct.oddsratio_confint()

    # Save BOTH rows (Absent / Present)
    for presence_label, presence_code in [("Absent", 0), ("Present", 1)]:
        results.append({
            "Microbe": microbe,
            "Presence": presence_label,
            "Improved": contingency.loc[presence_code, "Improved"],
            "Deteriorated": contingency.loc[presence_code, "Deteriorated"],
            "Alternative": alternative,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
            "CI_lower": ci_low,
            "CI_upper": ci_up
        })

# ===============================
# SAVE RESULTS
# ===============================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values("p_value")

out_path = os.path.join(
    SAVE_DIR,
    "SkinQuality_Improved_vs_Deteriorated_fisher_one_sided.xlsx"
)

result_df.to_excel(out_path, index=False)
print(f"✅ Fisher test saved: {out_path}")

# %%
#"SkinQuality_Improved_vs_Deteriorated_fisher_one_sided.xlsx" imp/det proportion bar plot
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
input_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "SkinQuality_fisher_exact_test_one_sided/"
    "SkinQuality_Improved_vs_Deteriorated_fisher_one_sided.xlsx"
)

output_dir = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "plots/proportion_bar_plot_by_microbe"
)
os.makedirs(output_dir, exist_ok=True)

# ===============================
# COLORS
# ===============================
COLOR_PRESENT = "#35703E"
COLOR_ABSENT = "#949598"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel(input_file)

required_cols = {"Microbe", "Presence", "Improved", "Deteriorated"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"❌ Missing required columns: {required_cols}")

# ===============================
# PLOTTING
# ===============================
for microbe in df["Microbe"].unique():

    sub = df[df["Microbe"] == microbe]

    if set(sub["Presence"]) != {"Present", "Absent"}:
        print(f"⚠️ Skipping {microbe}: Present/Absent rows missing")
        continue

    # Extract rows
    present = sub[sub["Presence"] == "Present"].iloc[0]
    absent  = sub[sub["Presence"] == "Absent"].iloc[0]

    # Totals
    improved_total = present["Improved"] + absent["Improved"]
    deteriorated_total = present["Deteriorated"] + absent["Deteriorated"]

    # Proportions
    improved_present = present["Improved"] / improved_total
    improved_absent  = absent["Improved"] / improved_total

    deteriorated_present = present["Deteriorated"] / deteriorated_total
    deteriorated_absent  = absent["Deteriorated"] / deteriorated_total

    # ===============================
    # Plot
    # ===============================
    fig, ax = plt.subplots(figsize=(4.5, 5))

    ax.bar(
        ["Improved", "Deteriorated"],
        [improved_present, deteriorated_present],
        color=COLOR_PRESENT,
        label="Present"
    )

    ax.bar(
        ["Improved", "Deteriorated"],
        [improved_absent, deteriorated_absent],
        bottom=[improved_present, deteriorated_present],
        color=COLOR_ABSENT,
        label="Absent"
    )

    # Formatting
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(microbe.replace("_", " "))
    ax.legend(frameon=False)

    plt.tight_layout()

    # ===============================
    # SAVE
    # ===============================
    png_path = os.path.join(
        output_dir,
        f"{microbe}_Improved_vs_Deteriorated_proportion.png"
    )
    pdf_path = os.path.join(
        output_dir,
        f"{microbe}_Improved_vs_Deteriorated_proportion.pdf"
    )

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"✅ Saved: {microbe}")

print("\n🎉 All 4 microbe proportion bar plots generated!")


# %%
#young/old and imp/det fisher test among present sample

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
import os

# ===============================
# CONFIG
# ===============================
AGE_COL = "Age"
FEATURE = "OM_TE_score"

MICROBES = [
    "Cutibacterium_acnes",
    "Cutibacterium_granulosum",
    "Corynebacterium_propinquum",
    "Streptococcus_gordonii"
]

YOUNG_RANGE = (20, 49)
OLD_RANGE = (50, 79)

SAVE_DIR = "/home/parksb0518/SBI_Lab/COSMAX/Result/present_only_skinquality_Fisher_exact_test_microbe"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# PREPARE DATA
# ===============================
df = merged_data.copy()

# Numeric enforcement
df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")

# Assign AgeGroup
def assign_agegroup(age):
    if YOUNG_RANGE[0] <= age <= YOUNG_RANGE[1]:
        return "Young"
    elif OLD_RANGE[0] <= age <= OLD_RANGE[1]:
        return "Old"
    else:
        return np.nan

df["AgeGroup"] = df[AGE_COL].apply(assign_agegroup)
df = df.dropna(subset=["AgeGroup", FEATURE])

# Skin feature thresholds (global, same as before)
q_high = df[FEATURE].quantile(0.60)
q_low = df[FEATURE].quantile(0.40)

def assign_skingroup(x):
    if x >= q_high:
        return "Improved"
    elif x <= q_low:
        return "Deteriorated"
    else:
        return np.nan

df["SkinGroup"] = df[FEATURE].apply(assign_skingroup)
df = df.dropna(subset=["SkinGroup"])

# ===============================
# FISHER TEST (MICROBE PRESENT ONLY)
# ===============================
results = []

for microbe in MICROBES:

    if microbe not in df.columns:
        print(f"❌ {microbe} not found, skipping")
        continue

    # Keep ONLY microbe-present samples
    df_sub = df[df[microbe] > 0].copy()

    # Contingency table: AgeGroup × SkinGroup
    contingency = pd.crosstab(
        df_sub["AgeGroup"],
        df_sub["SkinGroup"]
    )

    # Enforce strict 2x2
    contingency = contingency.reindex(
        index=["Young", "Old"],
        columns=["Improved", "Deteriorated"],
        fill_value=0
    )

    table = contingency.values

    # Skip invalid tables
    if table.shape != (2, 2) or np.any(table == 0):
        odds_ratio = np.nan
        p_value = np.nan
        ci_low = np.nan
        ci_up = np.nan
        alternative = None
    else:
        # Two-sided OR
        odds_ratio, _ = fisher_exact(table, alternative="two-sided")

        # Direction decision
        if odds_ratio > 1:
            alternative = "greater"
        elif odds_ratio < 1:
            alternative = "less"
        else:
            alternative = "two-sided"

        _, p_value = fisher_exact(table, alternative=alternative)

        # Confidence interval
        ct = Table2x2(table)
        ci_low, ci_up = ct.oddsratio_confint()

    results.append({
        "Microbe": microbe,
        "Young_Improved": contingency.loc["Young", "Improved"],
        "Young_Deteriorated": contingency.loc["Young", "Deteriorated"],
        "Old_Improved": contingency.loc["Old", "Improved"],
        "Old_Deteriorated": contingency.loc["Old", "Deteriorated"],
        "Alternative": alternative,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "CI_lower": ci_low,
        "CI_upper": ci_up,
        "N_microbe_present": df_sub.shape[0]
    })

# ===============================
# SAVE RESULTS
# ===============================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values("p_value")

out_path = os.path.join(
    SAVE_DIR,
    "MicrobePresentOnly_Young_vs_Old_Improved_vs_Deteriorated_fisher.xlsx"
)

result_df.to_excel(out_path, index=False)
print(f"✅ Fisher test saved: {out_path}")

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
input_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "present_only_skinquality_Fisher_exact_test_microbe/"
    "MicrobePresentOnly_Young_vs_Old_Improved_vs_Deteriorated_fisher.xlsx"
)

output_dir = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "plots/proportion_bar_plot_present_only"
)
os.makedirs(output_dir, exist_ok=True)

# ===============================
# COLORS
# ===============================
COLOR_YOUNG = "#F5798F"
COLOR_OLD = "#872126"

COLOR_IMPROVED = "#0000E6"
COLOR_DETERIORATED = "#EAA72F"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel(input_file)

# ===============================
# PLOTTING
# ===============================
for _, row in df.iterrows():

    microbe = row["Microbe"]

    # Counts
    y_imp = row["Young_Improved"]
    y_det = row["Young_Deteriorated"]
    o_imp = row["Old_Improved"]
    o_det = row["Old_Deteriorated"]

    # Totals
    y_total = y_imp + y_det
    o_total = o_imp + o_det

    if y_total == 0 or o_total == 0:
        print(f"⚠️ Skipping {microbe}: zero total")
        continue

    # Proportions
    y_imp_p = y_imp / y_total
    y_det_p = y_det / y_total

    o_imp_p = o_imp / o_total
    o_det_p = o_det / o_total

    # ===============================
    # Plot
    # ===============================
    fig, ax = plt.subplots(figsize=(4.8, 5.2))

    x = [0, 1]  # Young, Old
    bar_width = 0.6

    # Improved (bottom)
    ax.bar(
        x[0], y_imp_p,
        width=bar_width,
        color=COLOR_IMPROVED,
        edgecolor=COLOR_YOUNG,
        linewidth=2,
        label="Improved"
    )
    ax.bar(
        x[1], o_imp_p,
        width=bar_width,
        color=COLOR_IMPROVED,
        edgecolor=COLOR_OLD,
        linewidth=2
    )

    # Deteriorated (top)
    ax.bar(
        x[0], y_det_p,
        bottom=y_imp_p,
        width=bar_width,
        color=COLOR_DETERIORATED,
        edgecolor=COLOR_YOUNG,
        linewidth=2,
        label="Deteriorated"
    )
    ax.bar(
        x[1], o_det_p,
        bottom=o_imp_p,
        width=bar_width,
        color=COLOR_DETERIORATED,
        edgecolor=COLOR_OLD,
        linewidth=2
    )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(["Young", "Old"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(microbe.replace("_", " "), fontsize=12)

    ax.legend(frameon=False)

    plt.tight_layout()

    # ===============================
    # SAVE
    # ===============================
    png_path = os.path.join(
        output_dir,
        f"{microbe}_present_only_Young_vs_Old_Improved_vs_Deteriorated.png"
    )
    pdf_path = os.path.join(
        output_dir,
        f"{microbe}_present_only_Young_vs_Old_Improved_vs_Deteriorated.pdf"
    )

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"✅ Saved: {microbe}")

print("\n🎉 All present-only proportion bar plots generated!")

# %%
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
input_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "present_only_skinquality_Fisher_exact_test_microbe/"
    "MicrobePresentOnly_Young_vs_Old_Improved_vs_Deteriorated_fisher.xlsx"
)

output_dir = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "plots/proportion_bar_plot_present_only_two_bar"
)
os.makedirs(output_dir, exist_ok=True)

# ===============================
# COLORS
# ===============================
COLOR_YOUNG = "#F5798F"
COLOR_OLD = "#872126"

COLOR_IMPROVED = "#0000E6"
COLOR_DETERIORATED = "#EAA72F"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel(input_file)

# ===============================
# PLOTTING
# ===============================
for _, row in df.iterrows():

    microbe = row["Microbe"]

    # Counts
    y_imp = row["Young_Improved"]
    y_det = row["Young_Deteriorated"]
    o_imp = row["Old_Improved"]
    o_det = row["Old_Deteriorated"]

    # Totals
    young_total = y_imp + y_det
    old_total = o_imp + o_det
    improved_total = y_imp + o_imp
    deteriorated_total = y_det + o_det

    if young_total == 0 or old_total == 0:
        print(f"⚠️ Skipping {microbe}: zero total")
        continue

    # ===============================
    # PROPORTIONS
    # ===============================
    # Age
    young_p = young_total / (young_total + old_total)
    old_p = old_total / (young_total + old_total)

    # Skin quality
    improved_p = improved_total / (improved_total + deteriorated_total)
    deteriorated_p = deteriorated_total / (improved_total + deteriorated_total)

    # ===============================
    # PLOT
    # ===============================
    fig, ax = plt.subplots(figsize=(4.8, 5.2))

    x = [0, 1]  # Age bar, Skin bar
    width = 0.6

    # --- AGE BAR ---
    ax.bar(
        x[0], young_p,
        width=width,
        color=COLOR_YOUNG,
        label="Young"
    )
    ax.bar(
        x[0], old_p,
        bottom=young_p,
        width=width,
        color=COLOR_OLD,
        label="Old"
    )

    # --- SKIN QUALITY BAR ---
    ax.bar(
        x[1], improved_p,
        width=width,
        color=COLOR_IMPROVED,
        label="Improved"
    )
    ax.bar(
        x[1], deteriorated_p,
        bottom=improved_p,
        width=width,
        color=COLOR_DETERIORATED,
        label="Deteriorated"
    )

    # ===============================
    # FORMATTING
    # ===============================
    ax.set_xticks(x)
    ax.set_xticklabels(["Age group", "Skin quality"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(microbe.replace("_", " "), fontsize=12)

    # Clean legend (no duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)

    plt.tight_layout()

    # ===============================
    # SAVE
    # ===============================
    png_path = os.path.join(
        output_dir,
        f"{microbe}_present_only_age_vs_skin_two_bar.png"
    )
    pdf_path = os.path.join(
        output_dir,
        f"{microbe}_present_only_age_vs_skin_two_bar.pdf"
    )

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"✅ Saved: {microbe}")

print("\n🎉 All two-bar proportion plots generated!")

# %%
#imp/det age group specific

import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# FILES
# ===============================
age_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "present_only_skinquality_Fisher_exact_test_microbe/"
    "MicrobePresentOnly_Young_vs_Old_Improved_vs_Deteriorated_fisher.xlsx"
)

skin_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/"
    "skinquality_Fisher_exact_test_one_sided/OM_TE_score/"
    "streptococcus_added_skinquality_fisher_results_one_sided_with_CI_by_age_group_OM_TE_score.xlsx"
)

out_dir = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/plots/"
    "two_bar_age_and_skinquality_specific_agewindow"
)
os.makedirs(out_dir, exist_ok=True)

# ===============================
# TARGET MICROBE + AGEWINDOW
# ===============================
target_map = {
    "Cutibacterium_acnes": "29-38",
    "Corynebacterium_propinquum": "40-61",
    "Cutibacterium_granulosum": "69-78"
}

# ===============================
# COLORS
# ===============================
COLOR_YOUNG = "#F5798F"
COLOR_OLD = "#872126"
COLOR_IMPROVED = "#0000E6"
COLOR_DETERIORATED = "#EAA72F"

# ===============================
# LOAD DATA
# ===============================
df_age = pd.read_excel(age_file)
df_skin = pd.read_excel(skin_file)

# ===============================
# LOOP
# ===============================
for microbe, age_window in target_map.items():

    # ---------- AGE COMPOSITION ----------
    row_age = df_age[df_age["Microbe"] == microbe]
    if row_age.empty:
        print(f"⚠️ Missing age data: {microbe}")
        continue

    row_age = row_age.iloc[0]

    young_total = row_age["Young_Improved"] + row_age["Young_Deteriorated"]
    old_total = row_age["Old_Improved"] + row_age["Old_Deteriorated"]

    age_sum = young_total + old_total
    young_p = young_total / age_sum
    old_p = old_total / age_sum

    # ---------- SKIN QUALITY (SPECIFIC AGE WINDOW) ----------
    df_sub = df_skin[
        (df_skin["Microbe"] == microbe) &
        (df_skin["AgeWindow"] == age_window) &
        #(df_skin["Presence"] == "Present") &
        (df_skin["SkinGroup"].isin(["Improved", "Deteriorated"]))
    ]

    if df_sub.shape[0] != 2:
        print(f"⚠️ Incomplete skin data: {microbe} {age_window}")
        continue

    improved_n = df_sub.loc[
        df_sub["SkinGroup"] == "Improved", "Present"
    ].values[0]

    deteriorated_n = df_sub.loc[
        df_sub["SkinGroup"] == "Deteriorated", "Present"
    ].values[0]

    skin_sum = improved_n + deteriorated_n
    improved_p = improved_n / skin_sum
    deteriorated_p = deteriorated_n / skin_sum

    # ===============================
    # PLOT
    # ===============================
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    x = [0, 1]
    width = 0.6

    # Age bar
    ax.bar(x[0], young_p, width, color=COLOR_YOUNG, label="Young")
    ax.bar(x[0], old_p, width, bottom=young_p, color=COLOR_OLD, label="Old")

    # Skin quality bar
    ax.bar(x[1], improved_p, width, color=COLOR_IMPROVED, label="Improved")
    ax.bar(
        x[1],
        deteriorated_p,
        width,
        bottom=improved_p,
        color=COLOR_DETERIORATED,
        label="Deteriorated"
    )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(["Age group", f"Skin quality\n({age_window})"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title(microbe.replace("_", " "), fontsize=12)

    # Unique legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(),
              dict(zip(labels, handles)).keys(),
              frameon=False)

    plt.tight_layout()

    # Save
    plt.savefig(
        os.path.join(out_dir, f"{microbe}_{age_window}_two_bar.png"),
        dpi=300
    )
    plt.savefig(
        os.path.join(out_dir, f"{microbe}_{age_window}_two_bar.pdf")
    )
    plt.close()

    print(f"✅ Saved: {microbe} ({age_window})")

print("\n🎉 All plots completed!")


# %%
