# %%
import pandas as pd
import os

# === Paths ===
data_dir = "/home/parksb0518/SBI_Lab/COSMAX/Data"
merged_data_file = os.path.join(data_dir, "merged_data.csv")
species_data_file = os.path.join(data_dir, "species_data.csv")

# === Save species_data ===
#species_data.to_csv(species_data_file, index=False)
print(f"✅ species_data saved to {species_data_file}")
# %%
# === Read merged_data and species_data ===
merged_data = pd.read_csv(merged_data_file)
species_data = pd.read_csv(species_data_file)

print("✅ merged_data and species_data loaded")
print("merged_data shape:", merged_data.shape)
print("species_data shape:", species_data.shape)
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

#%%
#5CV, MinMaxscaling Finding best window based on AUROC avg and STD 

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

# === Config ===
start_age, end_age = 20, 79
feature = "OM_TE_score"
age_label = f"{start_age}_{end_age}"
random_seed = 42

# === File save path ===
master_summary_dir = f"/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/{age_label}/window_step_grid_summary"
os.makedirs(master_summary_dir, exist_ok=True)

# === Preprocess feature and extract species columns ===
merged_data[feature] = pd.to_numeric(merged_data[feature], errors="coerce")
species_columns = list(species_data.columns[1:])  # Drop sample ID

summary_list = []

# === Grid over window_size and slide_step ===
for window_size in range(10, 31):  # 10 to 30
    for slide_step in range(1, window_size + 1):  # 1 to window_size
        all_window_aurocs = []
        all_fold_records = []

        # Directory for current window/step
        sub_save_dir = f"/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/{age_label}/window{window_size}_step{slide_step}"
        os.makedirs(sub_save_dir, exist_ok=True)

        for age_start in range(start_age, end_age + 1, slide_step):
            age_end = min(age_start + window_size - 1, end_age)
            window_label = f"{age_start}-{age_end}"

            df = merged_data[(merged_data["Age"] >= age_start) & (merged_data["Age"] <= age_end)].copy()
            df = df.dropna(subset=[feature])
            df["Rank"] = df[feature].rank(ascending=False, method="average")
            q_low, q_high = df["Rank"].quantile([0.4, 0.6])
            y = df["Rank"].apply(lambda x: 1 if x <= q_low else (0 if x >= q_high else np.nan))

            df = df[["mb.selected_sample-id", "Age", feature] + species_columns]
            X = df.drop(columns=["mb.selected_sample-id", "Age", "Rank", feature], errors="ignore")

            valid_idx = ~X.isna().any(axis=1) & ~y.isna()
            X, y = X[valid_idx], y[valid_idx]

            if len(X) < 10 or len(np.unique(y)) < 2:
                continue

            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            fold_aurocs = []
            fold_auprcs = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipe = Pipeline([
                    ('impute', SimpleImputer(strategy='constant', fill_value=0.)),
                    ('scaler', MinMaxScaler(feature_range=(0, 1), clip=True))
                ])
                X_train_scaled = pipe.fit_transform(X_train)
                X_test_scaled = pipe.transform(X_test)

                model = LogisticRegression(max_iter=1000, random_state=random_seed)
                model.fit(X_train_scaled, y_train)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

                if len(np.unique(y_test)) < 2:
                    continue

                auroc = roc_auc_score(y_test, y_prob)
                auprc = average_precision_score(y_test, y_prob)
                fold_aurocs.append(auroc)
                fold_auprcs.append(auprc)

                all_fold_records.append({
                    "Window": window_label,
                    "Start_Age": age_start,
                    "End_Age": age_end,
                    "Fold": fold + 1,
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "Sample_Size": len(X)
                })

            all_window_aurocs.extend(fold_aurocs)

        # === Save per-window raw performance ===
        if all_fold_records:
            raw_df = pd.DataFrame(all_fold_records)
            raw_path = os.path.join(sub_save_dir, f"raw_5CV_window{window_size}_step{slide_step}.xlsx")
            raw_df.to_excel(raw_path, index=False)

            summary_list.append({
                "Window_Size": window_size,
                "Slide_Step": slide_step,
                "Num_Folds_Total": len(all_window_aurocs),
                "AUROC_Mean": np.mean(all_window_aurocs),
                "AUROC_Std": np.std(all_window_aurocs)
            })

# === Save global AUROC summary ===
summary_df = pd.DataFrame(summary_list).sort_values(["Window_Size", "Slide_Step"])
summary_path = os.path.join(master_summary_dir, f"MinMaxScaler_5CV_{age_label}_AUROC_summary.xlsx")
summary_df.to_excel(summary_path, index=False)

print(f"✅ Saved 5-fold CV summary to: {summary_path}")

#%%

#Save the summarized AUROC avg and std of all age window

import os
import pandas as pd
import glob

# === Config ===
base_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/20_79"
output_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/All_Unique_Window_AUROC_Summary.xlsx"
# === Collect raw result files ===
raw_files = glob.glob(os.path.join(base_dir, "window*_step*", "raw_5CV_window*_step*.xlsx"))

window_seen = set()
summary_list = []

for file in raw_files:
    df = pd.read_excel(file)
    
    if df.empty or "AUROC" not in df.columns or "Window" not in df.columns:
        continue
    
    for window_label, group in df.groupby("Window"):
        if window_label in window_seen:
            continue  # Skip duplicate windows

        window_seen.add(window_label)
        aurocs = group["AUROC"].tolist()
        auroc_str = ", ".join([f"{x:.3f}" for x in aurocs])

        summary_list.append({
            "Window": window_label,
            "Start_Age": group["Start_Age"].iloc[0],
            "End_Age": group["End_Age"].iloc[0],
            "Sample_Size": group["Sample_Size"].mean(),
            "AUROC_Mean": group["AUROC"].mean(),
            "AUROC_Std": group["AUROC"].std(),
            "AUROC_Folds": auroc_str
        })

# === Save sorted summary ===
summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values(by="AUROC_Mean", ascending=False)
summary_df.to_excel(output_path, index=False)

print(f"✅ Saved full AUROC summary to: {output_path}")

#%%

#(5CV) All age minmaxscaler with adjusted OM_TE_group (distance-based)

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# === Setup ===
feature = "OM_TE_score"
start_age, end_age = 20, 79
random_seed = 42
species_columns = list(species_data.columns[1:])  # drop sample ID

# === Output Directory ===
output_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/all_age_20_79"
os.makedirs(output_dir, exist_ok=True)

# === Data Preparation ===
df = merged_data[(merged_data["Age"] >= start_age) & (merged_data["Age"] <= end_age)].copy()
df[feature] = pd.to_numeric(df[feature], errors="coerce")
df = df.dropna(subset=[feature])

# Fit linear regression to OM_TE_score ~ Age
reg = LinearRegression()
reg.fit(df[["Age"]], df[[feature]])
a = reg.coef_[0][0]
b = reg.intercept_[0]

# Compute perpendicular distance to the regression line
df["Distance"] = np.abs(a * df["Age"] - df[feature] + b) / np.sqrt(a ** 2 + 1)

# Ranking and labeling
df["DistRank"] = df["Distance"].rank(ascending=False, method="average")
q_low, q_high = df["DistRank"].quantile([0.4, 0.6])
y = df["DistRank"].apply(lambda r: 1 if r <= q_low else (0 if r >= q_high else np.nan))

# Final X and y
df = df[["mb.selected_sample-id", "Age", feature, "Distance"] + species_columns]
X = df.drop(columns=["mb.selected_sample-id", "Age", "Distance", feature], errors="ignore")

valid_idx = ~X.isna().any(axis=1) & ~y.isna()
X, y = X[valid_idx], y[valid_idx]

# Preprocessing
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=0.)),
    ("scale", MinMaxScaler(feature_range=(0, 1), clip=True))
])
X_scaled = pipe.fit_transform(X)

# === 5-Fold CV ===
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
fold_metrics = []

print("🔄 Running 5-Fold CV (Distance to regression line)...\n")

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    fold_metrics.append({
        "Fold": fold_idx,
        "Accuracy": acc,
        "F1": f1,
        "AUROC": auroc,
        "AUPRC": auprc
    })

    print(f"🧪 Fold {fold_idx}: Accuracy={acc:.3f}, F1={f1:.3f}, AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

# Save raw fold-wise performance
fold_df = pd.DataFrame(fold_metrics)
fold_df.to_excel(os.path.join(output_dir, "foldwise_performance_5CV_distance_rank.xlsx"), index=False)

# Compute mean and std
summary = {
    "Accuracy_Mean": fold_df["Accuracy"].mean(),
    "Accuracy_Std": fold_df["Accuracy"].std(),
    "F1_Mean": fold_df["F1"].mean(),
    "F1_Std": fold_df["F1"].std(),
    "AUROC_Mean": fold_df["AUROC"].mean(),
    "AUROC_Std": fold_df["AUROC"].std(),
    "AUPRC_Mean": fold_df["AUPRC"].mean(),
    "AUPRC_Std": fold_df["AUPRC"].std(),
    "Sample_Size": len(X),
    "Start_Age": start_age,
    "End_Age": end_age
}
pd.DataFrame([summary]).to_excel(os.path.join(output_dir, "performance_summary_distance_rank_5CV.xlsx"), index=False)

# === Feature Importance ===
model = LogisticRegression(max_iter=1000, random_state=random_seed)
model.fit(X_scaled, y)
coef = model.coef_.flatten()
z_scores = (coef - np.mean(coef)) / np.std(coef)
abs_z_scores = np.abs(z_scores)

importance_df = pd.DataFrame({
    "Species": X.columns,
    "Z-score Importance": z_scores,
    "Abs Z-score": abs_z_scores
}).sort_values("Abs Z-score", ascending=False)

importance_df.head(20).to_excel(os.path.join(output_dir, "top20_feature_importance_distance_rank.xlsx"), index=False)

print(f"\n✅ Saved all results to: {output_dir}")


#%%

##(5CV) random 10 age distance-based performance

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# === Setup ===
feature = "OM_TE_score"
start_age, end_age = 20, 79
random_seed = 42
np.random.seed(random_seed)
species_columns = list(species_data.columns[1:])  # drop sample ID

# === Output Directory ===
output_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/all_age_20_79/random10age"
os.makedirs(output_dir, exist_ok=True)

# === Data Preparation ===
df_all = merged_data[(merged_data["Age"] >= start_age) & (merged_data["Age"] <= end_age)].copy()
df_all[feature] = pd.to_numeric(df_all[feature], errors="coerce")
df_all = df_all.dropna(subset=[feature])

# === Randomly select 10 unique ages ===
unique_ages = df_all["Age"].unique()
selected_ages = np.random.choice(unique_ages, size=10, replace=False)
df = df_all[df_all["Age"].isin(selected_ages)].copy()

print(f"📌 Selected random 10 ages: {sorted(selected_ages.tolist())}")
print(f"🧪 Total samples selected: {len(df)}")

# === Fit linear regression: OM_TE_score ~ Age ===
reg = LinearRegression()
reg.fit(df[["Age"]], df[[feature]])
a = reg.coef_[0][0]
b = reg.intercept_[0]

# === Compute perpendicular distance to regression line ===
df["Distance"] = np.abs(a * df["Age"] - df[feature] + b) / np.sqrt(a ** 2 + 1)

# === Rank and label using 40/60 percentile ===
df["DistRank"] = df["Distance"].rank(ascending=False, method="average")
q_low, q_high = df["DistRank"].quantile([0.4, 0.6])
y = df["DistRank"].apply(lambda r: 1 if r <= q_low else (0 if r >= q_high else np.nan))

# === Feature matrix ===
df = df[["mb.selected_sample-id", "Age", feature, "Distance"] + species_columns]
X = df.drop(columns=["mb.selected_sample-id", "Age", "Distance", feature], errors="ignore")
valid_idx = ~X.isna().any(axis=1) & ~y.isna()
X, y = X[valid_idx], y[valid_idx]

# === Sanity check ===
if len(X) < 10 or len(np.unique(y)) < 2:
    raise ValueError("Not enough valid or balanced samples after filtering. Try a different seed.")

# === Pipeline ===
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=0.)),
    ("scale", MinMaxScaler(feature_range=(0, 1), clip=True))
])
X_scaled = pipe.fit_transform(X)

# === 5-Fold CV ===
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
fold_metrics = []

print("🔄 Running 5-Fold CV (random 10 ages, distance rank)...\n")

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    fold_metrics.append({
        "Fold": fold_idx,
        "Accuracy": acc,
        "F1": f1,
        "AUROC": auroc,
        "AUPRC": auprc
    })

    print(f"🧪 Fold {fold_idx}: Accuracy={acc:.3f}, F1={f1:.3f}, AUROC={auroc:.3f}, AUPRC={auprc:.3f}")

# === Save fold-wise results ===
fold_df = pd.DataFrame(fold_metrics)
fold_df.to_excel(os.path.join(output_dir, "foldwise_performance_5CV_random10age.xlsx"), index=False)

# === Summary ===
summary = {
    "Accuracy_Mean": fold_df["Accuracy"].mean(),
    "Accuracy_Std": fold_df["Accuracy"].std(),
    "F1_Mean": fold_df["F1"].mean(),
    "F1_Std": fold_df["F1"].std(),
    "AUROC_Mean": fold_df["AUROC"].mean(),
    "AUROC_Std": fold_df["AUROC"].std(),
    "AUPRC_Mean": fold_df["AUPRC"].mean(),
    "AUPRC_Std": fold_df["AUPRC"].std(),
    "Sample_Size": len(X),
    "Start_Age": start_age,
    "End_Age": end_age,
    "Selected_Ages": ", ".join(map(str, sorted(selected_ages)))
}
pd.DataFrame([summary]).to_excel(os.path.join(output_dir, "performance_summary_random10age_5CV.xlsx"), index=False)

# === Feature Importance ===
model = LogisticRegression(max_iter=1000, random_state=random_seed)
model.fit(X_scaled, y)
coef = model.coef_.flatten()
z_scores = (coef - np.mean(coef)) / np.std(coef)
abs_z_scores = np.abs(z_scores)

importance_df = pd.DataFrame({
    "Species": X.columns,
    "Z-score Importance": z_scores,
    "Abs Z-score": abs_z_scores
}).sort_values("Abs Z-score", ascending=False)

importance_df.head(20).to_excel(os.path.join(output_dir, "top20_feature_importance_random10age.xlsx"), index=False)

print(f"\n✅ All results saved to: {output_dir}")

# %%
# Supp.Fig.3C: (5CV) Boxplot of age window compare to all age (20-79) & random 10 age
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
import os

# === Paths ===
summary_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/20_79/All_Unique_Window_AUROC_Summary.xlsx"
foldwise_20_79_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/all_age_20_79/foldwise_performance_5CV_distance_rank.xlsx"
random10_path = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/all_age_20_79/random10age/foldwise_performance_5CV_random10age.xlsx"
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler"
os.makedirs(save_dir, exist_ok=True)

# === Load Data ===
df = pd.read_excel(summary_path)
fold_df_20_79 = pd.read_excel(foldwise_20_79_path)
fold_df_random10 = pd.read_excel(random10_path)

# === Compute stats for 20–79 ===
ref_folds_20_79 = fold_df_20_79["AUROC"].round(3).tolist()
ref_mean = round(np.mean(ref_folds_20_79), 3)
ref_std = round(np.std(ref_folds_20_79), 3)
ref_folds_str = ", ".join(map(str, ref_folds_20_79))

# === Clean & append true 20–79 row ===
df = df[df["Window"] != "20-79"]
df = pd.concat([df, pd.DataFrame([{
    "Window": "20-79",
    "Start_Age": 20,
    "End_Age": 79,
    "Sample_Size": 476,
    "AUROC_Mean": ref_mean,
    "AUROC_Std": ref_std,
    "AUROC_Folds": ref_folds_str
}])], ignore_index=True)

# === Add Random10Age row ===
random10_folds = fold_df_random10["AUROC"].round(3).tolist()
df = pd.concat([df, pd.DataFrame([{
    "Window": "Random10Age",
    "Start_Age": None,
    "End_Age": None,
    "Sample_Size": None,
    "AUROC_Mean": np.mean(random10_folds),
    "AUROC_Std": np.std(random10_folds),
    "AUROC_Folds": ", ".join(map(str, random10_folds))
}])], ignore_index=True)

# === Parse fold lists ===
df["AUROC_Fold_List"] = df["AUROC_Folds"].apply(lambda x: [float(v.strip()) for v in str(x).split(",") if v.strip()])

# === Select top 10 + 20–79 + Random10Age ===
top_df = df[df["Window"].isin(
    df[df["Window"] != "20-79"].nlargest(10, "AUROC_Mean")["Window"].tolist() + ["20-79", "Random10Age"]
)].copy()

# === First Plot: Top10 + 20–79 + Random10Age, compare to 20–79 ===
ref_folds = top_df[top_df["Window"] == "20-79"]["AUROC_Fold_List"].values[0]
data = top_df["AUROC_Fold_List"].tolist()
labels = top_df["Window"].tolist()
colors = ["gray" if w == "20-79" else ("black" if w == "Random10Age" else "cornflowerblue") for w in labels]

p_values = []
for win, folds in zip(labels, data):
    if win == "20-79":
        p_values.append(None)
    else:
        _, p = mannwhitneyu(folds, ref_folds, alternative='two-sided')
        p_values.append(p)

plt.figure(figsize=(max(10, len(labels) * 0.6), 6))
box = plt.boxplot(data, patch_artist=True, labels=labels)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# Annotate p-values vs 20–79
for i, p in enumerate(p_values):
    if p is None: continue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    if sig:
        plt.text(i + 1, max(data[i]) + 0.01, sig, ha='center', va='bottom', fontsize=12, color='red')

plt.xticks(rotation=45, ha="right")
plt.ylabel("AUROC (5 folds)")
plt.title("Top 10 Age Windows vs. All Age (20–79) + Random10Age\nMann–Whitney U Test vs. 20–79")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Top10_vs_20-79_with_Random10_MannWhitney.pdf"), dpi=300)
plt.show()



# %%
# Fig.2A (5CV) Boxplot of age window conventional age group (20s...70s)
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler"
os.makedirs(save_dir, exist_ok=True)

# === Target groups ===
target_order = [
    ("29-38", "AAML Young", "#F19495"),
    ("40-61", "AAML Middle", "#F13938"),
    ("69-78", "AAML Old", "#901E22"),
]

# === Decade groups ===
decade_windows = [
    ("20-29", "20s"),
    ("30-39", "30s"),
    ("40-49", "40s"),
    ("50-59", "50s"),
    ("60-69", "60s"),
    ("70-79", "70s"),
]

# === Whole age ===
whole_age = ("20-79", "Whole", "gray")

# === Build plotting list ===
plot_data = []
plot_labels = []
colors = []

# --- Add target AAML groups ---
for win, label, color in target_order:
    row = df[df["Window"] == win]
    if not row.empty:
        plot_data.append(row["AUROC_Fold_List"].values[0])
        plot_labels.append(label)
        colors.append(color)

# --- Add decade windows ---
for win, label in decade_windows:
    row = df[df["Window"] == win]
    if not row.empty:
        plot_data.append(row["AUROC_Fold_List"].values[0])
        plot_labels.append(label)
        colors.append("black")

# --- Add Whole age ---
row = df[df["Window"] == whole_age[0]]
if not row.empty:
    plot_data.append(row["AUROC_Fold_List"].values[0])
    plot_labels.append(whole_age[1])
    colors.append(whole_age[2])

# === Draw boxplot ===
plt.figure(figsize=(12, 6))

box = plt.boxplot(plot_data, patch_artist=True, labels=plot_labels)

# Coloring boxes
for patch, c in zip(box["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_edgecolor("black")
    patch.set_linewidth(1.5)

plt.ylabel("AUROC (5 folds)")
plt.title("AUROC Distribution Across Age Segments (AAML + Decades + Whole Age)")

plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# === Save PDF ===
save_path = os.path.join(save_dir, "AUROC_Boxplot_AAML_Decades_WholeAge.pdf")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved: {save_path}")
