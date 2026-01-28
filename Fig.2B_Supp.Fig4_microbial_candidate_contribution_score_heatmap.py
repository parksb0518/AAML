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
# %%
#(5CV) extract the feature importance

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

# === Parameters ===
start_age = 25
end_age = 57
feature = "OM_TE_score"
random_seed = 42

# === Save path ===
window_label = f"{start_age}-{end_age}"
base_dir = f"/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/40_61_centered_specific_window"
os.makedirs(base_dir, exist_ok=True)

raw_path = os.path.join(base_dir, f"raw_5CV_{window_label}.xlsx")
summary_path = os.path.join(base_dir, f"summary_5CV_{window_label}.xlsx")
top20_path = os.path.join(base_dir, f"top20_zscore_5CV_{window_label}.xlsx")

# === Prepare data ===
merged_data[feature] = pd.to_numeric(merged_data[feature], errors="coerce")
species_columns = list(species_data.columns[1:])  # Drop sample ID

df = merged_data[(merged_data["Age"] >= start_age) & (merged_data["Age"] <= end_age)].copy()
df = df.dropna(subset=[feature])
df["Rank"] = df[feature].rank(ascending=False, method="average")
q_low, q_high = df["Rank"].quantile([0.4, 0.6])
y = df["Rank"].apply(lambda x: 1 if x <= q_low else (0 if x >= q_high else np.nan))

df = df[["mb.selected_sample-id", "Age", feature] + species_columns]
X = df.drop(columns=["mb.selected_sample-id", "Age", "Rank", feature], errors="ignore")

valid_idx = ~X.isna().any(axis=1) & ~y.isna()
X, y = X[valid_idx], y[valid_idx]

if len(X) < 10 or len(np.unique(y)) < 2:
    raise ValueError("❌ Not enough samples or only one class present in selected window.")

# === 5-fold CV ===
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
fold_aurocs, fold_auprcs = [], []
fold_records = []
zscore_list = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0.)),
        ('scale', MinMaxScaler(feature_range=(0, 1), clip=True))
    ])
    X_train_scaled = pipe.fit_transform(X_train)
    X_test_scaled = pipe.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_seed)
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    if len(np.unique(y_test)) < 2:
        continue

    # AUROC/AUPRC
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    fold_aurocs.append(auroc)
    fold_auprcs.append(auprc)

    fold_records.append({
        "Window": window_label,
        "Start_Age": start_age,
        "End_Age": end_age,
        "Fold": fold + 1,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sample_Size": len(X)
    })

    # Z-score Feature Importance
    coef = model.coef_.flatten()
    z_scores = (coef - np.mean(coef)) / np.std(coef)
    zscore_list.append(z_scores)

# === Save raw fold performance ===
pd.DataFrame(fold_records).to_excel(raw_path, index=False)

# === Save summary ===
summary_df = pd.DataFrame([{
    "Window": window_label,
    "Start_Age": start_age,
    "End_Age": end_age,
    "Sample_Size": len(X),
    "AUROC_Mean": np.mean(fold_aurocs),
    "AUROC_Std": np.std(fold_aurocs),
    "AUROC_Folds": ", ".join([f"{x:.3f}" for x in fold_aurocs])
}])
summary_df.to_excel(summary_path, index=False)

# === Aggregate Z-score across folds ===
z_array = np.vstack(zscore_list)  # shape: (5, n_features)
z_mean = np.mean(z_array, axis=0)
z_abs = np.abs(z_mean)

importance_df = pd.DataFrame({
    "Species": X.columns,
    "Z-score Mean": z_mean,
    "Abs Z-score Mean": z_abs
}).sort_values("Abs Z-score Mean", ascending=False)

# === Save top 20 features ===
importance_df.head(20).to_excel(top20_path, index=False)

print(f"✅ Saved raw CV results → {raw_path}")
print(f"✅ Saved summary         → {summary_path}")
print(f"✅ Saved top 20 features → {top20_path}")
#%%
#(5CV) conduct feature importance extraction with multiple feature in (29,38), (40,61), (69,78) group

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

# === Setup ===
age_windows = [(29, 38), (40, 61), (69, 78)]
feature_list = ["OM_TE_score"]
 #["OM_TE_score", "OM.ss", "TE.ss", "Oil.ss", "Moist.ss", "ITA.ss", "R7.ss"] 
random_seed = 42

# === Input ===
# Make sure `merged_data` and `species_data` are defined in your environment
# merged_data: contains metadata, age, feature columns, etc.
# species_data: first column = sample ID, rest = species columns

species_columns = list(species_data.columns[1:])  # exclude sample ID

# === Run for each feature and age window ===
for feature in feature_list:
    for start_age, end_age in age_windows:
        window_label = f"{start_age}-{end_age}"
        print(f"▶ Running: Feature={feature}, Age={window_label}")

        # === Save paths ===
        save_dir = f"/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/{feature}"
        os.makedirs(save_dir, exist_ok=True)
        raw_path = os.path.join(save_dir, f"raw_5CV_{window_label}.xlsx")
        summary_path = os.path.join(save_dir, f"summary_5CV_{window_label}.xlsx")
        top20_path = os.path.join(save_dir, f"top20_zscore_5CV_{window_label}.xlsx")

        # === Prepare Data ===
        merged_data[feature] = pd.to_numeric(merged_data[feature], errors="coerce")
        df = merged_data[(merged_data["Age"] >= start_age) & (merged_data["Age"] <= end_age)].copy()
        df = df.dropna(subset=[feature])
        df["Rank"] = df[feature].rank(ascending=False, method="average")
        q_low, q_high = df["Rank"].quantile([0.4, 0.6])
        y = df["Rank"].apply(lambda x: 1 if x <= q_low else (0 if x >= q_high else np.nan))

        df = df[["mb.selected_sample-id", "Age", feature] + species_columns]
        X = df.drop(columns=["mb.selected_sample-id", "Age", "Rank", feature], errors="ignore")

        valid_idx = ~X.isna().any(axis=1) & ~y.isna()
        X, y = X[valid_idx], y[valid_idx]

        if len(X) < 10 or len(np.unique(y)) < 2:
            print(f"❌ Skipped: Not enough samples or only one class for {feature} {window_label}")
            continue

        # === 5-fold CV ===
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        fold_aurocs, fold_auprcs, fold_records, zscore_list = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value=0.)),
                ('scale', MinMaxScaler(feature_range=(0, 1), clip=True))
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

            fold_records.append({
                "Window": window_label,
                "Start_Age": start_age,
                "End_Age": end_age,
                "Fold": fold + 1,
                "AUROC": auroc,
                "AUPRC": auprc,
                "Sample_Size": len(X)
            })

            coef = model.coef_.flatten()
            z_scores = (coef - np.mean(coef)) / np.std(coef)
            zscore_list.append(z_scores)

        # === Save raw results ===
        pd.DataFrame(fold_records).to_excel(raw_path, index=False)

        # === Save summary ===
        summary_df = pd.DataFrame([{
            "Window": window_label,
            "Start_Age": start_age,
            "End_Age": end_age,
            "Sample_Size": len(X),
            "AUROC_Mean": np.mean(fold_aurocs),
            "AUROC_Std": np.std(fold_aurocs),
            "AUROC_Folds": ", ".join([f"{x:.3f}" for x in fold_aurocs])
        }])
        summary_df.to_excel(summary_path, index=False)

        # === Save top 20 features ===
        z_array = np.vstack(zscore_list)
        z_mean = np.mean(z_array, axis=0)
        z_abs = np.abs(z_mean)

        importance_df = pd.DataFrame({
            "Species": X.columns,
            "Z-score Mean": z_mean,
            "Abs Z-score Mean": z_abs
        }).sort_values("Abs Z-score Mean", ascending=False)

        importance_df.head(20).to_excel(top20_path, index=False)
        #importance_df.to_excel(top20_path, index=False)

        print(f"✅ Done: {feature} [{window_label}] → {top20_path}")
#%%
#(5CV) conduct feature importance extraction with multiple feature in (29,38), (40,61), (69,78) group

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

# === Setup ===
age_windows = [(29, 38), (40, 61), (69, 78), (20, 79)]
feature_list = ["OM_TE_score"]  # ["OM_TE_score", "OM.ss", "TE.ss", "Oil.ss", "Moist.ss", "ITA.ss", "R7.ss"]
random_seed = 42

# Resident skin flora to filter
resident_flora = ["Cutibacterium", "Streptococcus", "Staphylococcus", "Rothia", "Corynebacterium", "Neisseria"]

# === Input ===
# Make sure `merged_data` and `species_data` are defined in your environment
# merged_data: contains metadata, age, feature columns, etc.
# species_data: first column = sample ID, rest = species columns

species_columns = list(species_data.columns[1:])  # exclude sample ID

# === Run for each feature and age window ===
for feature in feature_list:
    for start_age, end_age in age_windows:
        window_label = f"{start_age}-{end_age}"
        print(f"▶ Running: Feature={feature}, Age={window_label}")

        # === Save paths ===
        save_dir = f"/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/{feature}"
        os.makedirs(save_dir, exist_ok=True)
        raw_path = os.path.join(save_dir, f"raw_5CV_{window_label}.xlsx")
        summary_path = os.path.join(save_dir, f"summary_5CV_{window_label}.xlsx")
        top20_path = os.path.join(save_dir, f"top20_zscore_5CV_{window_label}.xlsx")
        all_path = os.path.join(save_dir, f"all_skin_resident_flora_zscore_5CV_{window_label}.xlsx")

        # === Prepare Data ===
        merged_data[feature] = pd.to_numeric(merged_data[feature], errors="coerce")
        df = merged_data[(merged_data["Age"] >= start_age) & (merged_data["Age"] <= end_age)].copy()
        df = df.dropna(subset=[feature])
        df["Rank"] = df[feature].rank(ascending=False, method="average")
        q_low, q_high = df["Rank"].quantile([0.4, 0.6])
        y = df["Rank"].apply(lambda x: 1 if x <= q_low else (0 if x >= q_high else np.nan))

        df = df[["mb.selected_sample-id", "Age", feature] + species_columns]
        X = df.drop(columns=["mb.selected_sample-id", "Age", "Rank", feature], errors="ignore")

        valid_idx = ~X.isna().any(axis=1) & ~y.isna()
        X, y = X[valid_idx], y[valid_idx]

        if len(X) < 10 or len(np.unique(y)) < 2:
            print(f"❌ Skipped: Not enough samples or only one class for {feature} {window_label}")
            continue

        # === 5-fold CV ===
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        fold_aurocs, fold_auprcs, fold_records, zscore_list = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value=0.)),
                ('scale', MinMaxScaler(feature_range=(0, 1), clip=True))
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

            fold_records.append({
                "Window": window_label,
                "Start_Age": start_age,
                "End_Age": end_age,
                "Fold": fold + 1,
                "AUROC": auroc,
                "AUPRC": auprc,
                "Sample_Size": len(X)
            })

            coef = model.coef_.flatten()
            z_scores = (coef - np.mean(coef)) / np.std(coef)
            zscore_list.append(z_scores)

        # === Save raw results ===
        pd.DataFrame(fold_records).to_excel(raw_path, index=False)

        # === Save summary ===
        summary_df = pd.DataFrame([{
            "Window": window_label,
            "Start_Age": start_age,
            "End_Age": end_age,
            "Sample_Size": len(X),
            "AUROC_Mean": np.mean(fold_aurocs),
            "AUROC_Std": np.std(fold_aurocs),
            "AUROC_Folds": ", ".join([f"{x:.3f}" for x in fold_aurocs])
        }])
        summary_df.to_excel(summary_path, index=False)

        # === Save top 20 features ===
        z_array = np.vstack(zscore_list)
        z_mean = np.mean(z_array, axis=0)
        z_abs = np.abs(z_mean)

        importance_df = pd.DataFrame({
            "Species": X.columns,
            "Z-score Mean": z_mean,
            "Abs Z-score Mean": z_abs
        }).sort_values("Abs Z-score Mean", ascending=False)

        # Filter top 20 to include only resident skin flora
        importance_df_filtered = importance_df[
            importance_df["Species"].apply(lambda x: any(flora in x for flora in resident_flora))
        ].sort_values("Abs Z-score Mean", ascending=False)

        importance_df_filtered.head(20).to_excel(top20_path, index=False)
        importance_df_filtered.to_excel(all_path, index=False)

        print(f"✅ Done: {feature} [{window_label}] → {top20_path}")


#%%
#Top10 feature importance species heatmap within three age group (window) (ordered_species)

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Input Directory ===
feature_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/OM_TE_score"

# === Custom colormap ===
custom_cmap = LinearSegmentedColormap.from_list(
    "yellow_white_blue",
    ["#EAA72F", "white", "#0000E6"],
    N=256
)

# === Load files ===
all_files = sorted(glob.glob(os.path.join(feature_dir, "top20_*")))

# Collect species in order of appearance in each top10 (no duplicates)
ordered_species = []
window_zscore_dict = {}

for file in all_files:
    window_label = os.path.basename(file).replace("top20_zscore_5CV_", "").replace(".xlsx", "")
    df = pd.read_excel(file)
    df["Abs Z-score Mean"] = df["Z-score Mean"].abs()

    # Get top10 species by absolute Z-score
    top10_df = df.sort_values("Abs Z-score Mean", ascending=False).head(5)
    for species in top10_df["Species"]:
        if species not in ordered_species:
            ordered_species.append(species)

    # Store full Z-scores
    z_scores = df.set_index("Species")["Z-score Mean"]
    window_zscore_dict[window_label] = z_scores

# === Build heatmap matrix ===
heatmap_data = pd.DataFrame(index=ordered_species)

for window in sorted(window_zscore_dict.keys()):
    values = [window_zscore_dict[window].get(species, float('nan')) for species in ordered_species]
    heatmap_data[window] = values

# === Plot ===
plt.figure(figsize=(max(7, 0.3 * len(all_files)), max(7, 0.2 * len(ordered_species))))
sns.heatmap(
    heatmap_data,
    cmap=custom_cmap,
    center=0,
    linewidths=0.5,
    annot=False,
    cbar_kws={"label": "Z-score Mean"}
)

#plt.title("Top Species Z-score Mean by Age Window", fontsize=14)
plt.xlabel("Age Window")
plt.ylabel("Species (Top5 union)")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# === Save plot ===
save_base = os.path.join(feature_dir, "resident_flora_feature_importance_heatmap_fixed_order")
plt.savefig(f"{save_base}.png", dpi=300)
plt.savefig(f"{save_base}.pdf")
plt.show()
plt.close()

print(f"✅ Saved heatmap to: {save_base}.png and {save_base}.pdf")


# %%
# Top Species Feature Importance Heatmap with Clustering
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Input Directory ===
feature_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/OM_TE_score"

# === Custom colormap ===
custom_cmap = LinearSegmentedColormap.from_list(
    "yellow_white_blue",
    ["#EAA72F", "white", "#0000E6"],
    N=256
)

# === Load files ===
all_files = sorted(glob.glob(os.path.join(feature_dir, "top20_*")))

# Collect species in order of appearance in each top5 top20 (union)
ordered_species = []
window_zscore_dict = {}

for file in all_files:
    window_label = os.path.basename(file).replace("top20_zscore_5CV_", "").replace(".xlsx", "")
    df = pd.read_excel(file)
    df["Abs Z-score Mean"] = df["Z-score Mean"].abs()

    # Get top5 species by absolute Z-score
    top5_df = df.sort_values("Abs Z-scoref Mean", ascending=False).head(5)
    for species in top5_df["Species"]:
        if species not in ordered_species:
            ordered_species.append(species)

    # Store full Z-scores for all species in the file
    z_scores = df.set_index("Species")["Z-score Mean"]
    window_zscore_dict[window_label] = z_scores

# === Build heatmap matrix ===
heatmap_data = pd.DataFrame(index=ordered_species)

for window in sorted(window_zscore_dict.keys()):
    values = [window_zscore_dict[window].get(species, float('nan')) for species in ordered_species]
    heatmap_data[window] = values

# === Plot clustered heatmap ===
sns.set(font_scale=0.8)

heatmap_data_filled = heatmap_data.fillna(0)

g = sns.clustermap(
    heatmap_data_filled,
    cmap=custom_cmap,
    center=0,
    linewidths=0.5,
    annot=False,
    figsize=(max(7, 0.3 * len(all_files)), max(7, 0.2 * len(ordered_species))),
    row_cluster=True,      # Cluster species (rows)
    col_cluster=False,     # Keep age windows in original order
    cbar_kws={"label": "Z-score Mean"}
)

# Labels
g.ax_heatmap.set_xlabel("Age Window")
g.ax_heatmap.set_ylabel("Species (Top5 union)")

plt.tight_layout()

# === Save clustered heatmap ===
save_base = os.path.join(feature_dir, "resident_flora_feature_importance_heatmap_clustered")
g.savefig(f"{save_base}.png", dpi=300)
g.savefig(f"{save_base}.pdf")
plt.show()
plt.close()

print(f"✅ Saved clustered heatmap to: {save_base}.png and {save_base}.pdf")


# %%
# Top Species Feature Importance Heatmap with Clustering and Z-score Mean bar
# Top Species Feature Importance Heatmap with Clustering (Z-score legend on right, no blank top)
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# === Input Directory ===
feature_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/ML/performance/agewindow/5CV_MinMaxScaler/specific_window/skin_resident_flora/OM_TE_score"

# === Custom colormap ===
custom_cmap = LinearSegmentedColormap.from_list(
    "yellow_white_blue",
    ["#EAA72F", "white", "#0000E6"],
    N=256
)

# === Load files ===
all_files = sorted(glob.glob(os.path.join(feature_dir, "top20_*")))

# Collect species in order of appearance in each top5 top20 (union)
ordered_species = []
window_zscore_dict = {}

for file in all_files:
    window_label = os.path.basename(file).replace("top20_zscore_5CV_", "").replace(".xlsx", "")
    df = pd.read_excel(file)
    df["Abs Z-score Mean"] = df["Z-score Mean"].abs()

    # Get top5 species by absolute Z-score
    top5_df = df.sort_values("Abs Z-score Mean", ascending=False).head(10)
    for species in top5_df["Species"]:
        if species not in ordered_species:
            ordered_species.append(species)

    # Store full Z-scores for all species in the file
    z_scores = df.set_index("Species")["Z-score Mean"]
    window_zscore_dict[window_label] = z_scores

# === Build heatmap matrix ===
heatmap_data = pd.DataFrame(index=ordered_species)

for window in sorted(window_zscore_dict.keys()):
    values = [window_zscore_dict[window].get(species, 0) for species in ordered_species]
    heatmap_data[window] = values

# === SAVE HEATMAP DATA (AFTER FILLING) ===
heatmap_data_path = os.path.join(
    feature_dir,
    "All_age_included_resident_flora_feature_importance_heatmap_data.xlsx"
)
heatmap_data.to_excel(heatmap_data_path, index=True)

# === Plot clustered heatmap ===
sns.set(font_scale=0.8)

# Adjust figure size to allow colorbar on the right
fig_width = max(6.5, 0.2 * len(all_files))   # width based on number of columns
fig_height = max(5, 0.3 * len(ordered_species))  # height based on number of species

g = sns.clustermap(
    heatmap_data,
    cmap=custom_cmap,
    center=0,
    linewidths=0.5,
    annot=False,
    figsize=(fig_width, fig_height),
    row_cluster=True,      
    col_cluster=False,     
    cbar_pos=(1.05, 0.2, 0.03, 0.8),  # colorbar on right inside figure
    cbar_kws={"label": "Z-score Mean"}
)

# Labels
g.ax_heatmap.set_xlabel("Age Window")
g.ax_heatmap.set_ylabel("Species (Top5 union)")

# Do NOT call plt.tight_layout() with clustermap
# plt.tight_layout()  

# === Save clustered heatmap ===
save_base = os.path.join(feature_dir, "All_age_included_resident_flora_feature_importance_heatmap_clustered_colorbar_right_fixed")
g.savefig(f"{save_base}.png", dpi=300)
g.savefig(f"{save_base}.pdf")
plt.show()
plt.close()

print(f"✅ Saved clustered heatmap with Z-score legend on the right to: {save_base}.png and {save_base}.pdf")

# %%
