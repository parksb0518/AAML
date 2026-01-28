# %% 
## read the data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import os

# 🔹 파일 경로 설정
data_dir = "/home/parksb0518/SBI_Lab/COSMAX"
clinical_file = os.path.join(data_dir, "피부 임상 측정.xlsx")
species_file = os.path.join(data_dir, "마이크로바이옴 분석 데이터.xlsx")

# 🔹 데이터 불러오기 (클리닉 및 마이크로바이옴 데이터)
clinical_data = pd.read_excel(clinical_file, sheet_name="clinical_KSC_994ea", header=1)
species_data = pd.read_excel(species_file, sheet_name="SILVA138v_Species")

# 🔹 Species 데이터 전처리
species_data = species_data.iloc[:, 2:]  # 첫 두 개 컬럼 제거
species_data = species_data.drop_duplicates(subset=["Species"], keep="first")  # 중복 제거
species_data = species_data[~species_data["Species"].isin(["unidentified", "uncultured"])]  # 미확인된 species 제거

# 🔹 Species 데이터를 샘플별로 정리 (Transpose)
species_data.set_index("Species", inplace=True)
species_data = species_data.T.reset_index().rename(columns={"index": "mb.selected_sample-id"})

# Drop columns where the column name contains 'unidentified' or 'uncultured'
species_data = species_data.loc[:, ~species_data.columns.str.contains("unidentified|uncultured", case=False)]

# 🔹 임상 데이터와 마이크로바이옴 데이터 병합
merged_data = pd.merge(clinical_data, species_data, on="mb.selected_sample-id", how="inner")
merged_data = merged_data.dropna()
# %%

#minmaxscaling

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

## residual 

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
#All age vs skin quality score with line 
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# === Settings ===
save_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result"
os.makedirs(save_dir, exist_ok=True)

feature = "OM_TE_score"

deter_color = "#EAA72F"   # orange
improv_color = "#0000E6"  # blue

# === Data prep ===
df_all = merged_data.dropna(subset=[feature]).copy()
df_all = df_all[(df_all["Age"] >= 20) & (df_all["Age"] <= 70)]

# === Regression fit ===
X = sm.add_constant(df_all["Age"])   # add intercept
y = df_all[feature]
model = sm.OLS(y, X).fit()
df_all["pred"] = model.predict(X)

# Residuals = distance from regression line
df_all["residual"] = df_all[feature] - df_all["pred"]

# Percentile thresholds
low_cut = df_all["residual"].quantile(0.4)   # bottom 40%
high_cut = df_all["residual"].quantile(0.6)  # top 40%

# Assign group
df_all["group"] = "middle"
df_all.loc[df_all["residual"] <= low_cut, "group"] = "deteriorated"
df_all.loc[df_all["residual"] >= high_cut, "group"] = "improved"

# === Plot ===
plt.figure(figsize=(12, 6))

# Regression line
ages_sorted = np.linspace(df_all["Age"].min(), df_all["Age"].max(), 200)
X_line = sm.add_constant(ages_sorted)
y_line = model.predict(X_line)
plt.plot(ages_sorted, y_line, color="black", lw=2, label="Regression line")

# Scatter points by group
plt.scatter(df_all.loc[df_all["group"] == "middle", "Age"],
            df_all.loc[df_all["group"] == "middle", feature],
            color="lightgrey", s=30, alpha=0.6, label="Middle 20%")

plt.scatter(df_all.loc[df_all["group"] == "improved", "Age"],
            df_all.loc[df_all["group"] == "improved", feature],
            color=improv_color, s=50, alpha=0.9, label="Improved (Top 40%)")

plt.scatter(df_all.loc[df_all["group"] == "deteriorated", "Age"],
            df_all.loc[df_all["group"] == "deteriorated", feature],
            color=deter_color, s=50, alpha=0.9, label="Deteriorated (Bottom 40%)")

plt.xlabel("Age")
plt.ylabel(feature)
plt.title(f"{feature} vs Age with Regression & Residual-based Groups (40% rule)")
plt.legend()
#plt.xlim(20, 70)
plt.tight_layout()

# Save
save_path = os.path.join(save_dir, f"{feature}_regression_residuals_topbottom40percent.pdf")
#plt.savefig(save_path, dpi=300)
plt.show()
plt.close()
print(f"✅ Saved: {save_path}")
# %%
