"""
Exploratory Data Analysis (EDA) - Titanic Dataset
Generates all plots used in the notebook/report and prints summary stats.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/titanic.csv")

# -----------------------------------------------------------------
# 2. Explore data structure
# -----------------------------------------------------------------
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDescribe (numeric):\n", df.describe())
print("\nDescribe (categorical):\n", df.describe(include="object"))

# -----------------------------------------------------------------
# 3. Trends, patterns & anomalies
# -----------------------------------------------------------------

# Survival rate overall
surv_rate = df["survived"].mean()
print(f"\nOverall survival rate: {surv_rate:.2%}")

# Plot 1: Survival counts
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="survived", palette=["#d64545", "#4caf78"])
plt.xticks([0, 1], ["Died", "Survived"])
plt.title("Survival Counts")
plt.tight_layout()
plt.savefig("plots/01_survival_counts.png")
plt.close()

# Plot 2: Survival by sex
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="sex", y="survived", palette="pastel")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("plots/02_survival_by_sex.png")
plt.close()

# Plot 3: Survival by class
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="pclass", y="survived", palette="viridis")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.tight_layout()
plt.savefig("plots/03_survival_by_class.png")
plt.close()

# Plot 4: Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["age"].dropna(), bins=30, kde=True, color="#4472c4")
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig("plots/04_age_distribution.png")
plt.close()

# Plot 5: Age vs Survival
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="age", hue="survived", common_norm=False, fill=True, alpha=0.4)
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("plots/05_age_vs_survival.png")
plt.close()

# Plot 6: Fare distribution & outliers
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="pclass", y="fare", palette="Set2")
plt.title("Fare by Passenger Class (outlier check)")
plt.tight_layout()
plt.savefig("plots/06_fare_by_class_boxplot.png")
plt.close()

# Plot 7: Correlation heatmap
plt.figure(figsize=(7, 6))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("plots/07_correlation_heatmap.png")
plt.close()

# Plot 8: Missing values
plt.figure(figsize=(6, 4))
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
sns.barplot(x=missing.values, y=missing.index, palette="Reds_r")
plt.title("Missing Values by Column")
plt.xlabel("Count Missing")
plt.tight_layout()
plt.savefig("plots/08_missing_values.png")
plt.close()

# Plot 9: Family size vs survival
df["family_size"] = df["sibsp"] + df["parch"] + 1
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="family_size", y="survived", palette="crest")
plt.title("Survival Rate by Family Size")
plt.tight_layout()
plt.savefig("plots/09_family_size_vs_survival.png")
plt.close()

# -----------------------------------------------------------------
# 4. Hypothesis testing
# -----------------------------------------------------------------
print("\n--- Hypothesis Tests ---")

# H1: Sex and survival are independent (Chi-square)
contingency_sex = pd.crosstab(df["sex"], df["survived"])
chi2, p_sex, dof, expected = stats.chi2_contingency(contingency_sex)
print(f"H1 - Sex vs Survival: chi2={chi2:.2f}, p={p_sex:.6f} -> "
      f"{'REJECT' if p_sex < 0.05 else 'FAIL TO REJECT'} null (independence)")

# H2: Passenger class and survival are independent (Chi-square)
contingency_class = pd.crosstab(df["pclass"], df["survived"])
chi2c, p_class, dofc, expc = stats.chi2_contingency(contingency_class)
print(f"H2 - Class vs Survival: chi2={chi2c:.2f}, p={p_class:.6f} -> "
      f"{'REJECT' if p_class < 0.05 else 'FAIL TO REJECT'} null (independence)")

# H3: Mean age differs between survivors and non-survivors (t-test)
age_survived = df.loc[df["survived"] == 1, "age"].dropna()
age_died = df.loc[df["survived"] == 0, "age"].dropna()
t_stat, p_age = stats.ttest_ind(age_survived, age_died, equal_var=False)
print(f"H3 - Age (survived vs died): t={t_stat:.2f}, p={p_age:.6f} -> "
      f"{'REJECT' if p_age < 0.05 else 'FAIL TO REJECT'} null (equal means)")

# H4: Fare differs between survivors and non-survivors (t-test)
fare_survived = df.loc[df["survived"] == 1, "fare"].dropna()
fare_died = df.loc[df["survived"] == 0, "fare"].dropna()
t_stat_f, p_fare = stats.ttest_ind(fare_survived, fare_died, equal_var=False)
print(f"H4 - Fare (survived vs died): t={t_stat_f:.2f}, p={p_fare:.6f} -> "
      f"{'REJECT' if p_fare < 0.05 else 'FAIL TO REJECT'} null (equal means)")

# -----------------------------------------------------------------
# 5. Data issues / anomalies
# -----------------------------------------------------------------
print("\n--- Data Issues ---")
print("Duplicate rows:", df.duplicated().sum())
print("Missing 'age':", df["age"].isnull().sum(), f"({df['age'].isnull().mean():.1%})")
print("Missing 'deck':", df["deck"].isnull().sum(), f"({df['deck'].isnull().mean():.1%})")
print("Missing 'embarked'/'embark_town':", df["embarked"].isnull().sum())

# Fare outliers via IQR
Q1, Q3 = df["fare"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df["fare"] < Q1 - 1.5 * IQR) | (df["fare"] > Q3 + 1.5 * IQR)]
print(f"Fare outliers (IQR method): {len(outliers)} rows")
print(f"Max fare: {df['fare'].max()}, Min fare: {df['fare'].min()} (0 fare is suspicious)")

print("\nDone. Plots saved to ./plots/")
