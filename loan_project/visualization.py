import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

STATIC_DIR = "loan_app/static/loan_app/"
os.makedirs(STATIC_DIR, exist_ok=True)
df = pd.read_csv("prepared_loan_data.csv")
sns.set_theme(style="whitegrid", palette=["#800020", "#228B22"])

# --- 1. Loan Status Distribution (after preprocessing) ---
plt.figure(figsize=(6, 6))
df["Loan_Status"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["Approved", "Rejected"],
    colors=["#228B22", "#800020"],
)
plt.title("Loan Status Distribution (Prepared Data)")
plt.savefig(f"{STATIC_DIR}/loan_status_distribution_prepared.png")
plt.close()

# --- 2. Box Plot: Loan Amount by Loan Status ---
plt.figure(figsize=(6, 6))
sns.boxplot(x="Loan_Status", y="LoanAmount", data=df)
plt.title("Loan Amount Distribution by Loan Status")
plt.savefig(f"{STATIC_DIR}/loan_amount_by_status.png")
plt.close()

# --- 3. Income Distribution ---
plt.figure(figsize=(8, 6))
sns.histplot(df["ApplicantIncome"], bins=30, kde=True, color="#800020")
plt.title("Applicant Income Distribution")
plt.savefig(f"{STATIC_DIR}/applicant_income_distribution.png")
plt.close()

# --- 4. Count Plots for Categorical Features ---
categorical_features = [
    "Education",
    "Married",
    "Self_Employed",
    "Property_Area",
    "Dependents",
]

for feature in categorical_features:
    plt.figure(figsize=(6, 6))
    sns.countplot(x=feature, hue="Loan_Status", data=df)
    plt.title(f"Loan Distribution by {feature}")
    plt.savefig(f"{STATIC_DIR}/{feature}_distribution.png")
    plt.close()

# --- 5. Correlation Heatmap (After Preprocessing) ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="RdYlGn")
plt.title("Correlation Between Numerical Features (Prepared Data)")
plt.savefig(f"{STATIC_DIR}/correlation_matrix_prepared.png")
plt.close()

print("✅ Visualizations for prepared data saved successfully!")
