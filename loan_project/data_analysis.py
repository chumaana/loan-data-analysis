import pandas as pd

df = pd.read_csv("loan_data.csv")

# Fill missing values using the most common (mode) or median values for each column
fill_values = {
    "Gender": df["Gender"].mode()[0],
    "Married": df["Married"].mode()[0],
    "Dependents": df["Dependents"].mode()[0],
    "Self_Employed": df["Self_Employed"].mode()[0],
    "LoanAmount": df["LoanAmount"].median(),
    "Loan_Amount_Term": df["Loan_Amount_Term"].mode()[0],
    "Credit_History": df["Credit_History"].mode()[0],
}
df.fillna(fill_values, inplace=True)  # Apply the fill values

# Convert `Loan_Status` column from categorical ('Y'/'N') to numerical (1/0)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

df.to_csv("prepared_loan_data.csv", index=False)
print("✅ Data successfully prepared.")
