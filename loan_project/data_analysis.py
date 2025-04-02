import pandas as pd

df = pd.read_csv("loan_data.csv")

# Fill missing values
fill_values = {
    "Gender": df["Gender"].mode()[0],
    "Married": df["Married"].mode()[0],
    "Dependents": df["Dependents"].mode()[0],
    "Self_Employed": df["Self_Employed"].mode()[0],
    "LoanAmount": df["LoanAmount"].median(),
    "Loan_Amount_Term": df["Loan_Amount_Term"].mode()[0],
    "Credit_History": df["Credit_History"].mode()[0],
}
df.fillna(fill_values, inplace=True)

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].clip(lower=12, upper=360)
df["LoanAmount"] = df["LoanAmount"].clip(upper=1000)

df.to_csv("prepared_loan_data.csv", index=False)
print("✅ Data successfully prepared.")
