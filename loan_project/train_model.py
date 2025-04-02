import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# Load the prepared dataset
df = pd.read_csv("prepared_loan_data.csv")

# Drop irrelevant columns (e.g., Loan_ID)
df.drop("Loan_ID", axis=1, inplace=True)

# Split data into features (X) and target (y)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define preprocessing steps for numerical and categorical features
numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
categorical_features = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Dependents",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

# Evaluate each model using cross-validation and test set performance
results = {}
for model_name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # Train the pipeline on training data
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[model_name] = {"Accuracy": accuracy, "F1 Score": f1}

# Output model comparison results
print("Model Comparison Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")

# Choose the best model based on F1 Score or Accuracy and save it using joblib
best_model_name = max(results, key=lambda x: results[x]["F1 Score"])
best_model_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("classifier", models[best_model_name])]
)
best_model_pipeline.fit(X_train, y_train)

print(f"Best Model: {best_model_name}")
joblib.dump(best_model_pipeline, "ml_model/best_model.pkl")
