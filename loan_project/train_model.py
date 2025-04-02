from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTENC
import joblib
import pandas as pd

df = pd.read_csv("prepared_loan_data.csv")

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split BEFORE balancing to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify categorical columns for SMOTE-NC and encoding
categorical_features = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Dependents",
]
numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Encode categorical features as integers for SMOTE-NC compatibility
encoder = OrdinalEncoder()
X_train[categorical_features] = encoder.fit_transform(X_train[categorical_features])
X_test[categorical_features] = encoder.transform(X_test[categorical_features])

# Save the encoder for use during prediction
joblib.dump(encoder, "ml_model/ordinal_encoder.pkl")

# Get indices of categorical columns in the DataFrame for SMOTE-NC
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_features]

# Apply SMOTE-NC to balance the training dataset
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_train_balanced, y_train_balanced = smote_nc.fit_resample(X_train, y_train)

# Preprocessing pipeline: scale numeric features and encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Define models with hyperparameters
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
    ),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine": SVC(
        probability=True, kernel="rbf", C=5, gamma="scale", random_state=42
    ),
}

# Evaluate models and store results
results = {}
for model_name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train_balanced, y_train_balanced)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    results[model_name] = {"Accuracy": accuracy, "F1 Score": f1}

# Print results for comparison
print("\nModel Comparison Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")

# Select the best model based on F1 Score and save it for deployment
best_model_name = max(results, key=lambda x: results[x]["F1 Score"])
best_model_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("classifier", models[best_model_name])]
)
best_model_pipeline.fit(X_train_balanced, y_train_balanced)

print(f"\n✅ Best Model: {best_model_name} saved successfully.")
joblib.dump(best_model_pipeline, "ml_model/best_model.pkl")
