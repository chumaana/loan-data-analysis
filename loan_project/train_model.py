# Import necessary libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTENC

# Load dataset
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

# Identify categorical and numerical features
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

# Define the pipeline with preprocessing and RandomForestClassifier
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Define hyperparameters for GridSearchCV
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=5, scoring="f1_weighted"
)

# Train the model using balanced data (SMOTE-NC)
grid_search.fit(X_train_balanced, y_train_balanced)

# Save the best model to a file for deployment
joblib.dump(grid_search.best_estimator_, "ml_model/best_model.pkl")
print("✅ Best model saved successfully.")

# Print best parameters from GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nModel Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
