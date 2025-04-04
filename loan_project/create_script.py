import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


def prepared_data(dataframe: pd.DataFrame) -> tuple[DataFrame, list[str], list[str]]:
    dataframe["Gender"] = dataframe["Gender"].fillna(dataframe["Gender"].mode()[0])
    dataframe["Married"] = dataframe["Married"].fillna(dataframe["Married"].mode()[0])
    dataframe["Dependents"] = dataframe["Dependents"].fillna(
        dataframe["Dependents"].mode()[0]
    )
    dataframe["Dependents"] = dataframe["Dependents"].replace("3+", "5").astype(int)
    dataframe["Self_Employed"] = dataframe["Self_Employed"].fillna(
        dataframe["Self_Employed"].mode()[0]
    )
    dataframe["Credit_History"] = dataframe["Credit_History"].fillna(
        dataframe["Credit_History"].mode()[0]
    )
    dataframe["LoanAmount"] = dataframe["LoanAmount"].fillna(
        dataframe["LoanAmount"].mean()
    )
    dataframe["Loan_Amount_Term"] = dataframe["Loan_Amount_Term"].fillna(
        dataframe["Loan_Amount_Term"].mean()
    )

    categorical_features = [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    numerical_features = [
        "Dependents",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
    ]
    return dataframe, categorical_features, numerical_features


def train_model(train_data: tuple) -> GridSearchCV:
    dataframe, categorical_features, numerical_features = train_data
    X = dataframe.drop(columns=["Loan_Status"]).drop(columns=["Loan_ID"])
    y = dataframe["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    feature_importances = best_pipeline.named_steps["classifier"].feature_importances_
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )

    # Plot Feature Importance
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance_df.sort_values(by="Importance", ascending=False),
        color="#920f0f",
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    # Save plot to file
    output_path = "static/loan_app/feature_importance.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return grid_search.best_estimator_


def main():
    df = pd.read_csv("loan_data.csv")
    data = prepared_data(df)
    model = train_model(data)
    joblib.dump(model, "ml_model/best_model.pkl")
    print("Model training and saving completed")


if __name__ == "__main__":
    print("Running creation script...")
    main()
