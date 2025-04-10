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
    """
    Prepares the data for training by handling missing values and categorizing features.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing raw data.

    Returns:
        tuple: A tuple containing the processed DataFrame, a list of categorical feature names,
               and a list of numerical feature names.
    """

    print("Data before preparation:")
    print(dataframe.tail())
    # Fill missing values for categorical columns with their mode
    dataframe["Gender"] = dataframe["Gender"].fillna(dataframe["Gender"].mode()[0])
    dataframe["Married"] = dataframe["Married"].fillna(dataframe["Married"].mode()[0])
    dataframe["Dependents"] = dataframe["Dependents"].fillna(
        dataframe["Dependents"].mode()[0]
    )

    # Replace '3+' in Dependents with 5 and convert to integer
    dataframe["Dependents"] = dataframe["Dependents"].replace("3+", "5").astype(int)

    # Fill missing values
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

    print("Data after praparation:")
    print(dataframe.tail())
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
    """
    Trains a Random Forest model using a pipeline with preprocessing and hyperparameter tuning.

    Args:
        train_data (tuple): A tuple containing the processed DataFrame,
                            categorical features, and numerical features.

    Returns:
        GridSearchCV: The best estimator after hyperparameter tuning.
    """
    dataframe, categorical_features, numerical_features = train_data

    X = dataframe.drop(columns=["Loan_Status", "Loan_ID"])
    y = dataframe["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    # Create a pipeline combining preprocessing and model training
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

    # Train the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Extract the best pipeline from GridSearchCV results
    best_pipeline = grid_search.best_estimator_

    model_save_path = "ml_model/best_model.pkl"
    joblib.dump(best_pipeline, model_save_path)

    print(f"Model saved to {model_save_path}")

    # Retrieve feature importances from the trained Random Forest model
    feature_importances = best_pipeline.named_steps["classifier"].feature_importances_

    # Retrieve feature names after preprocessing transformation
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )

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

    # Save plot to file in a static directory
    output_path = "static/loan_app/feature_importance.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Feature importance graph saved to {output_path}")


if __name__ == "__main__":
    data_path = "loan_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    processed_data, categorical_features, numerical_features = prepared_data(df)

    train_data = (processed_data, categorical_features, numerical_features)
    trained_model = train_model(train_data)

    print("Model training complete.")
