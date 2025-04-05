import os
import base64
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from django.shortcuts import render

from create_script import prepared_data
from django.conf import settings
from django.shortcuts import render, redirect
from django.urls import reverse
import joblib
from matplotlib import pyplot as plt
import pandas as pd

from .forms import LoanForm
from .validators import validate_input


def home(request):
    """Render the home page with the loan prediction form."""
    form = LoanForm()
    return render(request, "loan_app/home.html", {"form": form})


def predict_loan(request):
    """Handle loan prediction form and display results."""
    form = LoanForm(request.POST or None)
    prediction = None
    probability = None
    input_data = None

    if request.method == "POST" and form.is_valid():
        try:
            # Load model with absolute path
            model_path = os.path.join(settings.BASE_DIR, "ml_model", "best_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found. Train the model first.")

            model = joblib.load(model_path)

            # Prepare input data
            input_data = {
                "Gender": form.cleaned_data["gender"],
                "Married": form.cleaned_data["married"],
                "Dependents": str(form.cleaned_data["dependents"]),
                "Education": form.cleaned_data["education"],
                "Self_Employed": form.cleaned_data["self_employed"],
                "ApplicantIncome": float(form.cleaned_data["applicant_income"]),
                "CoapplicantIncome": float(form.cleaned_data["coapplicant_income"]),
                "LoanAmount": float(form.cleaned_data["loan_amount"]),
                "Loan_Amount_Term": float(form.cleaned_data["loan_amount_term"]),
                "Credit_History": int(form.cleaned_data["credit_history"]),
                "Property_Area": form.cleaned_data["property_area"],
            }

            # Validate input values
            errors = validate_input(input_data)
            if errors:
                for field, error in errors.items():
                    form.add_error(field, error)
            else:
                X_pred = pd.DataFrame([input_data])

                # Make prediction and get probabilities
                prediction_value = model.predict(X_pred)[0]
                probabilities = model.predict_proba(X_pred)[0]

                prediction = "Approved" if prediction_value == 1 else "Rejected"
                probability = round(probabilities[1] * 100, 2)  # Approval probability

        except FileNotFoundError as e:
            form.add_error(None, str(e))
        except Exception as e:
            form.add_error(None, f"Prediction error: {str(e)}")

    return render(
        request,
        "loan_app/loan_prediction.html",
        {
            "form": form,
            "prediction": prediction,
            "probability": probability,
            "input_data": input_data,
        },
    )


def graphs(request):
    """
    Generate and display various visualizations related to loan data.
    """
    df = pd.read_csv("loan_data.csv")
    dataframe, categorical_features, numerical_features = prepared_data(df)

    MAIN_COLOR = "#920f0f"
    SECONDARY_COLOR = "#f5c6cb"
    sns.set_palette([MAIN_COLOR, SECONDARY_COLOR])
    sns.set_theme(style="whitegrid")

    # Plot 1: Loan Status Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Loan_Status", data=df, palette=[MAIN_COLOR, SECONDARY_COLOR])
    plt.title("Loan Approval Distribution")
    buffer_loan_status = BytesIO()
    plt.savefig(buffer_loan_status, format="png")
    buffer_loan_status.seek(0)
    loan_status_plot = base64.b64encode(buffer_loan_status.getvalue()).decode("utf-8")
    plt.close()

    # Plot 2: Loan Amount Distribution
    plt.figure(figsize=(12, 8))
    hist = sns.histplot(df["LoanAmount"], kde=True, color=MAIN_COLOR)
    hist.lines[0].set_color(MAIN_COLOR)  # KDE curve color
    plt.title("Loan Amount Distribution")
    buffer_loan_amount = BytesIO()
    plt.savefig(buffer_loan_amount, format="png")
    buffer_loan_amount.seek(0)
    loan_amount_plot = base64.b64encode(buffer_loan_amount.getvalue()).decode("utf-8")
    plt.close()

    # Plot 3: Income vs Loan Status
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x="Loan_Status",
        y="ApplicantIncome",
        data=df,
        palette=[MAIN_COLOR, SECONDARY_COLOR],
    )
    plt.title("Applicant Income vs Loan Status")
    buffer_income_vs_loan = BytesIO()
    plt.savefig(buffer_income_vs_loan, format="png")
    buffer_income_vs_loan.seek(0)
    income_vs_loan_plot = base64.b64encode(buffer_income_vs_loan.getvalue()).decode(
        "utf-8"
    )
    plt.close()

    # Plot 4: Credit History Impact
    plt.figure(figsize=(8, 6))
    sns.countplot(
        x="Credit_History",
        hue="Loan_Status",
        data=df,
        palette=[MAIN_COLOR, SECONDARY_COLOR],
    )
    plt.title("Loan Approval by Credit History")
    buffer_credit_history = BytesIO()
    plt.savefig(buffer_credit_history, format="png")
    buffer_credit_history.seek(0)
    credit_history_plot = base64.b64encode(buffer_credit_history.getvalue()).decode(
        "utf-8"
    )
    plt.close()

    # Plot 5: Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[numerical_features].corr(),
        annot=True,
        cmap=sns.light_palette(MAIN_COLOR, as_cmap=True),
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix")
    buffer_correlation_matrix = BytesIO()
    plt.savefig(buffer_correlation_matrix, format="png")
    buffer_correlation_matrix.seek(0)
    correlation_matrix_plot = base64.b64encode(
        buffer_correlation_matrix.getvalue()
    ).decode("utf-8")
    plt.close()

    return render(
        request,
        "loan_app/visualization.html",
        {
            "loan_status_plot": loan_status_plot,
            "loan_amount_plot": loan_amount_plot,
            "income_vs_loan_plot": income_vs_loan_plot,
            "credit_history_plot": credit_history_plot,
            "correlation_matrix_plot": correlation_matrix_plot,
        },
    )
