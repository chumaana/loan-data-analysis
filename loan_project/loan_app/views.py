from django.shortcuts import render
import joblib
import pandas as pd
from .forms import LoanForm
from .validators import validate_input


def home(request):
    """Render the home page with the loan prediction form."""
    form = LoanForm()
    return render(request, "loan_app/home.html", {"form": form})


def predict_loan(request):
    """Handle loan prediction form and display results."""
    form = LoanForm()
    prediction = None
    probability = None
    input_data = None

    if request.method == "POST":
        form = LoanForm(request.POST)
        if form.is_valid():
            try:
                # Load model and encoder
                model = joblib.load("ml_model/best_model.pkl")
                encoder = joblib.load("ml_model/ordinal_encoder.pkl")

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
                    # Assign errors to specific form fields
                    for field, error_msg in errors.items():
                        print("Validation Errors:", field)

                        form.add_error(field, error_msg)  # Attach errors to fields
                else:
                    # Convert input to DataFrame and encode categorical features
                    X_pred = pd.DataFrame([input_data])
                    categorical_features = [
                        "Gender",
                        "Married",
                        "Education",
                        "Self_Employed",
                        "Property_Area",
                        "Dependents",
                    ]
                    X_pred[categorical_features] = encoder.transform(
                        X_pred[categorical_features]
                    )

                    # Predict loan approval probability
                    probabilities = model.predict_proba(X_pred)[0]
                    probability = probabilities[1] * 100  # Approval probability

                    # Adjust classification threshold (e.g., 70%)
                    threshold = 70.0
                    if probability >= threshold:
                        prediction = f"Approved with a {probability:.2f}% probability."
                    else:
                        prediction = (
                            f"Rejected with a {100 - probability:.2f}% probability."
                        )

            except Exception as e:
                form.add_error(None, f"Prediction error: {str(e)}")  # Non-field error

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
    """Render the page displaying generated graphs."""
    return render(request, "loan_app/visualization.html")
