from django.shortcuts import render
import joblib
import pandas as pd
from .forms import LoanForm
from .validators import validate_input  # Import the validation function


def home(request):
    """Render the home page with the loan prediction form."""
    form = LoanForm()
    return render(request, "loan_app/home.html", {"form": form})


from django.shortcuts import render
import joblib
import pandas as pd


def predict_loan(request):
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
                    return render(request, "loan_app/error.html", {"error": errors})

                # Convert input data to DataFrame
                X_pred = pd.DataFrame([input_data])

                # Encode categorical features
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

                # Predict using the model pipeline
                probabilities = model.predict_proba(X_pred)[0]
                approval_probability = probabilities[1] * 100

                # Adjust classification threshold
                threshold = 70.0
                if approval_probability >= threshold:
                    result_message = (
                        f"Approved with a {approval_probability:.2f}% probability."
                    )
                else:
                    result_message = f"Rejected with a {100 - approval_probability:.2f}% probability."

                return render(
                    request,
                    "loan_app/result.html",
                    {
                        "result_message": result_message,
                        "input_data": input_data,
                    },
                )

            except Exception as e:
                return render(
                    request,
                    "loan_app/error.html",
                    {"error": f"Prediction error: {str(e)}"},
                )

    else:
        form = LoanForm()

    return render(request, "loan_app/home.html", {"form": form})
