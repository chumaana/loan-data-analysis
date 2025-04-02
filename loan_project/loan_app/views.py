from django.shortcuts import render
import joblib
from .forms import LoanForm
import pandas as pd


def predict_loan(request):
    model = joblib.load("ml_model/loan_model.pkl")
    prediction = None

    if request.method == "POST":
        form = LoanForm(request.POST)
        if form.is_valid():
            data = pd.DataFrame([form.cleaned_data])
            prediction = model.predict(data)[0]
            prediction = "Approved" if prediction == 1 else "Rejected"
    else:
        form = LoanForm()

    return render(
        request, "loan_app/predict.html", {"form": form, "prediction": prediction}
    )
