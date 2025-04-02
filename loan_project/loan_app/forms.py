from django import forms


class LoanForm(forms.Form):
    gender = forms.ChoiceField(
        choices=[("Male", "Male"), ("Female", "Female")],
        label="Gender",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    married = forms.ChoiceField(
        choices=[("Yes", "Married"), ("No", "Single")],
        label="Marital Status",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    dependents = forms.IntegerField(
        label="Number of Dependents",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Enter number of dependents"}
        ),
    )
    education = forms.ChoiceField(
        choices=[("Graduate", "Graduate"), ("Not Graduate", "Not Graduate")],
        label="Education Level",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    self_employed = forms.ChoiceField(
        choices=[("Yes", "Yes"), ("No", "No")],
        label="Self-Employed",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    applicant_income = forms.FloatField(
        label="Applicant Income",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Enter applicant income"}
        ),
    )
    coapplicant_income = forms.FloatField(
        label="Co-applicant Income",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Enter co-applicant income"}
        ),
    )
    loan_amount = forms.FloatField(
        label="Loan Amount",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Enter loan amount"}
        ),
    )
    loan_amount_term = forms.FloatField(
        label="Loan Term (Months)",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"class": "form-control", "placeholder": "Enter loan term in months"}
        ),
    )
    credit_history = forms.ChoiceField(
        choices=[(1, "Yes"), (0, "No")],
        label="Credit History",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    property_area = forms.ChoiceField(
        choices=[("Urban", "Urban"), ("Semiurban", "Semiurban"), ("Rural", "Rural")],
        label="Property Area",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
