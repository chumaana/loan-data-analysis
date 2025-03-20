from django import forms


class LoanForm(forms.form):
    Gender = forms.ChoiceField(choices=[("Male", "Male"), ("Female", "Female")])
    Married = forms.ChoiceField(choices=[("Yes", "Yes"), ("No", "No")])
    Dependents = forms.IntegerField()
    Education = forms.ChoiceField(
        choices=[("Graduate", "Graduate"), ("Not Graduate", "Not Graduate")]
    )
    Self_Employed = forms.ChoiceField(choices=[("Yes", "Yes"), ("No", "No")])
    ApplicantIncome = forms.FloatField()
    CoapplicantIncome = forms.FloatField()
    LoanAmount = forms.FloatField()
    Loan_Amount_Term = forms.FloatField()
    Credit_History = forms.ChoiceField(choices=[(1, "Yes"), (0, "No")])
    Property_Area = forms.ChoiceField(
        choices=[("Urban", "Urban"), ("Semiurban", "Semiurban"), ("Rural", "Rural")]
    )
