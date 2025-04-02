from django import forms


class LoanForm(forms.Form):
    gender = forms.ChoiceField(choices=[("Male", "Male"), ("Female", "Female")])
    married = forms.ChoiceField(choices=[("Yes", "Yes"), ("No", "No")])
    dependents = forms.ChoiceField(
        choices=[("0", "0"), ("1", "1"), ("2", "2"), ("3+", "3+")]
    )
    education = forms.ChoiceField(
        choices=[("Graduate", "Graduate"), ("Not Graduate", "Not Graduate")]
    )
    self_employed = forms.ChoiceField(choices=[("Yes", "Yes"), ("No", "No")])

    applicant_income = forms.FloatField()
    coapplicant_income = forms.FloatField()
    loan_amount = forms.FloatField()
    loan_amount_term = forms.FloatField()

    credit_history = forms.ChoiceField(choices=[("1", "Yes"), ("0", "No")])
    property_area = forms.ChoiceField(
        choices=[("Urban", "Urban"), ("Semiurban", "Semiurban"), ("Rural", "Rural")]
    )
