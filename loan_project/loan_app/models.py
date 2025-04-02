from django.db import models


class LoanRequest(models.Model):
    name = models.CharField(max_length=100)
    gender = models.CharField(
        max_length=10, choices=[("Male", "Male"), ("Female", "Female")]
    )
    married = models.CharField(max_length=3, choices=[("Yes", "Yes"), ("No", "No")])
    applicant_income = models.FloatField()
    loan_amount = models.FloatField()
    status = models.CharField(
        max_length=10, choices=[("Approved", "Approved"), ("Rejected", "Rejected")]
    )
    created_at = models.DateTimeField(auto_now_add=True)
