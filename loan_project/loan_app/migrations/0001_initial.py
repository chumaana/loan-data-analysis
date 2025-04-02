# Generated by Django 5.1.6 on 2025-04-02 20:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LoanRequest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female')], max_length=10)),
                ('married', models.CharField(choices=[('Yes', 'Yes'), ('No', 'No')], max_length=3)),
                ('applicant_income', models.FloatField()),
                ('loan_amount', models.FloatField()),
                ('status', models.CharField(choices=[('Approved', 'Approved'), ('Rejected', 'Rejected')], max_length=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
