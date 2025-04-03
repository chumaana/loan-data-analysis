def validate_input(data):
    """Validate input data to ensure realistic values."""
    errors = {}

    # Define valid ranges
    valid_ranges = {
        "LoanAmount": (0, 1000),  # Assuming units are in thousands
        "Loan_Amount_Term": (12, 360),  # Loan term in months
        "ApplicantIncome": (0, 100000),  # Example range for applicant income
        "CoapplicantIncome": (0, 100000),  # Example range for co-applicant income
    }

    # Check Loan Amount
    if not (
        valid_ranges["LoanAmount"][0]
        <= data["LoanAmount"]
        <= valid_ranges["LoanAmount"][1]
    ):
        errors["loan_amount"] = "Loan amount must be between $0 and $1,000,000."

    # Check Loan Term
    if not (
        valid_ranges["Loan_Amount_Term"][0]
        <= data["Loan_Amount_Term"]
        <= valid_ranges["Loan_Amount_Term"][1]
    ):
        errors["loan_amount_term"] = "Loan term must be between 12 and 360 months."

    # Check Applicant Income
    if not (
        valid_ranges["ApplicantIncome"][0]
        <= data["ApplicantIncome"]
        <= valid_ranges["ApplicantIncome"][1]
    ):
        errors["applicant_income"] = "Applicant income must be realistic."

    # Check Coapplicant Income
    if not (
        valid_ranges["CoapplicantIncome"][0]
        <= data["CoapplicantIncome"]
        <= valid_ranges["CoapplicantIncome"][1]
    ):
        errors["coapplicant_income"] = "Coapplicant income must be realistic."

    return errors  # Now returns a dictionary!
