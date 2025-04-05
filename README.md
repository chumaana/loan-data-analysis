# Loan Prediction System

This project is a Django-based web application that predicts loan approval based on user input. It uses a machine learning model trained on loan data to make predictions and provides visualizations of the dataset.

---

## Features

- **Loan Approval Prediction**: Users can input loan-related details to predict whether their loan will be approved or rejected.
- **Dataset Visualizations**: Explore visual insights from the loan dataset, including:
  - Loan status distribution.
  - Loan amount distribution.
  - Applicant income vs loan status.
  - Credit history impact on loan approval.
  - Correlation matrix of numerical features.

---

## Installation

### Prerequisites

- Python 3.12 or later
- pip
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**:

```
git clone https://github.com/chumaana/loan-prediction.git
cd loan-prediction
```

2. **Create a virtual environment (optional)**:

```
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:

```
pip install -r requirements.txt
```

4. **Train the model**:

- Place your dataset (`loan_data.csv`) in the root directory.
- Run the training script to preprocess data, train the model, and save it:

  ```
  python create_script.py
  ```

5. **Start the development server**:

```
python manage.py runserver
```

6. **Access the application**:
   Open your browser and navigate to:
   http://127.0.0.1:8000/

---

## Usage

### Loan Prediction Form

1. Navigate to the homepage (`http://127.0.0.1:8000/`) or to (`http://127.0.0.1:8000/predict`).
2. Fill out the form with applicant details such as gender, income, loan amount, etc.
3. Submit the form to receive a prediction (`Approved` or `Rejected`) along with an approval probability.

### Visualizations Page

1. Navigate to `/graphs` (or click "Visualizations" in the navigation bar).
2. Explore various plots related to loan data for insights into approval trends and feature correlations.

---

## Dependencies

See `requirements.txt` for all required Python packages:

## Author

Anastasiia Chumak
https://github.com/chumaana
