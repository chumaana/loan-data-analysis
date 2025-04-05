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

- Python 3.8 or later
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**:

```
git clone https://github.com/chumaana/loan-prediction.git
cd loan-prediction
```

text

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

1. Navigate to the homepage (`http://127.0.0.1:8000/`).
2. Fill out the form with applicant details such as gender, income, loan amount, etc.
3. Submit the form to receive a prediction (`Approved` or `Rejected`) along with an approval probability.

### Visualizations Page

1. Navigate to `/graphs` (or click "Visualizations" in the navigation bar).
2. Explore various plots related to loan data for insights into approval trends and feature correlations.

---

## File Structure

loan-prediction/
│
├── loan_app/ # Main Django app with views, forms, templates, etc.
│ ├── forms.py # Django forms for user input.
│ ├── validators.py # Input validation logic.
│ ├── views.py # Core application logic (prediction and graphs).
│ ├── templates/ # HTML templates for rendering pages.
│ │ ├── home.html # Homepage template with prediction form.
│ │ ├── loan_prediction.html # Template for displaying prediction results.
│ │ └── visualization.html # Template for displaying visualizations.
│ └── static/ # Static files for CSS, JS, and images.
│
├── ml_model/ # Directory for storing trained machine learning models.
│ └── best_model.pkl # Saved Random Forest model after training.
│
├── create_script.py # Script for training and saving the ML model.
├── loan_data.csv # Dataset used for training and visualization.
├── requirements.txt # List of dependencies for the project.
├── README.md # Documentation for the project.
└── manage.py # Django management script.

---

## Dependencies

See `requirements.txt` for all required Python packages:

## Author

Anastasiia Chumak
https://github.com/chumaana
