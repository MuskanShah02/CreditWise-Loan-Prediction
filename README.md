# CreditWise – Loan Prediction System

## Project Overview
CreditWise is a Machine Learning web application that predicts loan approval outcomes based on applicant information.

## Features
- Loan approval prediction
- Interactive Streamlit interface
- Applicant financial information input
- Machine Learning model integration
- Automated data preprocessing

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning
- Algorithm: Logistic Regression
- Data preprocessing: One-Hot Encoding
- Feature scaling: StandardScaler

## Project Structure
- app.py – Streamlit application
- credit_wise (1).ipynb – Model development notebook
- loan_approval_data.csv – Dataset
- loan_model.pkl – Trained model
- encoder.pkl – Categorical encoder
- scaler.pkl – Feature scaler
- label_encoder.pkl – Target label encoder
- feature_names.pkl – Model feature names
- requirements.txt – Project dependencies

## Run Locally

1. Install dependencies:
   pip install -r requirements.txt

2. Start the application:
   streamlit run app.py

## Disclaimer
This project is developed for educational and demonstration purposes. Predictions are not financial advice or actual loan approval decisions.
