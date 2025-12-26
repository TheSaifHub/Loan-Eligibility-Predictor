🏦 Loan Eligibility Predictor

A machine learning–based web application that predicts whether a loan is likely to be approved based on applicant financial and demographic details. The project covers the complete ML lifecycle—from data preprocessing and model training to deployment using Streamlit.

-----------------------------------------------------------------------------------------------------------------

📌 Project Overview

Financial institutions receive thousands of loan applications daily. Manually evaluating each application is time-consuming and error-prone.
This project automates the loan eligibility prediction process using machine learning to provide quick, consistent, and data-driven decisions.

The application predicts:

Loan approval status

Approval probability

Risk level (Low / Medium / High)

-----------------------------------------------------------------------------------------------------------------

🎯 Objectives

Automate loan approval prediction

Analyze the impact of income, credit history, loan amount, and demographics

Build a clean, interactive web application for real-time predictions

Ensure consistent preprocessing during training and inference

-----------------------------------------------------------------------------------------------------------------

🗂️ Dataset Description

The dataset contains applicant information such as:

Gender

Marital Status

Education

Self Employed

Number of Dependents

Applicant Income

Coapplicant Income

Loan Amount

Loan Amount Term

Credit History

Property Area

Loan Status (Target variable)

-----------------------------------------------------------------------------------------------------------------

🛠️ Tech Stack

Programming Language: Python

Data Analysis: Pandas, NumPy

Data Preprocessing: PowerTransformer (Yeo-Johnson), StandardScaler

Machine Learning Model: Logistic Regression

Web Framework: Streamlit

Model Persistence: Joblib

-----------------------------------------------------------------------------------------------------------------

⚙️ Machine Learning Pipeline

Data Cleaning & Preparation

Handling missing values

Encoding categorical variables

Outlier treatment and skewness handling

Feature Transformation

Yeo-Johnson power transformation

Standard scaling for numerical features

Model Training

Logistic Regression

Train–test split with stratification

Model evaluation using accuracy and classification report

Model Deployment

Interactive Streamlit application

Real-time predictions

Approval probability and risk visualization

-----------------------------------------------------------------------------------------------------------------

📊 Model Output

The application displays:

Loan Status: Approved / Rejected

Approval Probability (%): Confidence score

Risk Level: Low / Medium / High

This helps users understand why a loan may be approved or rejected.

-----------------------------------------------------------------------------------------------------------------

🖥️ Application Interface

Clean, responsive UI built using Streamlit

Works seamlessly on desktop and mobile

User-friendly input forms

Visual indicators for probability and risk

-----------------------------------------------------------------------------------------------------------------

🚀 How to Run the Project

1️⃣ Clone the Repository:
git clone https://github.com/TheSaifHub/Loan-Eligibility-Predictor.git ->
cd Loan-Eligibility-Predictor

2️⃣ Install Dependencies:
pip install -r requirements.txt

3️⃣ Run the Streamlit App:
python -m streamlit run app.py

-----------------------------------------------------------------------------------------------------------------

📌 Key Learnings

Importance of consistent preprocessing between training and inference

Handling skewed financial data using power transformations

Building end-to-end ML systems, not just models

Designing ML applications with both technical correctness and UX in mind

-----------------------------------------------------------------------------------------------------------------

🔮 Future Enhancements

Use advanced models (Random Forest, XGBoost)

Add feature importance explanations

Deploy on Streamlit Cloud or Render

Improve model performance with hyperparameter tuning

-----------------------------------------------------------------------------------------------------------------

🤝 Acknowledgements

This project was built as a hands-on practice to understand real-world machine learning workflows and deployment using Streamlit.

-----------------------------------------------------------------------------------------------------------------

📬 Contact

If you have any suggestions or feedback, feel free to connect!

⭐ If you like this project, don’t forget to give it a star on GitHub!
