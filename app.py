import streamlit as st
import pandas as pd
import joblib
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Eligibility AI",
    page_icon="🏦",
    layout="centered"
)

# ---------------- LOAD ASSETS ----------------
@st.cache_resource
def load_assets():
    model = joblib.load("pickel_files/loan_model.pkl")
    scaler = joblib.load("pickel_files/scaler.pkl")
    pt = joblib.load("pickel_files/power_transformer.pkl")
    return model, scaler, pt

model, scaler, pt = load_assets()

# ---------------- HEADER ----------------
st.title("🏦 Loan Eligibility Predictor")
st.caption("AI-powered loan approval assessment based on applicant details")
st.divider()

# ================= FORM =================
with st.form("loan_form"):

    st.subheader("👤 Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])

    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    st.divider()

    st.subheader("💰 Financial Details")

    col3, col4 = st.columns(2)
    with col3:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)

    with col4:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        loan_term = st.selectbox(
            "Loan Term (months)", [12, 36, 60, 120, 180, 240, 360], index=6
        )

    st.divider()

    st.subheader("📄 Credit Profile")
    credit_history = st.radio(
        "Do you have a credit history?",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
    )

    submitted = st.form_submit_button("🔍 Analyze Loan Eligibility")

# ================= PREDICTION =================
if submitted:
    # Encoding
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    property_rural = 1 if property_area == "Rural" else 0
    property_semiurban = 1 if property_area == "Semiurban" else 0
    property_urban = 1 if property_area == "Urban" else 0

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area_Rural": [property_rural],
        "Property_Area_Semiurban": [property_semiurban],
        "Property_Area_Urban": [property_urban],
    })

    # ---------------- TRANSFORM + SCALE ----------------
    num_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
    ]

    input_data[num_cols] = pt.transform(input_data[num_cols])
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # ---------------- PREDICTION ----------------
    with st.spinner("Analyzing applicant profile..."):
        time.sleep(0.6)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.success("✅ Loan is likely to be **APPROVED**")
    else:
        st.error("❌ Loan is likely to be **REJECTED**")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Approval Probability", f"{probability*100:.2f}%")
    with col_r2:
        st.metric(
            "Risk Level",
            "Low" if probability > 0.7 else "Medium" if probability > 0.5 else "High",
        )

    progress_value = max(1, min(99, int(round(probability * 100))))
    st.progress(progress_value)

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Streamlit & Machine Learning • Educational Demo")
