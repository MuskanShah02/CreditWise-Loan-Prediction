
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CreditWise | Loan Intelligence",
    page_icon="◆",
    layout="wide"
)

# ---------- PREMIUM DESIGN ----------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #081522;
    color: #F1F5F9;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    padding: 38px 0 30px 0;
    border-bottom: 1px solid #203849;
    margin-bottom: 30px;
}

.brand {
    color: #22D3EE;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 4px;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: clamp(38px, 6vw, 62px);
    line-height: 1.12;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -2px;
}

.accent {
    color: #22D3EE;
}

.hero p {
    color: #9CB0C1;
    font-size: 16px;
    max-width: 650px;
    line-height: 1.8;
    margin-top: 18px;
}

.section-heading {
    color: #22D3EE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin: 22px 0 15px 0;
}

div[data-testid="stForm"] {
    background: #10283A;
    border: 1px solid #244456;
    border-radius: 16px;
    padding: 25px;
}

label {
    color: #D7E5EF !important;
    font-size: 13px !important;
}

div[data-testid="stNumberInput"] input {
    background: #0B1D2B;
    color: white;
    border: 1px solid #34536A;
    border-radius: 8px;
}

div[data-baseweb="select"] > div {
    background: #0B1D2B;
    border: 1px solid #34536A;
    border-radius: 8px;
}

.stFormSubmitButton > button {
    background: #22D3EE;
    color: #081522;
    border: none;
    border-radius: 9px;
    font-weight: 800;
    height: 48px;
    width: 100%;
    letter-spacing: 1px;
    transition: 0.2s;
}

.stFormSubmitButton > button:hover {
    background: #67E8F9;
    color: #081522;
    border: none;
    transform: translateY(-2px);
}

.result {
    background: #10283A;
    border-radius: 16px;
    padding: 30px;
    margin-top: 30px;
    border: 1px solid #34536A;
    text-align: center;
}

.approved {
    border-color: #22C55E;
}

.rejected {
    border-color: #F87171;
}

.result-label {
    color: #22D3EE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}

.result h2 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin: 14px 0;
}

.result p {
    color: #A8BDCC;
}

.footer {
    text-align: center;
    color: #71889A;
    font-size: 12px;
    border-top: 1px solid #203849;
    margin-top: 55px;
    padding-top: 22px;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------

@st.cache_resource
def load_objects():
    model = joblib.load("loan_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")

    return model, encoder, scaler, label_encoder, feature_names


model, encoder, scaler, label_encoder, feature_names = load_objects()


# ---------- HERO ----------

st.markdown("""
<div class="hero">
    <div class="brand">◆ CREDITWISE</div>
    <h1>Smarter Lending Decisions.</h1>
    <h1 class="accent">Powered by Intelligence.</h1>
    <p>
        Evaluate applicant information with machine learning
        and generate an instant loan approval prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- INPUT FORM ----------

st.markdown(
    '<div class="section-heading">01 / APPLICANT PROFILE</div>',
    unsafe_allow_html=True
)

with st.form("loan_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )

    with c2:
        dependents = st.number_input(
            "Dependents", min_value=0, value=0
        )
        marital_status = st.selectbox(
            "Marital Status", ["Married", "Single"]
        )
        employment = st.selectbox(
            "Employment Status",
            ["Contract", "Salaried", "Self-employed", "Unemployed"]
        )

    with c3:
        employer_category = st.selectbox(
            "Employer Category",
            ["Business", "Government", "MNC", "Private", "Unemployed"]
        )
        property_area = st.selectbox(
            "Property Area", ["Rural", "Semiurban", "Urban"]
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["Business", "Car", "Education", "Home", "Personal"]
        )

    st.markdown(
        '<div class="section-heading">02 / FINANCIAL INFORMATION</div>',
        unsafe_allow_html=True
    )

    c4, c5, c6 = st.columns(3)

    with c4:
        applicant_income = st.number_input(
            "Applicant Income", min_value=0.0, value=5000.0
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0.0, value=0.0
        )
        savings = st.number_input(
            "Savings", min_value=0.0, value=10000.0
        )

    with c5:
        credit_score = st.number_input(
            "Credit Score", 300, 900, 650
        )
        existing_loans = st.number_input(
            "Existing Loans", min_value=0, value=0
        )
        dti_ratio = st.number_input(
            "DTI Ratio", min_value=0.0, value=0.30, step=0.01
        )

    with c6:
        collateral_value = st.number_input(
            "Collateral Value", min_value=0.0, value=20000.0
        )
        loan_amount = st.number_input(
            "Loan Amount", min_value=0.0, value=15000.0
        )
        loan_term = st.number_input(
            "Loan Term", min_value=1, value=60
        )

    st.write("")
    submitted = st.form_submit_button(
        "◆  ANALYZE APPLICATION"
    )


# ---------- PREDICTION ----------

if submitted:

    education_mapping = {
        "Graduate": 0,
        "Not Graduate": 1
    }

    numeric_data = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_mapping[education],
        "DTI_Ratio_sq": dti_ratio ** 2,
        "Credit_Score_sq": credit_score ** 2
    }])

    categorical_data = pd.DataFrame([{
        "Employment_Status": employment,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category
    }])

    encoded_array = encoder.transform(categorical_data)

    encoded_data = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out()
    )

    input_data = pd.concat(
        [numeric_data, encoded_data],
        axis=1
    )

    input_data = input_data.reindex(columns=feature_names)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    result = label_encoder.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(scaled_data)[0]
    confidence = max(probabilities) * 100

    st.markdown(
        '<div class="section-heading">03 / MODEL ASSESSMENT</div>',
        unsafe_allow_html=True
    )

    if result == "Yes":
        st.markdown(f"""
        <div class="result approved">
            <div class="result-label">PREDICTION RESULT</div>
            <h2 style="color:#4ADE80;">✓ LOAN APPROVED</h2>
            <p>The model predicts a positive loan approval outcome.</p>
            <h2>{confidence:.1f}%</h2>
            <p>Model confidence for this prediction</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result rejected">
            <div class="result-label">PREDICTION RESULT</div>
            <h2 style="color:#F87171;">✕ LOAN NOT APPROVED</h2>
            <p>The model predicts a negative loan approval outcome.</p>
            <h2>{confidence:.1f}%</h2>
            <p>Model confidence for this prediction</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Model confidence is the model's raw predicted class score, "
        "not a guarantee or a calibrated real-world approval probability. "
        "This application is an educational demonstration."
    )


# ---------- FOOTER ----------

st.markdown("""
<div class="footer">
    CREDITWISE &nbsp; | &nbsp; MACHINE LEARNING PROJECT
    <br><br>
    Built with Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
