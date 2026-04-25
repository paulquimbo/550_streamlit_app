import streamlit as st
import joblib
import pandas as pd

# Load both models
model_all = joblib.load("xgb_all_readmission.pkl")
model_30 = joblib.load("xgb_lessthan30.pkl")

st.title("Hospital Readmission Prediction Tool")

st.write("""
This tool predicts two outcomes:
1. **Whether the patient is likely to be readmitted at all**
2. **Whether the patient is likely to be readmitted within 30 days**
""")

# ---------------------------------------------------------
# INPUT FIELDS
# ---------------------------------------------------------

st.header("Patient Information")

age = st.number_input("Age", min_value=0, max_value=120, value=50)
time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=5)
num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=40)
num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=20, value=1)
num_medications = st.number_input("Number of Medications", min_value=0, max_value=100, value=10)
number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=20, value=0)
number_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=20, value=0)
number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=20, value=0)
number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=5)

# Binary fields
st.header("Binary / Yes-No Information")

diabetesMed = st.selectbox("Is the patient on diabetes medication?", ["No", "Yes"])
GenderMale = st.selectbox("Gender", ["Female", "Male"])
ERAdmission = st.selectbox("Admission Type: Was it through the ER?", ["No", "Yes"])

Race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other"])
PDX_diabetes_related = st.selectbox("Primary Diagnosis Diabetes-Related?", ["No", "Yes"])

# ---------------------------------------------------------
# CONVERT TO MODEL INPUT FORMAT
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "age": [age],
    "time_in_hospital": [time_in_hospital],
    "num_lab_procedures": [num_lab_procedures],
    "num_procedures": [num_procedures],
    "num_medications": [num_medications],
    "number_outpatient": [number_outpatient],
    "number_emergency": [number_emergency],
    "number_inpatient": [number_inpatient],
    "number_diagnoses": [number_diagnoses],
    "diabetesMed": [1 if diabetesMed == "Yes" else 0],
    "GenderMale": [1 if GenderMale == "Male" else 0],
    "ERAdmission": [1 if ERAdmission == "Yes" else 0],
    "RaceCaucasian": [1 if Race == "Caucasian" else 0],
    "RaceAfricanAmerican": [1 if Race == "AfricanAmerican" else 0],
    "RaceOther": [1 if Race == "Other" else 0],
    "PDX_diabetes_related": [1 if PDX_diabetes_related == "Yes" else 0]
})

# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

if st.button("Predict Readmission Risk"):

    prob_all = model_all.predict_proba(input_data)[0][1]
    prob_30 = model_30.predict_proba(input_data)[0][1]

    st.subheader("Prediction Results")

    st.write(f"**Likelihood of ANY readmission:** {prob_all:.2f}")
    st.write(f"**Likelihood of readmission within 30 days:** {prob_30:.2f}")

    # Optional risk labels
    def risk_label(p):
        if p >= 0.70:
            return "High Risk"
        elif p >= 0.40:
            return "Moderate Risk"
        else:
            return "Low Risk"

    st.write("---")
    st.write("### Risk Interpretation")
    st.write(f"**Overall Readmission Risk:** {risk_label(prob_all)}")
    st.write(f"**<30‑Day Readmission Risk:** {risk_label(prob_30)}")
