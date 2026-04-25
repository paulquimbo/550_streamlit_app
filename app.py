import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

# -----------------------------
# Load Models
# -----------------------------
model_all = joblib.load("xgb_all_readmission.pkl")
model_30 = joblib.load("xgb_lessthan30.pkl")

# -----------------------------
# Title + Description
# -----------------------------
st.markdown("""
## 🏥 Hospital Readmission Prediction Tool  
This tool uses machine learning to predict the likelihood of hospital readmission for diabetic patients based on their demographics, clinical history, and visit details. 
### 1️⃣ Likelihood of **any** hospital readmission  
### 2️⃣ Likelihood of **readmission within 30 days**

Please enter the patient information below.
""")

st.markdown("---")

# -----------------------------
# INPUT SECTIONS WITH STYLING
# -----------------------------

st.markdown("## 🧍 Patient Demographics & Visit Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=5)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=5)
    GenderMale = st.selectbox("Gender", ["Female", "Male"])

with col2:
    diabetesMed = st.selectbox("On Diabetes Medication?", ["No", "Yes"])
    ERAdmission = st.selectbox("Admission Through ER/Urgent Care?", ["No", "Yes"])
    Race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other"])
    PDX_diabetes_related = st.selectbox("Primary Diagnosis Diabetes-Related?", ["No", "Yes"])

st.markdown("---")
st.markdown("## 🧪 Clinical & Utilization History")

col3, col4 = st.columns(2)

with col3:
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=40)
    num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=20, value=1)
    num_medications = st.number_input("Number of Medications", min_value=0, max_value=100, value=10)

with col4:
    number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=20, value=0)
    number_emergency = st.number_input("Emergency Visits", min_value=0, max_value=20, value=0)
    number_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=20, value=0)

st.markdown("---")

# -----------------------------
# Convert Inputs to Model Format
# -----------------------------
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

# -----------------------------
# Prediction Button
# -----------------------------
st.markdown("## 💡 Generate Prediction")

if st.button("Predict Readmission Risk"):
    prob_all = model_all.predict_proba(input_data)[0][1]
    prob_30 = model_30.predict_proba(input_data)[0][1]

    # Risk label helper
    def risk_label(p):
        if p >= 0.70:
            return "🔴 High Risk"
        elif p >= 0.40:
            return "🟠 Moderate Risk"
        else:
            return "🟢 Low Risk"

    st.markdown("### 📊 Prediction Results")

    st.success(f"**Likelihood of ANY readmission:** {prob_all:.2f} — {risk_label(prob_all)}")
    st.info(f"**Likelihood of readmission within 30 days:** {prob_30:.2f} — {risk_label(prob_30)}")

    st.markdown("---")
    st.markdown("### 📝 Interpretation")
    st.write("""
- **High Risk** → Consider proactive follow‑up, care coordination, or intervention  
- **Moderate Risk** → Monitor closely and review contributing factors  
- **Low Risk** → Standard discharge planning  
""")
