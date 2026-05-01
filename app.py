import streamlit as st
import joblib
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

# ============================================================
# LOAD MODEL
# ============================================================
model_30 = joblib.load("lessthan30_SGDClassifier_Final.Smoteenn.A1C.pkl")

# ============================================================
# DIAGNOSIS CATEGORY → BINARY MAP
# ============================================================
diag_binary_map = {
    "Endocrine/Metabolic": 1,
    "Circulatory System": 1,
    "Respiratory System": 1,
    "Neoplasms": 1,
    "Infectious and Parasitic Diseases": 1,
    "Injury/Poisoning": 1,
    "Genitourinary System": 1,
    "Nervous System": 1,
    "Blood Diseases": 1,
    "Mental Disorders": 1,

    "Symptoms/Ill-defined": 0,
    "Skin/Subcutaneous": 0,
    "Musculoskeletal": 0,
    "Digestive System": 0,
    "Pregnancy/Childbirth": 0,
    "Congenital Anomalies": 0,
    "V-Codes": 0,
    "E-Codes": 0,
    "Unknown": 0
}

diagnosis_options = list(diag_binary_map.keys())

# ============================================================
# ADMISSION TYPE NAMES + BINARY MAP
# ============================================================
admission_type_names = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped"
}

admission_type_binary_map = {
    1: 1, 2: 1, 7: 1,
    3: 0, 4: 0, 5: 0, 6: 0, 8: 0
}

# ============================================================
# DISCHARGE DISPOSITION NAMES + BINARY MAP
# ============================================================
discharge_names = {
    1: "Home",
    2: "Short-term Hospital",
    3: "Skilled Nursing Facility",
    4: "Intermediate Care Facility",
    5: "Other Inpatient Care",
    6: "Home Health",
    7: "Left Against Medical Advice",
    8: "Home IV Care",
    9: "Admitted to This Hospital",
    10: "Neonate to Another Hospital",
    11: "Expired",
    12: "Still a Patient",
    13: "Hospice - Home",
    14: "Hospice - Medical Facility",
    15: "Swing Bed",
    16: "Outpatient Services - Other Institution",
    17: "Outpatient Services - This Institution",
    18: "NULL",
    19: "Expired - Home Medicaid",
    20: "Expired - Facility Medicaid",
    21: "Expired - Unknown Medicaid",
    22: "Rehab Facility",
    23: "Long-term Care Hospital",
    24: "Medicaid Nursing Facility",
    25: "Not Mapped",
    27: "Federal Healthcare Facility",
    28: "Psychiatric Hospital",
    29: "Critical Access Hospital",
    30: "Other Healthcare Institution"
}

discharge_disposition_binary_map = {
    1: 0, 6: 0, 8: 0, 18: 0, 25: 0,
    2: 1, 3: 1, 4: 1, 5: 1, 7: 1, 9: 1, 10: 1, 11: 1,
    12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 19: 1,
    20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 27: 1, 28: 1,
    29: 1, 30: 1
}

# ============================================================
# TITLE
# ============================================================
st.markdown("""
## 🏥 Hospital Readmission Prediction Tool  
Predicts the likelihood of **readmission within 30 days** for diabetic patients.
""")

st.markdown("---")

# ============================================================
# INPUT SECTIONS
# ============================================================
st.markdown("## 🧍 Patient Demographics & Visit Details")

col1, col2 = st.columns(2)

age_map = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}

with col1:
    age_range = st.selectbox("Age Range", list(age_map.keys()))
    age = age_map[age_range]

    time_in_hospital = st.number_input("Time in Hospital (days)", 1, 30, 5)
    number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 5)

    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    GenderMale = 1 if gender == "Male" else 0

with col2:
    diabetesMed = st.selectbox("On Diabetes Medication?", ["No", "Yes"])
    diabetesMed = 0 if diabetesMed == "No" else 1

    admission_choice = st.selectbox("Admission Type", list(admission_type_names.values()))
    admission_type_id = [k for k, v in admission_type_names.items() if v == admission_choice][0]
    ERAdmission = admission_type_binary_map[admission_type_id]

    discharge_choice = st.selectbox("Discharge Disposition", list(discharge_names.values()))
    discharge_id = [k for k, v in discharge_names.items() if v == discharge_choice][0]
    DischargeRisk = discharge_disposition_binary_map[discharge_id]

    Race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other"])
    RaceCaucasian = 1 if Race == "Caucasian" else 0
    RaceAfricanAmerican = 1 if Race == "AfricanAmerican" else 0
    RaceOther = 1 if Race not in ["Caucasian", "AfricanAmerican"] else 0

st.markdown("---")

# ============================================================
# CLINICAL HISTORY
# ============================================================
st.markdown("## 🧪 Clinical & Utilization History")

col3, col4 = st.columns(2)

with col3:
    num_lab_procedures = st.number_input("Lab Procedures", 0, 200, 40)
    num_procedures = st.number_input("Procedures", 0, 20, 1)
    num_medications = st.number_input("Medications", 0, 100, 10)

with col4:
    number_outpatient = st.number_input("Outpatient Visits", 0, 20, 0)
    number_emergency = st.number_input("Emergency Visits", 0, 20, 0)
    number_inpatient = st.number_input("Inpatient Visits", 0, 20, 0)

st.markdown("---")

# ============================================================
# A1C INPUT
# ============================================================
st.markdown("## 🧬 A1C Level")

A1C_value = st.number_input("A1C (%)", min_value=3.0, max_value=15.0, value=6.0, step=0.1)
A1C_encoded = 0 if A1C_value <= 6.5 else 1

st.markdown("---")

# ============================================================
# DIAGNOSIS CATEGORIES
# ============================================================
st.markdown("## 🩺 Diagnosis Categories")

diag_1_name = st.selectbox("Primary Diagnosis Category", diagnosis_options)
diag_2_name = st.selectbox("Secondary Diagnosis Category", diagnosis_options)
diag_3_name = st.selectbox("Additional Diagnosis Category", diagnosis_options)

diag_1_chapter = diag_binary_map[diag_1_name]
diag_2_chapter = diag_binary_map[diag_2_name]
diag_3_chapter = diag_binary_map[diag_3_name]

# ============================================================
# MODEL INPUT — EXACT FEATURE ORDER
# ============================================================
input_data = pd.DataFrame([{
    'age': age,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_emergency': number_emergency,
    'number_outpatient': number_outpatient,
    'number_inpatient': number_inpatient,
    'number_diagnoses': number_diagnoses,
    'diabetesMed': diabetesMed,
    'GenderMale': GenderMale,
    'ERAdmission': ERAdmission,
    'discharge_disposition_id': DischargeRisk,
    'RaceCaucasian': RaceCaucasian,
    'RaceAfricanAmerican': RaceAfricanAmerican,
    'RaceOther': RaceOther,
    'diag_1_chapter': diag_1_chapter,
    'diag_2_chapter': diag_2_chapter,
    'diag_3_chapter': diag_3_chapter,
    'A1C_encoded': A1C_encoded
}])

# ============================================================
# PREDICTION
# ============================================================
st.markdown("## 💡 Generate Prediction")

if st.button("Predict Readmission Risk"):
    prob_30 = model_30.predict_proba(input_data)[0][1]

    def risk_label(p):
        if p >= 0.70:
            return "🔴 High Risk"
        elif p >= 0.40:
            return "🟠 Moderate Risk"
        else:
            return "🟢 Low Risk"

    st.markdown("### 📊 Prediction Results")
    st.info(f"**Readmission within 30 days:** {prob_30:.2f} — {risk_label(prob_30)}")

    st.markdown("---")

    colA, colB, colC = st.columns(3)

    if colA.button("High Risk Guidance"):
        st.error("""
### 🔴 High Risk
- Proactive follow‑up  
- Care coordination  
- Early outpatient check‑ins  
""")

    if colB.button("Moderate Risk Guidance"):
        st.warning("""
### 🟠 Moderate Risk
- Monitor closely  
- Review contributing factors  
""")

    if colC.button("Low Risk Guidance"):
        st.success("""
### 🟢 Low Risk
- Standard discharge planning  
""")
