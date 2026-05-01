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
# ADMISSION TYPE NAMES
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

# ============================================================
# DISCHARGE DISPOSITION NAMES
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

# ============================================================
# ADMISSION SOURCE NAMES
# ============================================================
admission_source_names = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from Hospital",
    5: "Transfer from Skilled Nursing",
    6: "Transfer from Other Health Care Facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from Critical Access Hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Mapped",
    17: "NULL"
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
st.markdown("## 🧍 Patient Visit Details")

col1, col2 = st.columns(2)

with col1:
    admission_choice = st.selectbox("Admission Type", [""] + list(admission_type_names.values()))
    admission_type_id = None
    if admission_choice:
        admission_type_id = [k for k, v in admission_type_names.items() if v == admission_choice][0]

    discharge_choice = st.selectbox("Discharge Disposition", [""] + list(discharge_names.values()))
    discharge_disposition_id = None
    if discharge_choice:
        discharge_disposition_id = [k for k, v in discharge_names.items() if v == discharge_choice][0]

    admission_source_choice = st.selectbox("Admission Source", [""] + list(admission_source_names.values()))
    admission_source_id = None
    if admission_source_choice:
        admission_source_id = [k for k, v in admission_source_names.items() if v == admission_source_choice][0]

with col2:
    diabetesMed = st.selectbox("On Diabetes Medication?", ["", "No", "Yes"])
    if diabetesMed == "Yes":
        diabetesMed = 1
    elif diabetesMed == "No":
        diabetesMed = 0
    else:
        diabetesMed = None

    gender = st.selectbox("Gender", ["", "Female", "Male", "Other"])
    GenderMale = None
    if gender == "Male":
        GenderMale = 1
    elif gender in ["Female", "Other"]:
        GenderMale = 0

    Race = st.selectbox("Race", ["", "Caucasian", "AfricanAmerican", "Other"])
    RaceCaucasian = RaceAfricanAmerican = RaceOther = None
    if Race:
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
    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=0, step=1, value=None, placeholder="Enter days")
    num_lab_procedures = st.number_input("Lab Procedures", min_value=0, step=1, value=None, placeholder="Enter count")
    num_procedures = st.number_input("Procedures", min_value=0, step=1, value=None, placeholder="Enter count")

with col4:
    num_medications = st.number_input("Medications", min_value=0, step=1, value=None, placeholder="Enter count")
    number_outpatient = st.number_input("Outpatient Visits", min_value=0, step=1, value=None, placeholder="Enter count")
    number_emergency = st.number_input("Emergency Visits", min_value=0, step=1, value=None, placeholder="Enter count")
    number_inpatient = st.number_input("Inpatient Visits", min_value=0, step=1, value=None, placeholder="Enter count")

number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, step=1, value=None, placeholder="Enter count")

st.markdown("---")

# ============================================================
# DIAGNOSIS CATEGORIES
# ============================================================
st.markdown("## 🩺 Diagnosis Categories")

diag_1_name = st.selectbox("Primary Diagnosis Category", [""] + diagnosis_options)
diag_2_name = st.selectbox("Secondary Diagnosis Category", [""] + diagnosis_options)
diag_3_name = st.selectbox("Additional Diagnosis Category", [""] + diagnosis_options)

diag_1_chapter = diag_binary_map[diag_1_name] if diag_1_name else None
diag_2_chapter = diag_binary_map[diag_2_name] if diag_2_name else None
diag_3_chapter = diag_binary_map[diag_3_name] if diag_3_name else None

# ============================================================
# VALIDATION
# ============================================================
all_fields = [
    admission_type_id, discharge_disposition_id, admission_source_id,
    time_in_hospital, num_lab_procedures, num_procedures, num_medications,
    number_outpatient, number_emergency, number_inpatient, number_diagnoses,
    diabetesMed, GenderMale, diag_1_chapter, diag_2_chapter, diag_3_chapter,
    RaceCaucasian, RaceAfricanAmerican, RaceOther
]

if any(v is None for v in all_fields):
    st.warning("Please complete all fields before predicting.")
else:
    input_data = pd.DataFrame([{
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'diabetesMed': diabetesMed,
        'GenderMale': GenderMale,
        'diag_1_chapter': diag_1_chapter,
        'diag_2_chapter': diag_2_chapter,
        'diag_3_chapter': diag_3_chapter,
        'RaceCaucasian': RaceCaucasian,
        'RaceAfricanAmerican': RaceAfricanAmerican,
        'RaceOther': RaceOther
    }])

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

        st.info(f"**Readmission within 30 days:** {prob_30:.2f} — {risk_label(prob_30)}")
