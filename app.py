import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Load Model
# ============================================================
MODEL_PATH = "lessthan30_SGDClassifier_Final.Smoteenn.A1C.pkl"
model = joblib.load(MODEL_PATH)

# ============================================================
# ICD → Chapter Mapping
# ============================================================
def icd_to_chapter(code):
    try:
        code = float(code)
    except:
        return "Unknown"

    if 1 <= code <= 139:
        return "Infectious and Parasitic Diseases"
    elif 140 <= code <= 239:
        return "Neoplasms"
    elif 240 <= code <= 279:
        return "Endocrine/Metabolic"
    elif 280 <= code <= 289:
        return "Blood Diseases"
    elif 290 <= code <= 319:
        return "Mental Disorders"
    elif 320 <= code <= 389:
        return "Nervous System"
    elif 390 <= code <= 459:
        return "Circulatory System"
    elif 460 <= code <= 519:
        return "Respiratory System"
    elif 520 <= code <= 579:
        return "Digestive System"
    elif 580 <= code <= 629:
        return "Genitourinary System"
    elif 630 <= code <= 679:
        return "Pregnancy/Childbirth"
    elif 680 <= code <= 709:
        return "Skin/Subcutaneous"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 740 <= code <= 759:
        return "Congenital Anomalies"
    elif 760 <= code <= 779:
        return "Perinatal Conditions"
    elif 780 <= code <= 799:
        return "Symptoms/Ill-defined"
    elif 800 <= code <= 999:
        return "Injury/Poisoning"
    else:
        return "Unknown"

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

# ============================================================
# Admission / Discharge Mappings
# ============================================================
admission_type_binary_map = {
    1: 1, 2: 1, 7: 1,
    3: 0, 4: 0, 5: 0, 6: 0, 8: 0
}

discharge_disposition_binary_map = {
    1: 0, 6: 0, 8: 0, 18: 0, 25: 0,
    2: 1, 3: 1, 4: 1, 5: 1, 7: 1, 9: 1, 10: 1,
    11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1,
    19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 27: 1,
    28: 1, 29: 1, 30: 1
}

# ============================================================
# Feature Order (from model.feature_names_in_)
# ============================================================
FEATURES = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'diabetesMed', 'GenderMale',
    'diag_1_chapter', 'diag_2_chapter', 'diag_3_chapter',
    'RaceCaucasian', 'RaceAfricanAmerican', 'RaceOther'
]

# ============================================================
# Streamlit UI
# ============================================================
st.title("30-Day Readmission Prediction")
st.write("Model: **SGDClassifier (SMOTEENN)** — Target: **lessthan30**")

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other"])
    diabetes_med = st.selectbox("On Diabetes Medication?", ["Yes", "No"])
    admission_type_raw = st.selectbox("Admission Type ID", [1,2,3,4,5,6,7,8])
    discharge_disposition_raw = st.selectbox(
        "Discharge Disposition ID",
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
         18,19,20,21,22,23,24,25,27,28,29,30]
    )
    admission_source_id = st.number_input("Admission Source ID", min_value=1, step=1, value=1)

with col2:
    time_in_hospital = st.number_input("Time in Hospital", min_value=1, step=1, value=3)
    num_lab_procedures = st.number_input("Lab Procedures", min_value=0, step=1, value=40)
    num_procedures = st.number_input("Procedures", min_value=0, step=1, value=0)
    num_medications = st.number_input("Medications", min_value=0, step=1, value=10)
    number_outpatient = st.number_input("Outpatient Visits", min_value=0, step=1, value=0)
    number_emergency = st.number_input("Emergency Visits", min_value=0, step=1, value=0)
    number_inpatient = st.number_input("Inpatient Visits", min_value=0, step=1, value=0)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, step=1, value=5)

st.header("Diagnosis Codes (ICD-9)")
diag_1 = st.text_input("Primary Diagnosis (diag_1)", value="250")
diag_2 = st.text_input("Secondary Diagnosis (diag_2)", value="401")
diag_3 = st.text_input("Tertiary Diagnosis (diag_3)", value="414")

# ============================================================
# Prediction
# ============================================================
if st.button("Predict Readmission Risk"):

    GenderMale = 1 if gender == "Male" else 0

    RaceCaucasian = 1 if race == "Caucasian" else 0
    RaceAfricanAmerican = 1 if race == "AfricanAmerican" else 0
    RaceOther = 1 if race not in ["Caucasian", "AfricanAmerican"] else 0

    diabetesMed_enc = 0 if diabetes_med == "No" else 1

    admission_type_id_enc = int(admission_type_binary_map.get(admission_type_raw, 0))
    discharge_disposition_id_enc = int(discharge_disposition_binary_map.get(discharge_disposition_raw, 0))

    diag_1_chapter = diag_binary_map.get(icd_to_chapter(diag_1), 0)
    diag_2_chapter = diag_binary_map.get(icd_to_chapter(diag_2), 0)
    diag_3_chapter = diag_binary_map.get(icd_to_chapter(diag_3), 0)

    X_input = pd.DataFrame([{
        "admission_type_id": admission_type_id_enc,
        "discharge_disposition_id": discharge_disposition_id_enc,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "diabetesMed": diabetesMed_enc,
        "GenderMale": GenderMale,
        "diag_1_chapter": diag_1_chapter,
        "diag_2_chapter": diag_2_chapter,
        "diag_3_chapter": diag_3_chapter,
        "RaceCaucasian": RaceCaucasian,
        "RaceAfricanAmerican": RaceAfricanAmerican,
        "RaceOther": RaceOther
    }])

    # Force correct order
    X_input = X_input[FEATURES]

    # Predict
    try:
        proba = model.predict_proba(X_input)[0, 1]
    except:
        score = model.decision_function(X_input)[0]
        proba = 1 / (1 + np.exp(-score))

    pred_class = int(proba >= 0.5)

    st.subheader("Prediction Result")
    st.write(f"**Probability of readmission < 30 days:** {proba:.3f}")
    st.write(f"**Predicted class:** {pred_class}")

    if pred_class == 1:
        st.warning("High risk of readmission within 30 days.")
    else:
        st.success("Lower risk of readmission within 30 days.")
