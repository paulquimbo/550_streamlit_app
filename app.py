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
# ICD → CHAPTER NAME FUNCTION
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
    elif 760 <= code <= 799:
        return "Symptoms/Ill-defined"
    elif 800 <= code <= 999:
        return "Injury/Poisoning"
    else:
        return "Unknown"

# ============================================================
# ICD CHAPTER → BINARY RISK MAP
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

# ============================================================
# ADMISSION TYPE BINARY MAP
# ============================================================
admission_type_binary_map = {
    1: 1, 2: 1, 7: 1,   # Emergency/Urgent/Trauma
    3: 0, 4: 0, 5: 0, 6: 0, 8: 0
}

# ============================================================
# DISCHARGE DISPOSITION BINARY MAP
# ============================================================
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

# AGE RANGE MAP
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

    admission_type_id = st.selectbox(
        "Admission Type ID",
        list(admission_type_binary_map.keys()),
        format_func=lambda x: "Emergency/Urgent" if admission_type_binary_map[x] == 1 else "Non‑Emergency"
    )
    ERAdmission = admission_type_binary_map[admission_type_id]

    discharge_id = st.selectbox(
        "Discharge Disposition ID",
        list(discharge_disposition_binary_map.keys()),
        format_func=lambda x: "High‑Risk" if discharge_disposition_binary_map[x] == 1 else "Routine"
    )
    DischargeRisk = discharge_disposition_binary_map[discharge_id]

    Race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other"])
    RaceCaucasian = 1 if Race == "Caucasian" else 0
    RaceAfricanAmerican = 1 if Race == "AfricanAmerican" else 0
    RaceOther = 1 if Race not in ["Caucasian", "AfricanAmerican"] else 0

    PDX_diabetes_related = st.selectbox("Primary Diagnosis Diabetes-Related?", ["No", "Yes"])
    PDX_diabetes_related = 1 if PDX_diabetes_related == "Yes" else 0

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
# DIAGNOSIS CODES
# ============================================================
st.markdown("## 🩺 Diagnosis Codes")

diag_1 = st.text_input("Primary Diagnosis Code (diag_1)", "250")
diag_2 = st.text_input("Secondary Diagnosis Code (diag_2)", "401")
diag_3 = st.text_input("Additional Diagnosis Code (diag_3)", "414")

diag_1_chapter = diag_binary_map.get(icd_to_chapter(diag_1), 0)
diag_2_chapter = diag_binary_map.get(icd_to_chapter(diag_2), 0)
diag_3_chapter = diag_binary_map.get(icd_to_chapter(diag_3), 0)

# ============================================================
# MODEL INPUT
# ============================================================
input_data = pd.DataFrame({
    "age": [age],
    "time_in_hospital": [time_in_hospital],
    "num_lab_procedures": [num_lab_procedures],
    "num_procedures": [num_procedures],
    "num_medications": [num_medications],
    "number_emergency": [number_emergency],
    "number_outpatient": [number_outpatient],
    "number_inpatient": [number_inpatient],
    "number_diagnoses": [number_diagnoses],
    "diabetesMed": [diabetesMed],
    "GenderMale": [GenderMale],
    "ERAdmission": [ERAdmission],
    "discharge_disposition_id": [DischargeRisk],
    "RaceCaucasian": [RaceCaucasian],
    "RaceAfricanAmerican": [RaceAfricanAmerican],
    "RaceOther": [RaceOther],
    "PDX_diabetes_related": [PDX_diabetes_related],
    "diag_1_chapter": [diag_1_chapter],
    "diag_2_chapter": [diag_2_chapter],
    "diag_3_chapter": [diag_3_chapter]
})

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
