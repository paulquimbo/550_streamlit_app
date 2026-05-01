import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pickle

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="30-Day Readmission Predictor", layout="wide")

st.title("🏥 Hospital Readmission Prediction (≤30 Days)")
st.markdown("Predict the likelihood of patient readmission within 30 days using machine learning.")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("lessthan30_RandomForest_Final.Smoteenn.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'lessthan30_RandomForest_Final.Smoteenn.pkl' is in the workspace.")
        return None

# ============================================================
# HELPER FUNCTIONS (FROM NOTEBOOK)
# ============================================================

def icd_to_chapter(code):
    """Convert ICD-9 code to diagnosis chapter."""
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
    elif code >= 1000:
        return "Unknown"
    else:
        return "Unknown"

def diag_chapter_to_binary(chapter):
    """Convert diagnosis chapter to binary risk indicator."""
    risk_chapters = {
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
    }
    return risk_chapters.get(chapter, 0)

def preprocess_input(input_data):
    """Apply all preprocessing steps from the notebook."""
    df = pd.DataFrame([input_data])
    
    # ---- GENDER ENCODING (matches training data) ----
    # Training used actual gender values from the dataset
    gender_mapping = {
        "M": "Male",
        "F": "Female"
    }
    actual_gender = gender_mapping.get(df['gender'].iloc[0], df['gender'].iloc[0])
    
    # Create all possible gender columns
    for g_val in ["Female", "Male", "Unknown/Invalid"]:
        col = f"Gender_{g_val}"
        df[col] = (actual_gender == g_val).astype(int)
    
    # ---- AGE ENCODING ----
    age_map = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)
    
    # ---- ADMISSION TYPE BINARY ----
    admission_type_binary_map = {
        1: 1, 2: 1, 7: 1,  # Emergency/Urgent/Trauma
        3: 0, 4: 0, 5: 0, 6: 0, 8: 0  # Elective/Newborn/NotAvailable/NULL/NotMapped
    }
    df["admission_emergency"] = df["admission_type_id"].map(admission_type_binary_map).fillna(0).astype(int)
    
    # ---- DISCHARGE DISPOSITION RISK ----
    discharge_disposition_binary_map = {
        2: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 19: 0, 20: 0, 21: 0,  # Lower-risk
        1: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 16: 1,
        17: 1, 18: 1, 22: 1, 23: 1, 24: 1, 25: 1, 27: 1, 28: 1, 29: 1, 30: 1  # Higher-risk
    }
    df["DischargeRisk"] = df["discharge_disposition_id"].map(discharge_disposition_binary_map).fillna(0).astype(int)
    
    # ---- DIAGNOSIS CHAPTERS ----
    for col in ["diag_1", "diag_2", "diag_3"]:
        chapter_col = f"{col}_chapter"
        df[chapter_col] = df[col].apply(icd_to_chapter)
        df[chapter_col] = df[chapter_col].apply(diag_chapter_to_binary)
    
    # ---- DIABETIC MEDICATION ----
    df['diabetesMed'] = (df['diabetesMed'].str.lower() == "yes").astype(int)
    
    # ---- RACE ENCODING (matches training data - no spaces) ----
    race_mapping = {
        "Caucasian": "Caucasian",
        "African American": "AfricanAmerican",
        "Hispanic": "Hispanic",
        "Asian": "Asian",
        "Other": "Other",
        "Unknown": "Unknown"
    }
    actual_race = race_mapping.get(df['race'].iloc[0], df['race'].iloc[0])
    
    # Create all possible race columns as they were in training
    for race_val in ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]:
        col_name = f"Race_{race_val}"
        df[col_name] = (actual_race == race_val).astype(int)
    
    return df


def get_model_features():
    """Get the exact feature names and order expected by the model."""
    return [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_outpatient',
        'number_inpatient', 'number_diagnoses', 'age',
        'admission_emergency', 'DischargeRisk',
        'diag_1_chapter', 'diag_2_chapter', 'diag_3_chapter',
        'diabetesMed', 'Gender_Female', 'Gender_Male', 'Gender_Unknown/Invalid',
        'Race_Caucasian', 'Race_AfricanAmerican', 'Race_Hispanic', 'Race_Asian', 'Race_Other', 'Race_Unknown'
    ]

# ============================================================
# STREAMLIT APP
# ============================================================

model = load_model()

if model is not None:
    st.sidebar.header("📋 Patient Information")
    
    # Create input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age_range = st.selectbox(
            "Age Range",
            ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
            help="Select patient age range"
        )
    
    with col2:
        gender = st.selectbox("Gender", ["M", "F"])
    
    time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 5)
    num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 0, 100, 40)
    num_procedures = st.sidebar.slider("Number of Procedures", 0, 10, 1)
    num_medications = st.sidebar.slider("Number of Medications", 0, 81, 16)
    number_emergency = st.sidebar.slider("Number of Emergency Visits", 0, 50, 0)
    number_outpatient = st.sidebar.slider("Number of Outpatient Visits", 0, 50, 0)
    number_inpatient = st.sidebar.slider("Number of Inpatient Visits", 0, 50, 0)
    number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 9)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🏥 Clinical Details")
    
    admission_type = st.sidebar.selectbox(
        "Admission Type",
        {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 
         5: "NotAvailable", 6: "NULL", 7: "TraumaCenter", 8: "NotMapped"},
        format_func=lambda x: {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
                              5: "NotAvailable", 6: "NULL", 7: "TraumaCenter", 8: "NotMapped"}[x]
    )
    
    discharge_disposition = st.sidebar.selectbox(
        "Discharge Disposition",
        {1: "Home", 2: "ShortTermHospital", 3: "SNF", 4: "ICF", 6: "HomeHealth", 
         11: "Expired", 13: "HospiceHome", 14: "HospiceFacility"},
        format_func=lambda x: {1: "Home", 2: "ShortTermHospital", 3: "SNF", 4: "ICF", 
                              6: "HomeHealth", 11: "Expired", 13: "HospiceHome", 14: "HospiceFacility"}[x]
    )
    
    race = st.sidebar.selectbox(
        "Race",
        ["Caucasian", "African American", "Hispanic", "Asian", "Other", "Unknown"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Diagnosis Codes (ICD-9)")
    
    diag_1 = st.sidebar.number_input("Primary Diagnosis Code", min_value=1, max_value=999, value=250, step=1)
    diag_2 = st.sidebar.number_input("Secondary Diagnosis Code", min_value=1, max_value=999, value=401, step=1)
    diag_3 = st.sidebar.number_input("Tertiary Diagnosis Code", min_value=1, max_value=999, value=414, step=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("💊 Medication")
    
    diabetesMed = st.sidebar.selectbox("Diabetic Medication Prescribed", ["Yes", "No"])
    
    # ============================================================
    # PREPARE INPUT
    # ============================================================
    input_data = {
        'age': age_range,
        'gender': gender,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_emergency': number_emergency,
        'number_outpatient': number_outpatient,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'admission_type_id': admission_type,
        'discharge_disposition_id': discharge_disposition,
        'race': race,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3,
        'diabetesMed': diabetesMed,
    }
    
    # ============================================================
    # MAIN CONTENT
    # ============================================================
    
    if st.sidebar.button("🔮 Predict Readmission Risk", use_container_width=True):
        try:
            # Preprocess
            processed_df = preprocess_input(input_data)
            
            # Get feature names that the model expects (in correct order)
            feature_cols = get_model_features()
            
            # Ensure all required features exist, fill missing with 0
            for col in feature_cols:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            # Select only the features the model expects (in exact order)
            X_input = processed_df[feature_cols].copy()
            
            # Make prediction
            prediction_proba = model.predict_proba(X_input)[0]
            prediction = model.predict(X_input)[0]
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Prediction",
                    "⚠️ HIGH RISK" if prediction == 1 else "✅ LOW RISK",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Readmission Probability",
                    f"{prediction_proba[1]:.1%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "No Readmission Probability",
                    f"{prediction_proba[0]:.1%}",
                    delta=None
                )
            
            # Detailed visualization
            st.markdown("---")
            st.subheader("📈 Risk Distribution")
            
            risk_data = {
                "Risk Class": ["No Readmission (≤30 days)", "Readmission (≤30 days)"],
                "Probability": [prediction_proba[0] * 100, prediction_proba[1] * 100]
            }
            risk_df = pd.DataFrame(risk_data)
            
            st.bar_chart(risk_df.set_index("Risk Class"))
            
            # Input summary
            st.markdown("---")
            st.subheader("👤 Input Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Demographics**")
                st.write(f"- Age: {age_range}")
                st.write(f"- Gender: {gender}")
                st.write(f"- Race: {race}")
            
            with summary_col2:
                st.write("**Clinical Metrics**")
                st.write(f"- Time in Hospital: {time_in_hospital} days")
                st.write(f"- Number of Medications: {num_medications}")
                st.write(f"- Number of Diagnoses: {number_diagnoses}")
            
            st.write("**Clinical Codes**")
            st.write(f"- Primary Diagnosis: {diag_1} ({icd_to_chapter(diag_1)})")
            st.write(f"- Secondary Diagnosis: {diag_2} ({icd_to_chapter(diag_2)})")
            st.write(f"- Tertiary Diagnosis: {diag_3} ({icd_to_chapter(diag_3)})")
            
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {str(e)}")
            st.warning("Check that all required input fields are correctly filled.")
