
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# 📌 Load ML and Deep Learning Models
with open("best_model.pkl", "rb") as file:
    ml_model = pickle.load(file)

dl_model = tf.keras.models.load_model("best_dl_model.h5")

# 📌 Streamlit UI
st.title("💓 Heart Attack Risk Prediction")

# 📌 Updated Input Fields (Matching Dataset Features)
age = st.number_input("Enter Age", min_value=18, max_value=100, step=1)
systolic_bp = st.number_input("Systolic Blood Pressure (S_AD_KBRIG)")
diastolic_bp = st.number_input("Diastolic Blood Pressure (D_AD_KBRIG)")
diabetes = st.selectbox("Do you have Diabetes? (GB)", [0, 1])
family_history = st.selectbox("Family History of Heart Disease? (IBS_NASL)", [0, 1])
hypertension = st.selectbox("Do you have Hypertension? (SIM_GIPERT)", [0, 1])
previous_mi = st.number_input("Number of Previous Heart Attacks (INF_ANAM)", min_value=0, step=1)
heart_failure = st.selectbox("Do you have Chronic Heart Failure? (ZSN)", [0, 1])

# 📌 Ensure Input Matches Model Expectations
input_data = np.array([[age, systolic_bp, diastolic_bp, diabetes, family_history, hypertension, previous_mi, heart_failure]])

# 📌 Predict Button
if st.button("Predict Risk"):
    # Ensure input has the right shape
    import pandas as pd

    # Convert input_data to a DataFrame with correct feature names
    feature_names = ["AGE", "S_AD_KBRIG", "D_AD_KBRIG", "GB", "IBS_NASL", "SIM_GIPERT", "INF_ANAM", "ZSN"]
    input_df = pd.DataFrame(input_data, columns=feature_names)
    ml_proba = ml_model.predict_proba(input_df)[0][1]
    # Predict using ML Model
    ml_pred = ml_model.predict(input_df)[0]

    dl_proba = dl_model.predict(input_data)[0][0]  # Get probability for High Risk

    # Set new threshold to 0.7 instead of 0.5
    dl_pred = 1 if dl_proba > 0.7 else 0
    # Display Results

    # Reduce threshold from 0.5 to 0.2
    risk_threshold = 0.01  
    ml_pred = 1 if ml_proba > risk_threshold else 0

    st.write("🩺 **ML Model Prediction:**", "High Risk" if ml_pred == 1 else "Low Risk")
    st.write("📊 ML Model Confidence:", ml_proba)



    st.write("🔬 **Deep Learning Prediction:**", "High Risk" if dl_pred == 1 else "Low Risk")
    st.write("📊 DL Model Confidence:", dl_proba)
