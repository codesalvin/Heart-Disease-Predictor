import streamlit as st
import numpy as np
import joblib

knn = joblib.load('models/knn_model.pkl')
svm = joblib.load('models/svm_model.pkl')
ann = joblib.load('models/ann_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("❤️ Heart Disease Prediction App")

model_choice = st.selectbox("Choose Model", ["KNN", "SVM", "ANN"])

st.write("Enter patient details:")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    if model_choice == "KNN":
        model = knn
    elif model_choice == "SVM":
        model = svm
    else:
        model = ann

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({model_choice})")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({model_choice})")