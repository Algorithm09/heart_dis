# app.py

import streamlit as st
import pandas as pd
import joblib

# Load components
model = joblib.load("model.pkl")
dv = joblib.load("dv.pkl")
selector = joblib.load("selector.pkl")

# App title
st.title("💓 Heart Disease Prediction App")

st.markdown("""
Use the sidebar to enter patient information.  
The model will predict whether the patient is at risk for heart disease.
""")

# Sidebar input form
st.sidebar.header("🩺 Patient Data Input")

age = st.sidebar.number_input('Age', 20, 100, 50)
sex = st.sidebar.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.sidebar.selectbox('Chest Pain Type', ['typical_angina', 'atypical_angina', 'non_anginal_pain', 'asymptomatic'])
trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)
chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', 100, 600, 200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
thalach = st.sidebar.number_input('Max Heart Rate Achieved', 60, 250, 150)
exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.sidebar.number_input('ST Depression', 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox('Slope of the Peak', ['Up sloping', 'Flat', 'Down sloping'])
ca = st.sidebar.selectbox('Number of Major Vessels (0–3)', [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed defect', 'Reversible defect'])

# Assemble input dictionary
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Display input summary
st.subheader("📝 Patient Input Summary")
st.write(pd.DataFrame([input_data]))

# Predict
if st.button("🔍 Predict"):
    # Vectorize using DictVectorizer
    input_vector = dv.transform([input_data])

    # Apply feature selection
    input_selected = selector.transform(input_vector)

    # Predict with trained model
    prediction = model.predict(input_selected)[0]
    proba = model.predict_proba(input_selected)[0]

    result = "💔 There is risk of heart disease" if prediction == 1 else "💚 There is no risk of heart disease"
    st.success(f"**Prediction:** {result}")
    st.info(f"**Prediction Probabilities:**\n- No Disease: {proba[0]:.4f}\n- Disease: {proba[1]:.4f}")
