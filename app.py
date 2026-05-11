import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Safe TensorFlow import (prevents crash)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="AI Diabetic Retinopathy Detection",
    layout="centered"
)

@st.cache_resource
def load_models():

    stage1_model = joblib.load("stage1_model.pkl")
    scaler = joblib.load("stage1_scaler.pkl")
    class_names = joblib.load("class_names.pkl")

    stage2_model = None

    if TF_AVAILABLE:
        stage2_model = tf.keras.models.load_model("stage2_model.keras")

    return stage1_model, scaler, stage2_model, class_names


stage1_model, scaler, stage2_model, class_names = load_models()

st.title("AI Based Diabetic Retinopathy Detection")

st.subheader("Patient Health Information")

pregnancies = st.number_input("Pregnancies", 0.0, 20.0)
glucose = st.number_input("Glucose", 0.0, 300.0)
blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0)
skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0)
insulin = st.number_input("Insulin", 0.0, 1000.0)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1.0, 120.0)

uploaded_file = st.file_uploader(
    "Upload Retina Image",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Retina Image",
        use_container_width=True
    )


if st.button("Analyze Patient"):

    patient_data = np.array([[pregnancies, glucose, blood_pressure,
                               skin_thickness, insulin, bmi, dpf, age]])

    patient_scaled = scaler.transform(patient_data)

    risk_probability = stage1_model.predict_proba(patient_scaled)[0][1] * 100

    st.subheader("Stage 1 Result")
    st.success(f"Diabetes Risk Probability : {risk_probability:.2f}%")

    if risk_probability > 30:

        st.warning("High Risk Detected")

        if uploaded_file is None:
            st.error("Please Upload Retina Image")

        elif not TF_AVAILABLE:
            st.error("TensorFlow not available in deployment environment")

        else:

            image = image.resize((224, 224))
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            image_array = tf.keras.applications.densenet.preprocess_input(image_array)

            prediction = stage2_model.predict(image_array)

            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = np.max(prediction) * 100

            st.subheader("Stage 2 Result")
            st.success(f"Prediction : {predicted_class}")
            st.info(f"Confidence : {confidence:.2f}%")

            st.subheader("Class Probabilities")

            for i, class_name in enumerate(class_names):
                st.write(f"{class_name} : {prediction[0][i]*100:.2f}%")

            if predicted_class == "Healthy":
                st.success("No Major DR Symptoms Detected")
            else:
                st.error("Consult Ophthalmologist Immediately")

    else:
        st.success("Low Diabetes Risk")