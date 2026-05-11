import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ---------------- SAFE TF IMPORT ----------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetic Retinopathy AI System",
    layout="centered"
)

st.title("🩺 AI Diabetic Retinopathy Detection System")
st.write("Enter patient details and upload retina image for prediction.")


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    stage1_model = joblib.load("stage1_model.pkl")
    scaler = joblib.load("stage1_scaler.pkl")
    class_names = joblib.load("class_names.pkl")

    stage2_model = None
    error_msg = None

    if TF_AVAILABLE:
        try:
            stage2_model = tf.keras.models.load_model("stage2_model.keras")
        except Exception as e:
            error_msg = str(e)

    return stage1_model, scaler, stage2_model, class_names, error_msg


stage1_model, scaler, stage2_model, class_names, stage2_error = load_models()


# ---------------- INPUT SECTION ----------------
st.header("📋 Patient Data")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0.0, 20.0, 1.0)
    glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
    blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
    skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)

with col2:
    insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1.0, 120.0, 30.0)


# ---------------- IMAGE UPLOAD ----------------
st.header("📷 Retina Image Upload")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Patient"):

    # -------- Stage 1 --------
    patient_data = np.array([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]])

    patient_scaled = scaler.transform(patient_data)
    risk = stage1_model.predict_proba(patient_scaled)[0][1] * 100

    st.subheader("🧠 Stage 1 Result (Risk Prediction)")
    st.success(f"Diabetes Risk: {risk:.2f}%")

    # -------- Stage 2 Decision --------
    if risk > 30:
        st.warning("⚠ High Risk Detected - Running Retina Analysis")

        if not uploaded_file:
            st.error("Please upload retina image for Stage 2 analysis")

        elif not TF_AVAILABLE:
            st.error("TensorFlow not installed in environment")

        elif stage2_model is None:
            st.error("Stage 2 model failed to load ❌")
            if stage2_error:
                st.code(stage2_error)

        else:
            # -------- IMAGE PROCESS --------
            img = image.resize((224, 224))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)

            img = tf.keras.applications.densenet.preprocess_input(img)

            prediction = stage2_model.predict(img)

            idx = np.argmax(prediction)
            predicted_class = class_names[idx]
            confidence = np.max(prediction) * 100

            st.subheader("🧠 Stage 2 Result (Retina Analysis)")
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.subheader("📊 Class Probabilities")
            for i, c in enumerate(class_names):
                st.write(f"{c}: {prediction[0][i]*100:.2f}%")

            if predicted_class.lower() == "healthy":
                st.success("No DR detected")
            else:
                st.error("Consult Ophthalmologist")

    else:
        st.success("Low Risk — No Retina Analysis Needed")