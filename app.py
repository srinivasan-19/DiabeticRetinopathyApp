import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetic Retinopathy AI System",
    layout="centered"
)

st.title("🩺 Diabetic Retinopathy Detection System")
st.write("Enter patient details and upload retina image for analysis")

# ---------------- SAFE TENSORFLOW IMPORT ----------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    stage1_model = joblib.load("stage1_model.pkl")
    scaler = joblib.load("stage1_scaler.pkl")
    class_names = joblib.load("class_names.pkl")

    stage2_model = None
    stage2_error = None

    if TF_AVAILABLE:
        try:
            if os.path.exists("stage2_model.h5"):
                stage2_model = tf.keras.models.load_model(
                    "stage2_model.h5",
                    compile=False
                )
            elif os.path.exists("stage2_model.keras"):
                stage2_model = tf.keras.models.load_model(
                    "stage2_model.keras",
                    compile=False
                )
            else:
                stage2_error = "Stage 2 model file not found"
        except Exception as e:
            stage2_error = str(e)

    return stage1_model, scaler, stage2_model, class_names, stage2_error


stage1_model, scaler, stage2_model, class_names, stage2_error = load_models()


# ---------------- INPUT UI ----------------
st.header("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0.0, 20.0, 1.0)
    glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
    bp = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
    skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)

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

    # ---------------- STAGE 1 ----------------
    input_data = np.array([[
        pregnancies, glucose, bp,
        skin, insulin, bmi, dpf, age
    ]])

    scaled = scaler.transform(input_data)
    risk = stage1_model.predict_proba(scaled)[0][1] * 100

    st.subheader("🧠 Stage 1 Result")
    st.success(f"Diabetes Risk: {risk:.2f}%")

    # ---------------- STAGE 2 LOGIC ----------------
    if risk > 30:
        st.warning("⚠ High Risk Detected")

        if not uploaded_file:
            st.error("Upload retina image for Stage 2 analysis")

        elif not TF_AVAILABLE:
            st.error("TensorFlow not available in deployment")

        elif stage2_model is None:
            st.error("Stage 2 model not loaded ❌")
            if stage2_error:
                st.code(stage2_error)

        else:
            # IMAGE PROCESSING
            img = image.resize((224, 224))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)

            img = tf.keras.applications.densenet.preprocess_input(img)

            prediction = stage2_model.predict(img)

            idx = np.argmax(prediction)
            predicted_class = class_names[idx]
            confidence = np.max(prediction) * 100

            st.subheader("🧠 Stage 2 Result")
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.subheader("📊 Class Probabilities")
            for i, c in enumerate(class_names):
                st.write(f"{c}: {prediction[0][i]*100:.2f}%")

            if str(predicted_class).lower() == "healthy":
                st.success("No DR detected")
            else:
                st.error("Consult Ophthalmologist")

    else:
        st.success("Low Risk — No Retina Analysis Needed")


# ---------------- DEBUG PANEL ----------------
with st.expander("🔧 Debug Info"):
    st.write("TensorFlow Available:", TF_AVAILABLE)
    st.write("Stage 2 Model:", "Loaded" if stage2_model else "Not Loaded")
    if stage2_error:
        st.code(stage2_error)