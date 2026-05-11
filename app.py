import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

st.set_page_config(page_title="DR Detection", layout="centered")

st.title("🩺 Diabetic Retinopathy System")

# ---------------- LOAD SAFE MODELS ----------------
@st.cache_resource
def load_models():

    stage1 = joblib.load("stage1_model.pkl")
    scaler = joblib.load("stage1_scaler.pkl")
    class_names = joblib.load("class_names.pkl")

    stage2 = None

    # ONLY load if valid file exists
    if os.path.exists("stage2_model.h5"):
        try:
            import tensorflow as tf
            stage2 = tf.keras.models.load_model("stage2_model.h5", compile=False)
        except Exception as e:
            st.error(f"Stage 2 load error: {e}")

    return stage1, scaler, stage2, class_names


stage1_model, scaler, stage2_model, class_names = load_models()

# ---------------- INPUT ----------------
preg = st.number_input("Pregnancies", 0.0, 20.0, 1.0)
glu = st.number_input("Glucose", 0.0, 300.0, 120.0)
bp = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("DPF", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1.0, 120.0, 30.0)

file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

img = None
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

# ---------------- RUN ----------------
if st.button("Analyze"):

    X = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
    X = scaler.transform(X)

    risk = stage1_model.predict_proba(X)[0][1] * 100

    st.subheader("Stage 1 Result")
    st.success(f"Risk: {risk:.2f}%")

    # ---------------- STAGE 2 ----------------
    if risk > 30:

        st.warning("High Risk Detected")

        if stage2_model is None:
            st.error("Stage 2 model NOT loaded — fix .h5 file")

        elif img is None:
            st.error("Upload retina image")

        else:
            import tensorflow as tf

            img2 = img.resize((224, 224))
            img2 = np.array(img2)
            img2 = np.expand_dims(img2, axis=0)

            img2 = tf.keras.applications.densenet.preprocess_input(img2)

            pred = stage2_model.predict(img2)

            idx = np.argmax(pred)
            label = class_names[idx]
            conf = np.max(pred) * 100

            st.subheader("Stage 2 Result")
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.2f}%")

    else:
        st.success("Low Risk")