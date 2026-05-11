import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetic Retinopathy AI System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Diabetic Retinopathy Detection System")
st.markdown("### Stage 1 (Risk Prediction) + Stage 2 (Retina Analysis)")

# ---------------- SAFE IMPORT ----------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    stage1_model = joblib.load("stage1_model.pkl")
    scaler = joblib.load("stage1_scaler.pkl")
    class_names = joblib.load("class_names.pkl")

    stage2_model = None
    if TF_AVAILABLE:
        try:
            stage2_model = tf.keras.models.load_model(
                "stage2_model.keras",
                compile=False
            )
        except:
            stage2_model = None

    return stage1_model, scaler, stage2_model, class_names


stage1_model, scaler, stage2_model, class_names = load_models()


# ---------------- INPUT SECTION ----------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0.0, 20.0, 1.0)
    glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
    blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, 80.0)
    skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)

with col2:
    insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1.0, 120.0, 30.0)


# ---------------- IMAGE UPLOAD ----------------
st.subheader("📷 Retina Image Upload")

uploaded_file = st.file_uploader(
    "Upload retina image",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


# ---------------- ANALYZE BUTTON ----------------
if st.button("🔍 Analyze Patient"):

    # ---------------- STAGE 1 ----------------
    input_data = np.array([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]])

    scaled = scaler.transform(input_data)
    risk = stage1_model.predict_proba(scaled)[0][1] * 100

    st.markdown("## 🩺 Stage 1 Result")
    st.success(f"Diabetes Risk: {risk:.2f}%")

    # ---------------- STAGE 2 ----------------
    if risk > 30:
        st.warning("⚠ High Risk Detected → Proceeding to Stage 2")

        if not uploaded_file:
            st.error("Please upload retina image for Stage 2 analysis")

        elif not TF_AVAILABLE:
            st.error("TensorFlow not available in this environment")

        elif stage2_model is None:
            st.error("Stage 2 model could not be loaded")

        else:
            try:
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)

                img_array = tf.keras.applications.densenet.preprocess_input(img_array)

                prediction = stage2_model.predict(img_array)

                predicted_index = np.argmax(prediction)
                predicted_class = class_names[predicted_index]
                confidence = np.max(prediction) * 100

                st.markdown("## 🧬 Stage 2 Result")

                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

                st.markdown("### 📊 Class Probabilities")

                for i, cls in enumerate(class_names):
                    st.write(f"{cls}: {prediction[0][i]*100:.2f}%")

                if predicted_class.lower() == "healthy":
                    st.success("✔ No Diabetic Retinopathy detected")
                else:
                    st.error("⚠ Consult Ophthalmologist Immediately")

            except Exception as e:
                st.error(f"Stage 2 error: {str(e)}")

    else:
        st.success("✔ Low Risk — No further analysis required")