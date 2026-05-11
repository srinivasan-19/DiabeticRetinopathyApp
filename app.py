import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Safe TensorFlow import
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

# ---------------- INPUTS ----------------
st.subheader("Patient Health Information")

pregnancies = st.number_input("Pregnancies", 0.0, 20.0, value=1.0)
glucose = st.number_input("Glucose", 0.0, 300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", 0.0, 200.0, value=80.0)
skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, value=20.0)
insulin = st.number_input("Insulin", 0.0, 1000.0, value=80.0)
bmi = st.number_input("BMI", 0.0, 70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, value=0.5)
age = st.number_input("Age", 1.0, 120.0, value=30.0)

# ---------------- IMAGE ----------------
uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])

image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------------- BUTTON ----------------
if st.button("Analyze Patient"):

    # -------- Stage 1 --------
    patient_data = np.array([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]])

    patient_scaled = scaler.transform(patient_data)
    risk_probability = stage1_model.predict_proba(patient_scaled)[0][1] * 100

    st.subheader("Stage 1 Result")
    st.success(f"Diabetes Risk Probability: {risk_probability:.2f}%")

    # -------- Stage 2 Trigger --------
    if risk_probability > 30:

        st.warning("High Risk Detected")

        if uploaded_file is None:
            st.error("Upload Retina Image to continue Stage 2")

        elif not TF_AVAILABLE:
            st.error("TensorFlow not installed in deployment")

        elif stage2_model is None:
            st.error("Stage 2 model not loaded")

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

                st.subheader("Stage 2 Result")
                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

                st.subheader("Class Probabilities")

                for i, cls in enumerate(class_names):
                    st.write(f"{cls}: {prediction[0][i]*100:.2f}%")

                if predicted_class.lower() == "healthy":
                    st.success("No DR detected")
                else:
                    st.error("Consult Ophthalmologist")

            except Exception as e:
                st.error(f"Stage 2 error: {str(e)}")

    else:
        st.success("Low Diabetes Risk")