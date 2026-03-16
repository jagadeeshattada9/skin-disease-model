import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
from PIL import Image
from keras.models import load_model
model = load_model("skin_disease_model.h5", compile=False)

# -------------------------
# Page Setup
# -------------------------

st.set_page_config(page_title="Skin Disease Detection", layout="centered")

st.title("🧴 AI Skin Disease Detection")
st.write("Upload a skin image and enter patient details")

# -------------------------
# Load Model and Encoders
# -------------------------

@st.cache_resource
def load_resources():

    model = load_model("skin_disease_model.h5", compile=False)

    label_encoder = joblib.load("label_encoder.pkl")
    sex_encoder = joblib.load("sex_encoder.pkl")
    loc_encoder = joblib.load("loc_encoder.pkl")
    dx_type_encoder = joblib.load("dx_type_encoder.pkl")
    scaler = joblib.load("age_scaler.pkl")

    return model, label_encoder, sex_encoder, loc_encoder, dx_type_encoder, scaler


model, label_encoder, sex_encoder, loc_encoder, dx_type_encoder, scaler = load_resources()

# -------------------------
# Disease Full Names
# -------------------------

disease_names = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

# -------------------------
# User Inputs
# -------------------------

age = st.number_input("Age", min_value=1, max_value=100, value=25)

sex = st.selectbox(
    "Sex",
    ["male", "female", "unknown"]
)

localization = st.selectbox(
    "Body Location",
    ["back","face","chest","arm","leg","abdomen","scalp","neck","hand","foot","unknown"]
)

dx_type = st.selectbox(
    "Diagnosis Type",
    ["histo","consensus","follow_up","confocal"]
)

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# Prediction
# -------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width="stretch")

    # Image preprocessing
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Metadata preprocessing
    sex_val = sex_encoder.transform([sex])
    loc_val = loc_encoder.transform([localization])
    dx_val = dx_type_encoder.transform([dx_type])

    age_val = scaler.transform([[age]])[0][0]

    meta = np.array([[age_val, sex_val[0], loc_val[0], dx_val[0]]])

    # Model prediction
    prediction = model.predict([img, meta])

    class_index = np.argmax(prediction)

    disease_code = label_encoder.inverse_transform([class_index])[0]

    disease = disease_names.get(disease_code, disease_code)

    confidence = np.max(prediction)

    # -------------------------
    # Display Result
    # -------------------------

    st.subheader("Prediction Result")

    st.success(f"Predicted Disease: {disease}")

    st.info(f"Confidence: {confidence:.2%}")

    # Show probabilities
    st.subheader("All Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        code = label_encoder.inverse_transform([i])[0]
        name = disease_names.get(code, code)
        st.write(f"{name}: {prob:.2%}")