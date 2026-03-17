import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    layout="centered",
    page_icon="🧴"
)

# -------------------------
# 🌈 UI Styling
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #dbeafe, #f0f9ff);
}
.big-title {
    font-size: 36px;
    font-weight: bold;
    color: #1E3A8A;
    text-align: center;
}
.sub-text {
    text-align: center;
    color: #374151;
}
.result-box {
    padding: 25px;
    border-radius: 20px;
    background-color: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown('<div class="big-title">🧴 AI Skin Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a close-up skin lesion image (not a selfie)</div>', unsafe_allow_html=True)
st.write("---")

# -------------------------
# Load Resources
# -------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("skin_disease_model.h5", compile=False)
    label_encoder = joblib.load("label_encoder.pkl")
    sex_encoder = joblib.load("sex_encoder.pkl")
    loc_encoder = joblib.load("loc_encoder.pkl")
    dx_type_encoder = joblib.load("dx_type_encoder.pkl")
    scaler = joblib.load("age_scaler.pkl")
    return model, label_encoder, sex_encoder, loc_encoder, dx_type_encoder, scaler


model, label_encoder, sex_encoder, loc_encoder, dx_type_encoder, scaler = load_resources()

# -------------------------
# Disease Names
# -------------------------
disease_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

# -------------------------
# 🔐 Safe Encoder Function
# -------------------------
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0  # fallback value

# -------------------------
# Validation Functions
# -------------------------
def is_skin_image(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio > 0.15


def contains_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0

# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 25)
    sex = st.selectbox("Sex", ["male", "female", "unknown"])

with col2:
    localization = st.selectbox(
        "Body Location",
        ["back","face","chest","arm","leg","abdomen","scalp","neck","hand","foot","unknown"]
    )
    dx_type = st.selectbox(
        "Diagnosis Type",
        ["histo","consensus","follow_up","confocal","unknown"]  # keep unknown
    )

uploaded_file = st.file_uploader("📷 Upload Skin Image", type=["jpg", "jpeg", "png"])

# -------------------------
# Prediction Flow
# -------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ❌ Reject selfie
    if contains_face(image):
        st.error("⚠️ Face detected. Please upload a close-up skin lesion image.")
        st.stop()

    # ❌ Reject non-skin
    if not is_skin_image(image):
        st.error("⚠️ This is not a valid skin image.")
        st.stop()

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # -------------------------
    # SAFE Metadata Encoding
    # -------------------------
    sex_val = safe_transform(sex_encoder, sex)
    loc_val = safe_transform(loc_encoder, localization)
    dx_val = safe_transform(dx_type_encoder, dx_type)

    age_val = scaler.transform([[age]])[0][0]

    meta = np.array([[age_val, sex_val, loc_val, dx_val]])

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict([img, meta])

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    disease_code = label_encoder.inverse_transform([class_index])[0]
    disease = disease_names.get(disease_code, disease_code)

    # -------------------------
    # Result Logic
    # -------------------------
    st.write("---")
    st.subheader("🧾 Result")

    if confidence < 0.45:
        st.success("✅ No visible skin disease detected (Normal Skin)")

    elif 0.45 <= confidence < 0.70:
        st.warning("⚠️ Possible skin condition detected")
        st.info(f"Prediction: {disease} ({confidence:.2%})")

    else:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"🦠 Predicted Disease: {disease}")
        st.progress(confidence)
        st.info(f"Confidence: {confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # Probabilities
    # -------------------------
    st.subheader("📊 All Predictions")

    for i, prob in enumerate(prediction[0]):
        code = label_encoder.inverse_transform([i])[0]
        name = disease_names.get(code, code)
        st.write(f"{name}: {prob:.2%}")