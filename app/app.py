import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os

# añade la carpeta raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extractor import load_conv_base, extract_features_from_image
from utils.load_model import load_model

st.set_page_config(page_title="♻️Waste Classifier", layout="wide")

st.title("🔍 Live AI-based Waste Classifier ♻️")
st.write("Selecciona el modo de entrada y obtén la predicción en tiempo real.")

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    conv = load_conv_base()
    svm = load_model()
    return conv, svm

conv_base, svm_model = load_models()

CLASS_NAMES = [
    "Cardboard", "Food Organics", "Glass", "Metal", "Misc Trash",
    "Paper", "Plastic", "Textile Trash", "Vegetation"
]

# ======================================
# Selección de modo
# ======================================
mode = st.radio("Select mode:", ("Upload Image", "Realtime Camera"))

# ======================================
# Modo 1: Subir imagen
# ======================================
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Extraer features y predecir
        features = extract_features_from_image(img, conv_base)
        pred_idx = svm_model.predict(features)[0]
        pred_class = CLASS_NAMES[pred_idx]

        # Dibujar resultado
        cv2.putText(img, f"Waste type: {pred_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(img, channels="BGR", caption=f"Predicted class: {pred_class}")

# ======================================
# Modo 2: Cámara en tiempo real
# ======================================
else:
    stframe = st.empty()

    PROCESS_EVERY_N_FRAMES = 5
    frame_count = 0

    st.warning("Click the **Stop** button to close the camera.")

    run_camera = st.checkbox("Start camera")
    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Unable to access camera.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera disconnected.")
                    break

                frame_count += 1

                # Procesar solo cada N frames
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    features = extract_features_from_image(frame, conv_base)
                    pred_idx = svm_model.predict(features)[0]
                    pred_class = CLASS_NAMES[pred_idx]

                    # Dibujar sobre la imagen
                    cv2.putText(frame, f"Predicted: {pred_class}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar imagen en Streamlit
                stframe.image(frame, channels="BGR")
                
                # Stop button logic
                if not st.checkbox("Camera running", value=True, key="run_cam"):
                    break

        cap.release()
        cv2.destroyAllWindows()