import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import time

# añade la carpeta raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extractor import load_conv_base, extract_features_from_image
from utils.load_model import load_model

st.set_page_config(page_title="♻️Waste Classifier", layout="wide")

st.title("🔍 Live AI-based Waste Classifier ♻️")
st.write("Select an input mode and get real-time waste classification.")

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
    # ------------------------------
    # Session state initialization
    # ------------------------------
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if "cap" not in st.session_state:
        st.session_state.cap = None

    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0

    PROCESS_EVERY_N_FRAMES = 5
    
    # ------------------------------
    # Camera discovery
    # ------------------------------
    st.subheader("📷 Realtime Camera Prediction")

    available_cameras = []
    for i in range(10):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cam is not None and cam.isOpened():
            ret, _ = cam.read()
            if ret:
                available_cameras.append(i)
        cam.release()
    
    if not available_cameras:
        st.error("❌ No camera detected")
        st.stop()

    cam_index = st.selectbox("Select camera device:", available_cameras)

    col1, col2 = st.columns(2)
    start_button = col1.button("▶️ Start camera")
    stop_button = col2.button("⏹️ Stop camera")
    
    # ------------------------------
    # START CAMERA
    # ------------------------------
    if start_button:
        st.session_state.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not st.session_state.cap.isOpened():
            st.error("❌ Could not open camera.")
        else:
            st.session_state.camera_running = True

    # ------------------------------
    # STOP CAMERA
    # ------------------------------
    if stop_button:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        st.session_state.camera_running = False
        st.session_state.cap = None

    frame_area = st.empty()

    # ------------------------------
    # CAMERA LOOP (SAFE)
    # ------------------------------
    if st.session_state.camera_running and st.session_state.cap is not None:

        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("❌ Cannot read from camera.")
            st.session_state.camera_running = False
        else:
            st.session_state.frame_counter += 1

            # Process every N frames
            if st.session_state.frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                resized = cv2.resize(frame, (380, 380))
                features = extract_features_from_image(resized, conv_base)
                pred_idx = svm_model.predict(features)[0]
                pred_class = CLASS_NAMES[pred_idx]

                cv2.putText(frame, f"Predicted: {pred_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame
            frame_area.image(frame, channels="BGR")

        # Small delay to improve smoothness
        time.sleep(0.03)


   
