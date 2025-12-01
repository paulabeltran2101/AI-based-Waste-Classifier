import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import time

# añade la carpeta raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#From project
from utils.feature_extractor import load_conv_base, extract_features_from_image
from utils.load_model import load_model
from utils.image_flow import get_frame_info, update_frame
from utils.camera_control import WebCamReader

page_bg = """
<style>

    /* ====== FONDO GENERAL ====== */
    .stApp {
        background-color: #0A1A2F !important;
        color: white !important;
    }

    /* ====== TEXTO BLANCO ====== */
    html, body, [class*="css"], p, label, span, div, h1, h2, h3, h4 {
        color: white !important;
    }

    h1 {
    font-size: 80px !important;
    font-weight: 900 !important;
    text-align: center !important;
    color: white !important;
        }

    .desc-text {
    font-size: 30px !important;
    text-align: center;
    color: white;
    opacity: 0.85;
    }

    /* ====== TÍTULOS CENTRADOS ====== */
    h1, h2, h3 {
        text-align: center !important;
        color: white !important;
    }

    /* ====== HEADER TRANSPARENTE ====== */
    .header-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background: rgba(135, 206, 250, 0.15); /* transparencia suave */
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }

    .header-title {
        font-size: 28px;
        font-weight: bold;
        color: white;
        letter-spacing: 1px;
    }

    .subheader-box {
        background: rgba(255, 255, 255, 0.20);
        padding: 10px 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-top: 25px;
        margin-bottom: 10px;
        text-align: center;
        width: fit-content; /* se ajusta al contenido */
        margin-left: auto;
        margin-right: auto;
    }
      
        /* Caja del file uploader */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.12) !important;
        padding: 15px;
        border-radius: 12px;
    }

    /* Texto dentro del uploader */
    .stFileUploader label, .stFileUploader div, .stFileUploader span {
        color: #0A1A2F !important;
    }

    /* Botón de seleccionador */
    .stFileUploader > div > button {
        background-color: rgba(255, 255, 255, 0.20) !important;
        color: #0A1A2F !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }
</style>

<!-- Barra superior -->
<div class="header-bar">
    <div class="header-title">♻️ Waste Classifier – Live AI</div>
</div>

<!-- Espacio para que el contenido no tape el header -->
<div style="margin-top: 80px;"></div>

"""

st.markdown(page_bg, unsafe_allow_html=True)


st.set_page_config(page_title="♻️Waste Classifier", layout="wide")
st.markdown("<h1>🔍 Live AI-based Waste Classifier ♻️</h1>", unsafe_allow_html=True)

# Descripción del proyecto

st.markdown("""
<div class="desc-text" style="margin-bottom:15px;">
An automatic and intelligent classification system based on Computer Vision and hybrid approach combining Machine Learning and Deep Learning.
</div>
""", unsafe_allow_html=True)
st.write("")
st.markdown("""
<div class="desc-text">
🔗 <a href="https://github.com/paulabeltran2101" target="_blank" style="color:white; text-decoration:underline;">
GitHub Repository
</a>
</div>
""", unsafe_allow_html=True)

st.write("")

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

st.markdown('<div class="card">', unsafe_allow_html=True)
mode = st.radio("👉 Select an input mode and get real-time waste classification:", ("Upload Image", "Realtime Camera"))
st.markdown('</div>', unsafe_allow_html=True)
# ======================================
# Modo 1: Subir imagen
# ======================================
if mode == "Upload Image":

    st.markdown('<div class="subheader-box"><h3>📤 Upload an Image</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Extraer features y predecir
        features = extract_features_from_image(img, conv_base)
        pred_idx = svm_model.predict(features)[0]
        pred_class = CLASS_NAMES[pred_idx]

        # Mostrar predicción arriba en texto
        st.markdown(f"""
        <div style="text-align:center; font-size:24px; font-weight:600;">
        📌 Prediction: <span style="color:#4CC9F0;">{pred_class}</span>
        </div>
        """, unsafe_allow_html=True)
        # Dibujar resultado
        cv2.putText(img, f"Waste type: {pred_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st.image(img, channels="BGR")
    
    

# ======================================
# Modo 2: Cámara en tiempo real
# ======================================
else:

    st.markdown('<div class="subheader-box"><h3>📸 Realtime Camera</h3></div>', unsafe_allow_html=True)
    # --- Cámara en directo ---
    #cam_index = st.number_input("Selecciona índice de cámara:", min_value=0, max_value=5, value=0, step=1)
    wc = WebCamReader(camera_index=0)

    # Layout: cámara a la izquierda, predicción a la derecha
    col1, col2 = st.columns([3, 1])
    frame_placeholder = col1.image([])
    pred_placeholder = col2.empty()

    run = st.checkbox("Iniciar cámara📸")

    frame_count = 0

    while run:
        frame, pred = wc.read_frame()
        if frame is None:
            st.warning("No se pudo leer frame de la cámara.")
            break

        # Convertir BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb)

        # Mostrar predicción en columna derecha
        if pred:
            pred_placeholder.markdown(f"### 📌 Predicción:\n**{pred}**")
        else:
            pred_placeholder.markdown("Esperando predicción…⏳")

        frame_count += 1
        print(f"Frame {frame_count} mostrado")
        time.sleep(0.1)  # ~30 FPS

    wc.release()

