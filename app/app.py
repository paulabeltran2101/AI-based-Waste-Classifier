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
    font-size: 65px !important;
    font-weight: 900 !important;
    text-align: center !important;
    color: white !important;
        }

    .desc-text {
    font-size: 22px !important;
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
        background: rgba(200, 230, 255, 0.40); /* transparencia suave */
        backdrop-filter: none;
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
    background: transparent;
    color: #4CC9F0;
    padding: 10px 20px;
    border-top: 2px solid #4CC9F0;  
    border-bottom: 2px solid #4CC9F0;
    text-align: center;
    margin: 20px auto;
    font-weight: bold;
    font-size: 20px;
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

st.markdown("""
<div style="
    background-color: rgba(135, 206, 250, 0.5);  /* azul clarito con transparencia */
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    text-align: center;
">
    <h1 style="margin-bottom:15px;">🔍 Live AI-based Waste Classifier ♻️</h1>
    <div style="font-size:22px; opacity:0.9; margin-bottom:15px;">
        An automatic and intelligent classification system based on Computer Vision and hybrid approach combining Machine Learning and Deep Learning.
    </div>
    <div style="font-size:20px; opacity:0.9;">
        🔗 <a href="https://github.com/paulabeltran2101" target="_blank" style="color:white; text-decoration:underline;">
        GitHub Repository
        </a>
    </div>
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
st.write("")

# Cargar la imagen local
img = Image.open("images/grid_clases.png")

# Crear columnas con st.columns
col1, col2 = st.columns([1,1])

# Columna 1: imagen
with col1:
    st.image(img, width=600, use_container_width=False)

# Columna 2: tips dentro de un recuadro
with col2:
    st.markdown("""
    <div style="
        display:flex;
        flex-direction:column;
        justify-content:center;  /* centra verticalmente */
        height:100%;             /* ocupa toda la altura de la columna */
        background: rgba(200, 230, 255, 0.40);
        padding:20px;
        border-radius:15px;
        text-align:center;
        margin-top:200px;
    ">
        <h3>💡 Tips for best results</h3>
        <ul style="font-size:18px; list-style:none; padding-left:0; margin:0;">
            <li>✅ Use well-lit images</li>
            <li>✅ Ensure the waste is clearly visible</li>
            <li>✅ Try one item at a time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="
    display:flex;
    align-items:center;
    margin:40px 0 15px;
">
    <div style="flex:1; height:2px; background:rgba(255,255,255,0.3);"></div>
    <div style="padding:0 15px; font-size:20px;">👉 Select an input mode and get real-time waste classification:</div>
    <div style="flex:1; height:2px; background:rgba(255,255,255,0.3);"></div>
</div>
""", unsafe_allow_html=True)


# Columnas: la primera columna es el margen
col1, col2 = st.columns([1, 20])  # ajusta [1,3] o [1,4] según lo quieras desplazado

with col2:
    mode = st.radio("", ("Upload Image", "Realtime Camera"))

st.markdown("""
<style>

    /* Cambiar tamaño y color de las opciones del radio */
    .stRadio > div[role="radiogroup"] > label > div:nth-child(2) > p {
    font-size: 26px !important;
    font-weight: 600 !important;
    color: white !important;
    }

    /* Espacio entre opciones */
    div[role="radiogroup"] {
        gap: 5px !important;
    }

</style>
""", unsafe_allow_html=True)

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

