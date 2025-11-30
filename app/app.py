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
/* fondo completo */
.stApp {
    background-image: url("https://images.unsplash.com/photo-XXXXXXXXXXXX?auto=format&fit=crop&w=1350&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* texto blanco */
html, body, [class*="css"] {
    color: white !important;
}

/* títulos centrados */
h1, h2, h3, h4 {
    text-align: center;
    color: white !important;
}

/* centrar todo el contenido */
.main > div {
    display: flex;
    flex-direction: column;
    align-items: center;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


st.set_page_config(page_title="♻️Waste Classifier", layout="centered")
st.title("🔍 Live AI-based Waste Classifier ♻️")

# Descripción del proyecto
st.markdown(
    """
### Un sistema de clasificación automática de residuos mediante visión artificial y aprendizaje automático.  
Este proyecto identifica tipos de basura en tiempo real usando un modelo CNN + SVM optimizado.

🔗 Repositorio: [Mi GitHub](https://github.com/paulabeltran2101)
"""
)

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
    
