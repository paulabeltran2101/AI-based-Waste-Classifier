import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# añade la carpeta raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#From project
from utils.feature_extractor import load_conv_base, extract_features_from_image
from utils.load_model import load_model
from utils.image_flow import get_frame_info, update_frame
from utils.camera_control import WebCamReader
from utils import time_control

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
    
    #Definición clase 
    class VideoTransformer(VideoTransformerBase):

        def __init__(self):
            #self.time_controller = time_control.FrameTimer()
            self.wc = WebCamReader(camera_index=1)
            self.pred_class = False

        def transform(self, frame):
            #Convertir frame
            img = frame.to_ndarray(format="bgr24")
            processed = self.wc.update()
            
            if processed is not None:
                self.pred_class = self.wc.pred_class
                return processed
            return img
    

    col1, col2 = st.columns([2, 1])  # columna izquierda más grande para vídeo
    
    with col1:
        ctx = webrtc_streamer(
            key="waste-demo",
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={
                "video": {
                    "width": {'ideal':1280}, 
                    "height": {'ideal':720}
                    },
                    "audio": False}
        )
    with col2:
        # Placeholder para predicción de texto
        pred_placeholder = st.empty()
        while True:
            if ctx and ctx.video_transformer:
                pred_cls = ctx.video_transformer.pred_class
                pred_placeholder.markdown(f"### Predicción en tiempo real\n**{pred_cls if pred_cls else 'Esperando predicción…'}**")
            else:
                pred_placeholder.markdown("Conectando…")
           # time.sleep(0.1)
            
