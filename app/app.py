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

    #Definición clase 
    class VideoTransformer(VideoTransformerBase):

        def __init__(self):
            self.counter = 0
            self.process_every = 5

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            self.counter += 1

            if self.counter % self.process_every == 0:
                resized = cv2.resize(img, (380, 380))
                features = extract_features_from_image(resized, conv_base)
                pred_idx = svm_model.predict(features)[0]
                pred_class = CLASS_NAMES[pred_idx]

                cv2.putText(img, f"Predicted: {pred_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return img
    
    webrtc_streamer(
        key="waste-demo",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
    


   
