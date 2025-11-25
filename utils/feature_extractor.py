import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
# ---------------------------------------------
# Modelo preentrenado
# ---------------------------------------------
def load_conv_base():
    conv_base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(380, 380, 3)
    )
    return conv_base

# ---------------------------------------------
# Preprocess and extract features from a image
# ---------------------------------------------
def extract_features_from_image(img, conv_base):
    
    # Redimensionar a 380x380
    img_resized = cv2.resize(img, (380, 380))
    
    # Convertir a float32 y expandir dimensiones
    img_array = img_resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess (EfficientNet)
    img_array = preprocess_input(img_array)

    # Extract features
    features = conv_base.predict(img_array, verbose=0)
    return features 
