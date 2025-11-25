import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

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
    img = img.resize((380, 380))
    img_array = image.img_to_array(img)

    # Preprocess (EfficientNet)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features
    features = conv_base.predict(img_array, verbose=0)
    return features.flatten()
