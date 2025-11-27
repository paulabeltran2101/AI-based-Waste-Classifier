import tensorflow as tf
import numpy as np
import sys
import os

# añade la carpeta raíz al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extractor import load_conv_base, extract_features_from_image
from utils.load_model import load_model

model_path = 'models/best_norm&model_svm.pkl'

CLASS_NAMES = [
    "Cardboard", "Food Organics", "Glass", "Metal", "Misc Trash",
    "Paper", "Plastic", "Textile Trash", "Vegetation"]

class Classifier():
    def __init__(self):
        self.conv = load_conv_base()
        self.model = load_model(model_path)
        self.last_predictions = []
    
    def predict(self, img, can_predict:bool):
        
        features = extract_features_from_image(img, self.conv)
        prediccion = self.model.predict(features)

        char = CLASS_NAMES[prediccion[0]]
        if can_predict:
            self.last_predictions.append(char)
        return char

    
    def finish_prediction(self):
        if len(self.last_predictions) <=0:
            return None
        char = ''
        count = 0
        for i in self.last_predictions:
            if self.last_predictions.count(i) >count:
                count = self.last_predictions.count(i)
                char = i
        self.reset_predictions()
        return char
    def reset_predictions(self):
        self.last_predictions = []