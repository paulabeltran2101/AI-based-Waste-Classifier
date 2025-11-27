import cv2
from utils.image_flow import update_frame, get_frame_size, get_frame_info
from utils.classifier import Classifier
from numpy import expand_dims

class WebCamReader():
    def __init__(self, camera_index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.model = Classifier()
        self.pred_class = None
        self.width = width
        self.height = height
    
    def read_frame(self):
        #captura frame de la cámara y calcula predicción
        ret, frame = self.cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara")
            return None, None
        
        # Predicción usando tu Classifier
        self.pred_class = self.model.predict(frame, can_predict=True)
        print("Predicción: ", self.pred_class)

        cv2.putText(frame, f"Predicted: {self.pred_class}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        
        update_frame(frame)
        return frame, self.pred_class

    def release(self):
        if self.cap:
            self.cap.release()
            print('Cámara liberada')
