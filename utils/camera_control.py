import cv2
from utils.image_flow import update_frame, get_frame_size, get_frame_info
from utils.classifier import Classifier
from numpy import expand_dims

class WebCamReader():
    def __init__(self, camera_index = 1):
        self.cap = cv2.VideoCapture(camera_index)
        self.width, self.height = 1280, 720
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        self.model = Classifier()
        self.pred_class = None
    
    def update(self):
        #captura frame
        ret, img = self.cap.read()
        if not ret:
            return None
        
        self.pred_class = self.model.predict(img, can_predict=True)

        if self.pred_class:
            cv2.putText(img, f"Predicted: {self.pred_class}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        return img

    def release(self):
        if self.cap:
            self.cap.release()
