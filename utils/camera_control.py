import cv2
from utils.image_flow import update_frame, get_frame_size, get_frame_info
from utils.clasiffier import Classifier
from numpy import expand_dims

class WebCamReader():
    def __init__(self, is_web = False):
        self.is_web = is_web
        self.model = Classifier()
        self.pred_class = None

        if not is_web:
            self.cap = cv2.VideoCapture(0)
            # self.width, self.height = 500,500
            # self.width, self.height = 1080,1920
            self.width, self.height = 1920,1080
            self.cap.set(3,self.width)
            self.cap.set(4,self.height)
        
    
    def start(self):
        self.update()

        
    def update(self):
        #captura frame
        if not self.is_web:
            ret, img = self.cap.read()
            if not ret:
                return None
        else:
            img = get_frame_info()
            if img is None:
                return None

        #Predicción
        self.pred_class =self.model.predict(img, can_predict=True)
        
        if self.pred_class is not None:
            cv2.putText(img, f"Predicted: {self.pred_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        update_frame(img)
        return img, self.pred_class
