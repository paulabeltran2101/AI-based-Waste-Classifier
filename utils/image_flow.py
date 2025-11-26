# import cv2
import numpy as np 

frame = None
def get_frame_info():
    return frame

def update_frame(fr):
    global frame
    frame = fr

def get_frame_size():
    global frame
    return np.array(frame).shape