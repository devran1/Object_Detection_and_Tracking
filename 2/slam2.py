import time
import cv2
import numpy as np
from display import Display
from extractor import Extractor

W = 500
H = 500

disp = Display(W,H)
fe = Extractor()



def process_frame(img):
    img = cv2.resize(img,(W,H))
    kps, des, matches = fe.extract(img)
    if matches is None:
        return
    
    for p in kps:
        u,v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (u,v), color=(0,255,0), radius=1)

    disp.paint(img)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture("Walking.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret== True:
            process_frame(frame)
        else:
            break    


