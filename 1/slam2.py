import time
import cv2
from display import Display
import numpy as np

W = 500
H = 500

disp = Display(W,H)
orb = cv2.ORB_create()
print(dir(orb))


class FeatureExtractor(object):
    GX = 8
    GY = 6
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        

    def extract(self, img):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance=3)
        print(feats)
        return feats

    
fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img,(W,H))
    kp = fe.extract(img)

    
    for p in kp:
        #u,v = map(lambda x: int(round(x)), f[0])
        u,v = map(lambda x: int(round(x)), p[0])
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



