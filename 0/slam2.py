#!/usr/bin/python3.8

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
        # run detect in grid
      #  sy = img.shape[0]//self.GY
      #  sx = img.shape[1]//self.GX
      #  akp =[]
      #  for ry in range(0,img.shape[0], sy):
       #     for rx in range(0,img.shape[1], sx):
       #         img_chunk = img[ry:ry+sy, rx:rx+sx]               
       #         kp = self.orb.detect(img_chunk, None)
       #         for p in kp:
       #             p.pt = (p.pt[0] + rx, p.pt[1] + ry)
       #             akp.append(p)
      #  return akp
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance=3)
        print(feats)
        return feats

    
fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img,(W,H))
    kp = fe.extract(img)

    
    for p in kp:
        u,v = map(lambda x: int(round(x)), p[0])
        cv2.circle(img, (u,v), color=(0,255,0), radius=1)

    disp.paint(img)
    
if __name__ == "__main__":
    x = input("Video Name:")
    cap = cv2.VideoCapture(x)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret== True:
            process_frame(frame)
        else:
            break    



