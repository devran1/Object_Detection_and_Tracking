#!/usr/bin/env python3.8

import time
import cv2
import numpy as np
from numpy import random
from display import Display
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Button

W = 500 #input array
H = 500 #input array

disp = Display(W,H)
orb = cv2.ORB_create()



class FeatureExtractor(object):   
    def __init__(self):
        self.orb = cv2.ORB_create(1000)        
    def extract(self, img): # in paint (extract is object functions are objects)      
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance=5) #numpy.AxisError: axis 3(must be 2 we said img=2) is out of bounds for array of dimension 3 quality level is (detected) dots
        return feats      
fe = FeatureExtractor()
def process_frame(img):
    img = cv2.resize(img,(W,H))
    kp = fe.extract(img) # <---kp is here    
    for p in kp: #         <---kp
        u,v = map(lambda x: int(round(x)), p[0]) # p[0] must be 0 otherwise index 1 is out of bounds for axis 0 with size 1 tuple and lamba tuple...?
        cv2.circle(img, (u,v), color=(0,255,0), radius=1)
        
        #color u,v points
        color = img[u,v]
        print("u,v", u,v)
        print ('color', color)
        
        #writing file
        with open("data.txt", "w") as file:
            file.write('{} / {} / {}'.format(u,v,color))
            #file.write('%s , %s , %s'% (u,v,color))
            file.close()        
    disp.paint(img)  #process_frame    
if __name__ == "__main__":
    cap = cv2.VideoCapture("Walking.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret== True:
            process_frame(frame)
        else:
            break    
