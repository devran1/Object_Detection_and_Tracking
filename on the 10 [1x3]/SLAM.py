#!/usr/bin/python3.8

import time
from cv2 import cv2
#import cv2
from display import Display
from extractor import Extractor
import numpy as np


W = 1920//2 #input array
H = 1080//2 #input array


F = 1


disp = Display(W,H)
K =np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))

print(K)
fe = Extractor(K)


      
def process_frame(img):
    img = cv2.resize(img,(W,H))
    matches = fe.extract(img) # <---kp is here
    
    print("%d matches" %(len(matches)))
   
    
    #def denormalize(pt):
    #    return int(round(pt[0]+img.shape[0]/2)), int(round(pt[1]+img.shape[1]/2))

    for pt1, pt2 in matches :
        u1,v1 = fe.denormalize(pt1)#map(lambda x: int(round(x)), pt1) # p[0] must be 0 otherwise index 1 is out of bounds for axis 0 with size 1 tuple and lamba tuple...?
        u2,v2 = fe.denormalize(pt2)

        
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=1) #line maker 1 (error no dots)
        cv2.line(img, (u1, v1),(u2, v2), color=(255,0,0)) # line maker2


    disp.paint(img)  #process_frame
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret== True:
            process_frame(frame)
        else:
            break    