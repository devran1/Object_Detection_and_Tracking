import time
import cv2
import numpy as np

from display import Display
from extractor import Extractor

W = 450
H = 450

F = 270
disp = Display(W,H)
K = np.array([[F,0,-W//2],[0,F,-H//2],[0,0,1]])
print(K)
fe = Extractor(K)

def process_frame(img):
    img = cv2.resize(img,(W,H))
    matches = fe.extract(img)
    
    print("%d matches" % (len(matches)))

    def denormalize(pt):
        return int(round(pt[0] + img.shape[0]/2)), int(round(pt[1] + img.shape[1]/2))    

    for pt1,pt2 in matches:
        
       
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)

        #denormalize for display
        #u1 += img.shape[0]
        #u2 += img.shape[0]
        #v1 += img.shape[1]
        #v2 += img.shape[1]
        
        cv2.circle(img, (u1,v1), color=(0,255,0), radius=1)
        cv2.line(img, (u1, v1), (u2,v2), color=(255,0,0))
    

    disp.paint(img)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture("Walking.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret== True:
            process_frame(frame)
        else:
            break    

##2.19
