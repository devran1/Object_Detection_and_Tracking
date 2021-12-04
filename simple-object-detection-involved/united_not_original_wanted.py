#!/usr/bin/env python3


import torchvision
import torchvision.transforms as transforms
import cv2
import torch
import argparse
import time
import numpy as np
#import detect_utils
from PIL import Image
from centroidcode import CentroidTracker

import pygame

#from deep_sort.tracker import Tracker

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

pygame.init()


transform = transforms.Compose([transforms.ToTensor()])


def predict(image, model, device, detection_threshold):
   
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image) # get the predictions on the image
    
    scores = list(outputs[0]['scores'].detach().numpy())
    
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]
    # get all the predicted bounding boxes
    bboxes = outputs[0]['boxes'].detach().numpy()
    # get boxes above the threshold score
   
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)
    
    labels = outputs[0]['labels'].numpy()
    pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes


threshold=0.6   #accuracy (and affects number of detection)

#model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, num_classes=91, min_size=200) #min_size=speed,number of detects
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91, min_size=200)


tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


device = torch.device( 'cpu')
model = model.eval() 

#3rd_phase_2.mp4
#FIND A GOOD VİDEO FOR USE
cap = cv2.VideoCapture("Walking.mp4") #chr.mp4 wonderful example

#fps = cap.get(cv2.CAP_PROP_FPS)
#print(fps)
#frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print(frame_count)

#duration = frame_count/fps
#print(duration)

#113
#4.7280334728033475
#0.041769912 time



#w,h=640,360#,640#,360#,640#,360#r 360,683 #r 1366,720 #p 640,360#ıd 360,640


#Bowling model 2 600

#1280-720 VV #with which model 2,1 edit video?
#chr.mp4 V # model 1

#Pexels-1280-720 XX
#lacy-1280-720.mp4 XX

#we dont need classes to tell us what is what
#k not seeing the ball
#a.mp4 fix size of the video
#v.mp4 close but fix print
#h.mp4 not detected
#i.mp4 more boxes on same place
#no s.mp4 ,g.mp4
#polina,blue-bird,dog_only

#chr 360,640

w,h=640,360

gameDisplay = pygame.display.set_mode((w,h))


frame_count=0
object_id_list = []

all_dict={}
while(cap.isOpened()):
    ret, frame1 = cap.read()
    #print(frame1.shape[0],frame1.shape[1]) # for ID 720,1280 (360,640)
    frame=cv2.resize(frame1,(w,h))
    #w,h=640,360




###########################################################################################################
#length of the (depth of the videos)? (horizontal)

    #ave=0.778
    #left_side
    #1
   
    #cv2.circle(frame, (257,297), 1,  (0,0,255),-1) 
    #cv2.circle(frame, (255,312), 1,  (0,0,255),-1) 
   
    #dx=257-255=2 dy=312-297=15 tanx=2/15=0.133  ilkinin boyu A1=(dx^2+dy^2)^(1/2)=15,13 #A1 kullanıırsa

    #2
    #cv2.circle(frame, (263,252), 1,  (0,0,255),-1) 
    #cv2.circle(frame, (261,265), 1,  (0,0,255),-1) 

    #dx=2 dy=13 A2=13.153
    #A2/A1=13.153/15.13 = 0.869

    #3

  #  cv2.circle(frame, (268,220), 1,  (0,0,255),-1) 
  #  cv2.circle(frame, (266,230), 1,  (0,0,255),-1)


    #dx=2 dy=10 A3=10.198 A3/A2=0.775   A3/A1=0.764

    #4

  #  cv2.circle(frame, (270,198), 1,  (0,0,255),-1) 
  #  cv2.circle(frame, (269,206), 1,  (0,0,255),-1)

    #dx=1, dy=8, A4=8.0623 A4/A3=0.790 A4/A2=0.613 A4/A1=0.533



    #right_side
    #1
  #  cv2.circle(frame, (378,322), 1,  (0,0,255),-1) 
  #  cv2.circle(frame, (383,341), 1,  (0,0,255),-1) 

    #dx=5 dy=19 A1=19.647

    #2
  #  cv2.circle(frame, (363,270), 1,  (0,0,255),-1) 
  #  cv2.circle(frame, (366,283), 1,  (0,0,255),-1) 

    #dx=3 dy=13 A2=13.342 A2/A1=0.679

###################################################################################################################
#direk=pole=column length (vertical) #not right
#ave=0.51

#    cv2.circle(frame, (51,42),  1,  (0,0,255),-1) 
#    cv2.circle(frame, (55,252), 1,  (0,0,255),-1) #dx=4 dy=210 A1=210.038 

#    cv2.circle(frame, (168,38),  1,  (0,0,255),-1) 
#    cv2.circle(frame, (172,146), 1,  (0,0,255),-1) #dx=4, dy=108 A2=108.074 A2/A1=0.5145


#    cv2.circle(frame, (288,28),  1,  (0,0,255),-1) 
#    cv2.circle(frame, (291,86), 1,  (0,0,255),-1) #dx=3 dy=56 A3=56.0802 A3/A2= 0.5189  A3/A1=0.2670


##################################################################################################################
############################################################################################################
#angle calculation               A=(dx^2+dy^2)^(1/2
#    cv2.circle(frame, (220,160), 5,  (0,0,255),-1) 
    
#    cv2.line(frame, (220,160),(107,360),(0,255,0),2)

    #dx=220-107=113 dy=360-160=200    #length of the vector A=229.71 
    #cosx=113/229.71=0.491 A.cosx=113 vectorun o yondeki izdüşümü #Acosx=229.71 * 
    #sinx=200/229.71=0.871 A.sinx=200

#do i need A.sinx ?

#    angle1=abs((220-107)/(360-160))
#    print(angle1) #0.565 positive
#   


#    cv2.circle(frame, (385,160), 5,  (0,0,255),-1) 
#    cv2.circle(frame, (527,360), 5,  (0,0,255),-1) 

#    cv2.line(frame, (385,160),(527,360),(0,255,0),2)

#    angle2=abs((527-385)/(360-160))
#    print(angle2) #0.71 negative



########################################################################################################
#perspective warp
    #fixed
    #cv2.circle(frame, (240,160), 5,  (0,0,255),-1) 
    #cv2.circle(frame, (385,160), 5,  (0,0,255),-1) 
    #cv2.circle(frame, (135,360), 5,  (0,0,255),-1) 
    #cv2.circle(frame, (527,360), 5,  (0,0,255),-1) 

    #pts1=np.float32([[240,160],[385,160],[135,360],[527,360]])
    #pts2 =np.float32([[0,0],[640,0],[0,360],[640,360]])

    #matrix =cv2.getPerspectiveTransform(pts1,pts2)
    #result = cv2.warpPerspective(frame, matrix,(640,360))
    #result = cv2.warpPerspective(frame, matrix,(640,200)) #do same frame width-height calculations here
    #frame=result
   
    #left_side
    #cv2.circle(frame, (479,336), 1,  (0,0,0),-1)
    #cv2.circle(frame, (575,336), 1,  (0,0,0),-1) #96#
    #cv2.circle(frame, (274,720), 1,  (0,0,0),-1) #720-336=384
    #cv2.circle(frame, (537,720), 1,  (0,0,0),-1) #537-274=263

    #pts1=np.float32([[479,336],[575,336],[274,720],[537,720]])
    #pts2 =np.float32([[0,0],[263,0],[0,384],[263,384]])
   
    #matrix =cv2.getPerspectiveTransform(pts1,pts2)
    #result = cv2.warpPerspective(frame, matrix,(263,384))
    #frame=result

    #cv2.imshow("pers",result)

    #middle

  #  cv2.circle(frame, (569,366), 1,  (0,0,0),-1) #680-569=111
  #  cv2.circle(frame, (680,366), 1,  (0,0,0),-1) #789-527=262 #263 al
  #  cv2.circle(frame, (527,720), 1,  (0,0,0),-1) 
  #  cv2.circle(frame, (789,720), 1,  (0,0,0),-1) 
    
    #pts1=np.float32([[568,366],[677,366],[527,720],[789,720]]) #
    #pts2 =np.float32([[0,0],[263,0],[0,384],[263,384]]) #
    #matrix =cv2.getPerspectiveTransform(pts1,pts2)
    #result = cv2.warpPerspective(frame, matrix,(263,384))
    #frame=result


    #right

    #cv2.circle(frame, (680,366),  1,  (0,0,0), -1)
    #cv2.circle(frame, (787,366),  1,  (0,0,0), -1) #720-366=384
    #cv2.circle(frame, (789,720),  1,  (0,0,0), -1) #787-680=107
    #cv2.circle(frame, (1054,720), 1,  (0,0,0), -1) #1054-789=265
    
  ##  pts1=np.float32([[680,366],[787,366],[789,720],[1054,720]]) #
  ##  pts2 =np.float32([[0,0],[265,0],[0,384],[265,384]]) #

  ##  matrix =cv2.getPerspectiveTransform(pts1,pts2)
  ##  result = cv2.warpPerspective(frame, matrix,(265,384))
  ##  frame=result
    
   
    #cv2.imshow('perspective',result)


#####################################################################################################################
    
    frame_count += 1
    #print("frame_number",frame_count)
    boxes, classes =predict(frame, model, device, threshold)

    objects=tracker.update(boxes)


    text_count=0
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        #print(classes,bbox)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        #cv2.line(frame, (0,h),((x1+x2)//2,(y1+y2)//2),(0,255,0),2)
        
        pygame.draw.circle(gameDisplay, (255,255,255),((x1+x2)//2,(y1+y2)//2), radius=1.0,width=1)
        for event in pygame.event.get(1):
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        pygame.display.update()

        text = "ID_{}".format(objectId)
        text_count +=1
        #only one thing
        if text == "ID_0":

            pass
        
        for i in range(text_count):
            if text == f"ID_{i}":
                a=[(x1+x2)//2,(y1+y2)//2]
                all_dict.setdefault(f"ID_{i}",[])
                all_dict[f"ID_{i}"].append(a)

        a=(text,(x1+x2)//2,(y1+y2)//2)
        print(a)
       
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

        if objectId not in object_id_list:
            object_id_list.append(objectId)
    cv2.imshow("App", frame)    



    with open("all.txt","w") as file:
        for key,value in all_dict.items():
            file.writelines(f"{key,value}\n")
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

