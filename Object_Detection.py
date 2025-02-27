#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms

import pygame
import cv2
import numpy as np

from centroidcode import CentroidTracker
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

pygame.init()

transform = transforms.Compose([transforms.ToTensor()])


def predict(image, model, device, detection_threshold):
   
    image = transform(image).to(device)
    # adds a batch dimension    
    image = image.unsqueeze(0) 
    with torch.no_grad():
        # get the predictions on the image        
        outputs = model(image)
    
    scores = list(outputs[0]['scores'].detach().numpy())
    
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]

    # get all the predicted bounding boxes
    bboxes = outputs[0]['boxes'].detach().numpy()

    # get boxes above the threshold score   
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)
    
    labels = outputs[0]['labels'].numpy()
    pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes


threshold=0.6   #accuracy (and also effects the number of detection)

#model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, num_classes=91, min_size=200)
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91, min_size=200)

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


device = torch.device( 'cpu')
model = model.eval() 

cap = cv2.VideoCapture("video.mp4")


w,h=640,360

gameDisplay = pygame.display.set_mode((w,h))


frame_count=0
object_id_list = []

all_dict={}
while(cap.isOpened()):
    ret, frame1 = cap.read()

    frame=cv2.resize(frame1,(w,h))    
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

        # id and x,y position
        a=(text,(x1+x2)//2,(y1+y2)//2)
        print(a)
       
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 0), 1)

        if objectId not in object_id_list:
            object_id_list.append(objectId)
    cv2.imshow("App", frame)    

    with open("all.txt","w") as file:
        for key,value in all_dict.items():
            file.writelines(f"{key,value}\n")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break