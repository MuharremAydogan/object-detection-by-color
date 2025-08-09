import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,960)
buffer_size=16
blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

buffer_array=deque(maxlen=buffer_size)

if not cap.isOpened():
    print("kamera bulunamadi")
    
    
while True:
    suc,frame=cap.read()
    if not suc:
        print("frameler okunamadi")
    
    blured_frame=cv2.blur(frame,ksize=(1,1))
    hsv_img=cv2.cvtColor(blured_frame,cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",hsv_img)
    
    
    mask=cv2.inRange(hsv_img, lowerb=blueLower, upperb=blueUpper)
    #cv2.imshow("maskelenmis",mask)                                  
    
    
    mask=cv2.erode(mask,kernel=(3,3), iterations=2)
    mask=cv2.dilate(mask, kernel=(3,3),iterations=2)
    #cv2.imshow("gurultu giderilmis maske",mask)
    
    
    
    
    (contours,_)=cv2.findContours(mask.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) 
    
    center=None
    
    if len(contours)>0:
        c=max(contours,key=cv2.contourArea)
        
        rect=cv2.minAreaRect(c)
        
        ((x,y), (width,height), rotation) = rect
        
        s = "x:{},y:{},width:{},height:{},rotation:{}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
       

        kutucuk=cv2.boxPoints(rect)
        box=np.int64(kutucuk)
        
        center_x = np.mean(kutucuk[:, 0])  
        center_y = np.mean(kutucuk[:, 1])  
        center = (center_x, center_y)
        
        cv2.drawContours(frame, [box], 0, (0,255,255),2)
            
        
        cv2.circle(frame, np.round(center).astype(int), 5, (255,0,255),-1)
            
        
        cv2.putText(frame, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
    
        #cv2.imshow("boxed img",frame)
        
    buffer_array.appendleft(np.round(center).astype(int))
    
    for i in range(1,len(buffer_array)):
        if buffer_array[i-1] is None or buffer_array is None: continue
        
        cv2.line(frame,buffer_array[i-1],buffer_array[i],(0,255,0),3)
    
    cv2.imshow("Center Track", frame)
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
cv2.destroyAllWindows()
cap.release()