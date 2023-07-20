import cv2
import numpy as np
import imutils

#START THE VIDEOCAPTURE 
Cap=cv2.VideoCapture(0)

#VALUES TO DETECT COLORS IN HSV SCALE
Dark_red1= np.array([0,50,120], np.uint8)
Light_red1= np.array([10,255,255], np.uint8)

Dark_red2= np.array([170,100,100], np.uint8)
Light_red2= np.array([179,255,255], np.uint8)
    
Dark_green= np.array([40,70,80], np.uint8)
Light_green= np.array([70,255,255], np.uint8)
    
Dark_blue= np.array([90,60,0], np.uint8)
Light_blue= np.array([121,255,255], np.uint8)
    
Dark_yellow= np.array([25,70,100], np.uint8)
Light_yellow= np.array([30,255,255], np.uint8)
    
while True:
    #READ THE FRAMES AND CHECK THE RET
    Ret, Frame= Cap.read()
    if Ret==True:
    #TRANSORM THE ORIGINALS BGR FRAMES TO HSV TO DETECT COLORS
        Transform_hsv= cv2.cvtColor(Frame,cv2.COLOR_BGR2HSV)
    
    #CREATE A MASK TO EVERY COLOR WITH THE DARK AND LIGHT VALUE
        Mask_red1=cv2.inRange(Transform_hsv, Dark_red1, Light_red1)
        Mask_red2=cv2.inRange(Transform_hsv, Dark_red2, Light_red2)
        Mask_red_end=Mask_red1 + Mask_red2
        
        Mask_green=cv2.inRange(Transform_hsv, Dark_green, Light_green)
        Mask_blue=cv2.inRange(Transform_hsv, Dark_blue, Light_blue)
        Mask_yellow=cv2.inRange(Transform_hsv, Dark_yellow, Light_yellow)
        
        #THRESHOLD 
        Thresh=100
        Ret, Thresh_red= cv2.threshold(Mask_red_end,Thresh,255,cv2.THRESH_BINARY)
        Ret, Thresh_green= cv2.threshold(Mask_green,Thresh,255,cv2.THRESH_BINARY)
        Ret, Thresh_blue= cv2.threshold(Mask_blue,Thresh,255,cv2.THRESH_BINARY)
        Ret, Thresh_yellow= cv2.threshold(Mask_yellow,Thresh,255,cv2.THRESH_BINARY)
    
    #FIND THE CONTOUR OF THE OBJECTS TO DETECT 
        Countour1= cv2.findContours(Thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Countour1= imutils.grab_contours(Countour1)
    
        Countour2= cv2.findContours(Thresh_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Countour2= imutils.grab_contours(Countour2)
    
        Countour3= cv2.findContours(Thresh_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Countour3= imutils.grab_contours(Countour3)
    
        Countour4= cv2.findContours(Thresh_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Countour4= imutils.grab_contours(Countour4)
    
    
    #DRAW THE CONTOURS, A CIRCLE AND PUT TEXT FOR EVERY CONTOUR IN THE FRAMES
        for c in Countour1:
            Area1=cv2.contourArea(c)
            if Area1 > 4000:
                cv2.drawContours(Frame,[c],-1,(0,255,0),2)
                M=cv2.moments(c)
                x=int(M["m10"]/M["m00"])
                y=int(M["m01"]/M["m00"])
                cv2.circle(Frame,(x,y),5,(255,255,255),-1)
                cv2.putText(Frame,"Red",(x-20,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
            
        for c in Countour2:
            Area2=cv2.contourArea(c)
            if Area2 > 4000:
                cv2.drawContours(Frame,[c],-1,(0,255,0),2)
                M=cv2.moments(c)
                x=int(M["m10"]/M["m00"])
                y=int(M["m01"]/M["m00"])
                cv2.circle(Frame,(x,y),5,(255,255,255),-1)
                cv2.putText(Frame,"Green",(x-20,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
            
        for c in Countour3:
            Area3=cv2.contourArea(c)
            if Area3 > 4000:
                cv2.drawContours(Frame,[c],-1,(0,255,0),2)
                M=cv2.moments(c)
                x=int(M["m10"]/M["m00"])
                y=int(M["m01"]/M["m00"])
                cv2.circle(Frame,(x,y),5,(255,255,255),-1)
                cv2.putText(Frame,"Blue",(x-20,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
                
        for c in Countour4:
            Area4=cv2.contourArea(c)
            if Area4 > 4000:
                cv2.drawContours(Frame,[c],-1,(0,255,0),2)
                M=cv2.moments(c)
                x=int(M["m10"]/M["m00"])
                y=int(M["m01"]/M["m00"])
                cv2.circle(Frame,(x,y),5,(255,255,255),-1)
                cv2.putText(Frame,"Yellow",(x-20,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
        
    
        cv2.imshow("COLOR DETECT", Frame)
        t=cv2.waitKey(1)
        if t==27:break
        
Cap.release()
cv2.destroyAllWindows()
