# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:04:17 2019

@author: SonyTF
"""

import cv2
import os

# read the image in its true color and gray scale
#img = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg")

img = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\many.jpg")

zoom_img = cv2.resize(img,(int(img.shape[1]*2),int(img.shape[0]*2)))

#gray_img2 = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg",0)
#or
gray_img=cv2.cvtColor(zoom_img,cv2.COLOR_BGR2GRAY)

### face detection####

haar_path =cv2.data.haarcascades


face_cascade_xml=os.path.join(haar_path,"haarcascade_frontalface_default.xml")
eye_cascade_xml = os.path.join(haar_path,"haarcascade_eye.xml")
smile_cascade_xml= os.path.join(haar_path,"haarcascade_smile.xml") 

face_cascade = cv2.CascadeClassifier(face_cascade_xml)
eye_cascade = cv2.CascadeClassifier(eye_cascade_xml)
smile_cascade= cv2.CascadeClassifier(smile_cascade_xml)


#Coordinates of the face image....followed by coordinates of eyes...followed by smile...
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05, minNeighbors=5)

#the haar cascades work better on gray images
#gray_faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05, minNeighbors=5)

print(type(faces))
print(faces)

# We process the gray scale image (gray_faces) as haar cascades work better on them 
for x,y,w,h in faces:
    img_new2=cv2.rectangle(zoom_img,(x,y),(x+w,y+h),(0,255,0),3)
    roi_gray=gray_img[y:y+h,x:x+h]
    roi_color= zoom_img[y:y+h,x:x+h]
    
    eyes=eye_cascade.detectMultiScale(roi_gray)
    for ex,ey,ew,eh in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        
    smiles=smile_cascade.detectMultiScale(roi_gray,1.15,15)
    for sx,sy,sw,sh in smiles:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    

actual_img = cv2.resize(img_new2,(int(img_new2.shape[1]/2),int(img_new2.shape[0]/2)))


cv2.imshow("zoom_pic",img_new2)

cv2.imshow("given_pic",actual_img)


cv2.waitKey(0)
cv2.destroyAllWindows
