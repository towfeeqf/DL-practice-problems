# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:04:17 2019

@author: SonyTF
"""

import cv2
import os

# read the image in its true color and gray scale
img = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg")

#gray_img2 = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg",0)
#or
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


### face detection####

haar_path =cv2.data.haarcascades

face_cascade_xml=os.path.join(haar_path,"haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(face_cascade_xml)

# now search the coordinates of the face image
faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05, minNeighbors=5)

print(type(faces))
print(faces)


 
# Adding the rectangular face box
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    # rectangle : a method to create a rectangle
    # 3 is the width of the rectangle
    # coordinates are the RGB values of the rectangle outline

resized_img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
# reshaping the image so as to properly display 
cv2.imshow("resized_pic",resized_img)

cv2.imshow("actual_pic",img)

cv2.waitKey(0)
cv2.destroyAllWindows
