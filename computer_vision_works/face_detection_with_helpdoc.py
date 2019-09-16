# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:24:25 2019

@author: SonyTF
"""

####for help with documentation


import cv2
import os

#haar_path= cv2.data.haarcascades
haar_path="C:\\Users\\SonyTF\\Anaconda3\\lib\\site-packages\\cv2\\data\\"
# path to the filw which contains the face features aka xml file

cascade_file=os.path.join(haar_path,"haarcascade_frontalface_default.xml")

# create a cascadecalssifier object
face_cascade= cv2.CascadeClassifier(cascade_file)
# Opencv already contains many pre-trained classifiers for face, eyes, smile etc.


# reading the image as it is
#img = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg") # reading the image as it is

img = cv2.imread("zidane.jpg") # reading the image as it is
#print(img)

# reading the image as a gray scale image
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
# for BGR to GRAY scale conversion we use flags cv2.COLOR_BGR2GRAY

#cv2.imshow("player",gray_img)

# the error says that the image you are trying to convert to grayscale has no color channels


# search the co-ordinates of the image
faces= face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05, minNeighbors=5)

#scalefactor : decreases the shape value by 5%, until the face is found.
#smaller this value. the greater is the accuracy



print(type(faces))
print(faces)



