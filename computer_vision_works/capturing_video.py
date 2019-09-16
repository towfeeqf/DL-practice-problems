# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:25:17 2019

@author: SonyTF
"""

import cv2
import time

video = cv2.VideoCapture(0)

check, frame =  video.read()
# check is a bool data type, return true if Python is able to read the 
# Video captue

print(check)
print(frame)

time.sleep(3)
video.release()


# add a window that shows the video
