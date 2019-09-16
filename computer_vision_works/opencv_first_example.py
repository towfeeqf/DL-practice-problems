import cv2

img = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg", 1)        # read the image in RGB / colored format
img_gr = cv2.imread("E:\DS_ML\OpenCV1\pycharm_work_ocv\zidane.jpg", 0)     # read the image as a gray scale image or black and white image

print(img)
print(img_gr)

print(img.shape)
print(img_gr.shape)

# displaying the image (black and white, gray scale)
cv2.imshow("player",img_gr)     # player : name of the window , can be legend or hero or anything

resized_img = cv2.resize(img_gr,(int(img_gr.shape[1]/2),int(img_gr.shape[0]/2)))
cv2.imshow("small_resize",resized_img)

resized_img = cv2.resize(img_gr,(int(img_gr.shape[1]*2),int(img_gr.shape[0]*2)))
cv2.imshow("big_resize",resized_img)

cv2.waitKey(0)      # wait until a user presses a key

#cv2.waitKey(2000)     # wait untl 2000 milli seconds

cv2.destroyAllWindows()     # closes the window 


