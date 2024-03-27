import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
img = cv2.imread("E:/Computer Vision slot D/programs/picture1.png")
img = cv2.resize(img,(600,600))
cv2.imshow("image",img)
cv2.waitKey(0)
