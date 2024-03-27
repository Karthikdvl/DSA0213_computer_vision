import cv2
import numpy as np
# read the input image
img = cv2.imread("E:/Computer Vision slot D/programs/picture1.png")
# access the image height and width
rows,cols,_ = img.shape
# define at three point on input image
pts1 = np.float32([[50,50],[200,50],[50,200]])
# define three points corresponding location to output image
pts2 = np.float32([[10,100],[200,50],[100,250]])
# get the affine transformation Matrix
M = cv2.getAffineTransform(pts1,pts2)
# apply affine transformation on the input image
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("Affine Transform", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
