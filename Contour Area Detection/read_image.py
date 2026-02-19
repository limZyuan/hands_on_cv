import cv2

image1 = cv2.imread('./Contour Area Detection/image1.jpg')
cv2.imshow('Image', image1)
cv2.waitKey(0)

cv2.imwrite('./Contour Area Detection/image1_copy.jpg', image1)