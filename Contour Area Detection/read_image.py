import cv2

image1 = cv2.imread('./hands_on_cv/image1.jpg')
cv2.imshow('Image', image1)
cv2.waitKey(0)

cv2.imwrite('./hands_on_cv/image1_copy.jpg', image1)