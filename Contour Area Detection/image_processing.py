import cv2

img = cv2.imread('./Contour Area Detection/image1.jpg')

resizing = cv2.resize(img, (200, 200))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(img, (15, 15), 0)
edges = cv2.Canny(img, 100, 200)

cv2.imshow('Resizing', resizing)
cv2.imshow('Grey', grey)
cv2.imshow('Blurring', blurring)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
