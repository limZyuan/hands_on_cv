import cv2
import numpy as np

canvas = np.zeros((512, 512, 3), dtype=np.uint8)

cv2.line(canvas, (0, 0), (512, 512), (0, 255, 0), thickness=5)
cv2.rectangle(canvas, (100, 100), (300, 300), (255, 0, 0), thickness=3)
cv2.putText(canvas, 'OpenCV', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()