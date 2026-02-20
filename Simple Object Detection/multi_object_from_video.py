import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('./Simple Object Detection/video.mp4')

model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()

    results = model(frame, classes = [0])

    annotated_frame = results[0].plot()
    resize_frame = cv2.resize(annotated_frame, (800, 600))
    cv2.imshow('Object Detection', resize_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()