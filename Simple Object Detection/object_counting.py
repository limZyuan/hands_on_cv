import cv2
from ultralytics import YOLO
import numpy

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('./Simple Object Detection/bottles.mp4')

unique_ids = set()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream or error.")
        break

    results = model.track(frame, classes = [39], persist=True, verbose=False)
    annotated_frame = results[0].plot()
    resize_frame = cv2.resize(annotated_frame, (800, 600))

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for oid in ids:
            unique_ids.add(oid)
        cv2.putText(resize_frame, f'Count: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', resize_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

