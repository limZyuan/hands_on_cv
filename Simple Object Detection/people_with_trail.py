import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('./Simple Object Detection/walkers.mp4')

id_map = {}
nex_id = 1

trail = defaultdict(lambda: deque(maxlen=100))
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = model.track(frame, classes = [0], persist=True, verbose=False)
    annnotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            # center of the box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            appear[oid] += 1

            # only start assigning IDs once the object has been seen for a few frames
            if appear[oid] > 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            if oid in id_map:
                sid = id_map[oid]
                # update trail history
                trail[oid].append((cx, cy))

                # draw bounding box and ID
                cv2.rectangle(annnotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(annnotated_frame, f'ID: {sid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.circle(annnotated_frame, (cx, cy), 1, (0, 255, 0), -1)

                # draw trail as a polyline connecting past centers
                pts = np.array(trail[oid], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(annnotated_frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            
    cv2.imshow('Annotated Frame', annnotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
