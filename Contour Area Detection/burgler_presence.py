import cv2

cap = cv2.VideoCapture('./hands_on_cv/vid.mp4')

frames = []
gap = 5
count = 0

# Properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./Contour Area Detection/output.mp4', fourcc, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames) > gap + 1:
        frames.pop(0)
    
    cv2.putText(frame, f'Frame Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if len(frames) > gap:
        diff = cv2.absdiff(frames[0], frames[-1])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 100:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        motion = any(cv2.contourArea(c) > 100 for c in contours)

        if motion:
            cv2.putText(frame, 'Motion Detected!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imwrite(f'motion_{count}.jpg', frame)
            print(f'Motion detected at frame {count}')

        cv2.imshow('Burglar Presence Detection', frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break 

    out.write(frame)
    
cap.release()
out.release()
cv2.destroyAllWindows()