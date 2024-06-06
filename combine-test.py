import mediapipe as mp
import cv2
from inference import get_model
import supervision as sv
import time

# Roboflow Object Detection setup
model = get_model(model_id="object-detection-ynoia/1")  

# Mediapipe Holistic setup
mp_holistic = mp.solutions.holistic

# Webcam setup
cap = cv2.VideoCapture(0)

prev_time = time.time()
fps = 0
frame_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (600, 500))

        # Object Detection
        results_obj = model.infer(frame)
        detections = sv.Detections.from_inference(results_obj[0].dict(by_alias=True, exclude_none=True))

        # Print confidence level of detected objects to logs
        for detection in detections.objects:
            print(f"Detected: {detection.label} - Confidence: {detection.confidence:.2f}")

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
