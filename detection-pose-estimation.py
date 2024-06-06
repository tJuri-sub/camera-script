import mediapipe as mp
import cv2
from inference import get_model
import supervision as sv
import time

# Roboflow Object Detection setup
model = get_model(model_id="object-detection-ynoia/1")  
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()



# Mediapipe Holistic setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from mediapipe.python.solutions.holistic import PoseLandmark
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_holistic.POSE_CONNECTIONS)

excluded_landmarks = [
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.RIGHT_THUMB,
]

for landmark in PoseLandmark:
    if landmark not in excluded_landmarks:
        custom_style[landmark] = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    else:
        custom_style[landmark] = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=None)

for landmark in excluded_landmarks:
    custom_style[landmark] = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=None)
    custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple]

connection_color = (0, 0, 255)

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
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Mediapipe Holistic
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(61, 177, 254), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 153, 254), thickness=3))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(61, 177, 254), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 153, 254), thickness=3))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, landmark_drawing_spec=custom_style)

        for connection in custom_connections:
            start = connection[0]
            end = connection[1]
            if start not in excluded_landmarks and end not in excluded_landmarks:
                cv2.line(image, (int(results.pose_landmarks.landmark[start].x * image.shape[1]), int(results.pose_landmarks.landmark[start].y * image.shape[0])),
                        (int(results.pose_landmarks.landmark[end].x * image.shape[1]), int(results.pose_landmarks.landmark[end].y * image.shape[0])),
                        connection_color, 3)

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0

        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Combined Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
