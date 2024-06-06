import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from mediapipe.python.solutions.holistic import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp.solutions

custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_holistic.POSE_CONNECTIONS)

# list of landmarks in pose landmarks to exclude from drawing
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
    PoseLandmark.RIGHT_THUMB,]

for landmark in PoseLandmark:
    if landmark not in excluded_landmarks:
        custom_style[landmark] = DrawingSpec(color=(0, 0, 255), thickness=2)  # Set color to blue (BGR format)
    else:
        custom_style[landmark] = DrawingSpec(color=(0, 0, 255), thickness=None)  # Exclude landmarks from drawing
    
for landmark in excluded_landmarks:
    # Change the way the excluded landmarks are drawn
    custom_style[landmark] = DrawingSpec(color=(0, 0, 255), thickness=None)

    # Remove all connections which contain these landmarks
    custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple]


custom_connections = list(mp_holistic.POSE_CONNECTIONS)

connection_color = (0, 0, 255)
       

# webcam
cap = cv2.VideoCapture(0)


#Initiate holistic model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5) as holistic:


    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        #print(results.face_landmarks)

        # pose_landmarks,  left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # 2. Right Hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(61,177,254), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,153,254), thickness=3)
                                  )

        # 3. Left Hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(61,177,254), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,153,254), thickness=3)
                                  )

        # 4. Pose Detection landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, landmark_drawing_spec=custom_style
                                  )

        # Draw connections in pose landmarks
        for connection in custom_connections:
            start = connection[0]
            end = connection[1]
            if start not in excluded_landmarks and end not in excluded_landmarks:
                cv2.line(image, (int(results.pose_landmarks.landmark[start].x * image.shape[1]), int(results.pose_landmarks.landmark[start].y * image.shape[0])),
                        (int(results.pose_landmarks.landmark[end].x * image.shape[1]), int(results.pose_landmarks.landmark[end].y * image.shape[0])),
                        connection_color, 3)

            
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) and 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()