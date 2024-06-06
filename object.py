import cv2
from roboflow import Roboflow

# Initialize Roboflow
import inference
model = inference.get_model("object-detection-ynoia/1")
model.infer(image="YOUR_IMAGE.jpg")


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform inference on the frame
    predictions = model.predict(frame, confidence=40, overlap=30).json()

    # Draw bounding boxes around detected objects
    for prediction in predictions:
        label = prediction["label"]
        confidence = prediction["confidence"]
        box = prediction["box"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Real-time Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
