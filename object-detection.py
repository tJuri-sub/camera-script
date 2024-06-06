# import utility function for loading Roboflow models
from inference import get_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to help load our image
import cv2
import time
import threading

# webcam
cap = cv2.VideoCapture(0)

# load a pre-trained yolov8n model
model = get_model(model_id="object-detection-mobile-phone/2")


# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# set the initial time
prev_time = time.time()
fps = 0
frame_count = 0

import threading

def inference_thread():
    while True:
        ret, frame = cap.read()
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        # Run inference
        results = model.infer(frame)
        # Process results (annotate frame, calculate FPS, etc.)
        ...

# Start inference thread
inference_thread = threading.Thread(target=inference_thread)
inference_thread.daemon = True
inference_thread.start()

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # resize frame for faster inference
    frame = cv2.resize(frame, (640, 480))

    # run inference on the captured frame
    results = model.infer(frame)

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

    # annotate the frame with our inference results
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    

    # display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()
