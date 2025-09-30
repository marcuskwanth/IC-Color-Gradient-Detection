import cv2
import torch
import numpy as np
import math
import time

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0


# Load the YOLOv5 model
# Replace 'yolov5s.pt' with your trained/custom model path if applicable
model_path = '/home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Export/IC_AI_model_03_Student/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Set the webcam source (0 for the default camera, or specify a video file path)
webcam_source = 0  # Change to 1, 2, etc., if multiple cameras are connected

# Open the webcam
cap = cv2.VideoCapture(webcam_source)

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame to RGB (YOLOv5 expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(frame_rgb)

    # Parse detection results and plot them directly on the frame
    detections = results.pandas().xyxy[0]  # Get the detections as a Pandas DataFrame
    for _, row in detections.iterrows():
        # Extract bounding box coordinates and other info
        x1, y1, x2, y2, conf, cls, name = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
            row["confidence"],
            int(row["class"]),
            row["name"],
        )
        if(0.5<conf):
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Put label and confidence score
            label = f"{name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the frame with detections
    cv2.imshow('YOLOv5 Webcam', frame)

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(frame_rgb, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    print(fps)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting webcam...")
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()