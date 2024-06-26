import torch
import cv2
import numpy as np

# Load YOLOv4 model weights
model = torch.load('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\yolov4.weights')

# Load COCO class names
with open('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load video
cap = cv2.VideoCapture('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\road.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    # Preprocess frame (resize, normalize, etc.)
    # ...

    # Perform object detection
    with torch.no_grad():
        detections = model(frame)

    # Process detections
    # ...

    # Display the resulting frame
    cv2.imshow('YOLOv4 Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()