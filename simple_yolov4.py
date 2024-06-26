import torch
import cv2
import numpy as np

# Load YOLOv4
net = cv2.dnn.readNet('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\yolov4.weights', 'E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\yolov4.cfg')
classes = []
with open('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load video or image
input_path = 'E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\road.mp4'  # Replace with your input file path
cap = cv2.VideoCapture(input_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video is finished

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO network
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get the output
    detections = net.forward(output_layer_names)

    # Process the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Get the bounding box coordinates
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w/2), int(center_y - h/2)

                # Draw bounding box and label on the frame
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv4 Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()