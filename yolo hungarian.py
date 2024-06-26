import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Function to calculate IoU (Intersection over Union) between two bounding boxes
def bbox_iou(box1, box2):
    x1_tl, y1_tl, w1, h1 = box1
    x2_tl, y2_tl, w2, h2 = box2

    x1_br, y1_br = x1_tl + w1, y1_tl + h1
    x2_br, y2_br = x2_tl + w2, y2_tl + h2

    # Intersection area
    x_tl = max(x1_tl, x2_tl)
    y_tl = max(y1_tl, y2_tl)
    x_br = min(x1_br, x2_br)
    y_br = min(y1_br, y2_br)

    if x_br < x_tl or y_br < y_tl:
        return 0.0

    intersection_area = (x_br - x_tl) * (y_br - y_tl)

    # Union Area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area

# Function to perform object tracking using Hungarian algorithm
def object_tracking(prev_boxes, current_boxes):
    cost_matrix = np.zeros((len(prev_boxes), len(current_boxes)), dtype=np.float32)
    for i, prev_box in enumerate(prev_boxes):
        for j, current_box in enumerate(current_boxes):
            cost_matrix[i, j] = 1 - bbox_iou(prev_box, current_box)

    prev_indices, current_indices = linear_sum_assignment(cost_matrix)
    matches = {}
    for prev_idx, current_idx in zip(prev_indices, current_indices):
        matches[prev_idx] = current_idx

    unmatched_prev_boxes = set(range(len(prev_boxes))) - set(matches.keys())
    unmatched_current_boxes = set(range(len(current_boxes))) - set(matches.values())

    return matches, list(unmatched_prev_boxes), list(unmatched_current_boxes)

# Load YOLOv4
net = cv2.dnn.readNet('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\yolov4.weights', 'E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\yolov4.cfg')
classes = []
with open('E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load video or image
input_path = 'E:\\PSIT\\4th year\\7th semester\\project\\Beginner level working\\YOLO\\road.mp4'  # Replace with your input file path
cap = cv2.VideoCapture(input_path)

prev_boxes = []

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

    current_boxes = []

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

                current_boxes.append((x, y, w, h))

                # Draw bounding box and label on the frame
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    matches, unmatched_prev, unmatched_current = object_tracking(prev_boxes, current_boxes)

    for idx, current_idx in matches.items():
        prev_box = prev_boxes[idx]
        current_box = current_boxes[current_idx]
        # Draw lines between matched boxes
        cv2.line(frame, (prev_box[0] + prev_box[2] // 2, prev_box[1] + prev_box[3] // 2),
                 (current_box[0] + current_box[2] // 2, current_box[1] + current_box[3] // 2), color, 2)

    prev_boxes = current_boxes

    # Display the resulting frame
    cv2.imshow('YOLOv4 Detection with Object Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()