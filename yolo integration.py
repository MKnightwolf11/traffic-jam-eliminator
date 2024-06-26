import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load YOLOv4
net = cv2.dnn.readNet('C:\\Users\\Rachit\\OneDrive\\Desktop\\Git in One Video\\YOLO\\yolov4.weights', 'C:\\Users\\Rachit\\OneDrive\\Desktop\\Git in One Video\\YOLO\\yolov4.cfg')
classes = []
with open('C:\\Users\\Rachit\\OneDrive\\Desktop\\Git in One Video\\YOLO\\coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Function to perform object association using the Hungarian algorithm
def object_association(detected_objects, predicted_objects):
    num_detected = len(detected_objects)
    num_predicted = len(predicted_objects)

    cost_matrix = np.zeros((num_detected, num_predicted), dtype=np.float32)

    for i in range(num_detected):
        for j in range(num_predicted):
            # Replace this with your logic to calculate the cost based on IoU or other criteria
            cost_matrix[i, j] = 1.0

    # Apply the Hungarian algorithm to find the optimal assignment
    _, assignment = linear_sum_assignment(cost_matrix)

    return assignment

# Function to perform Kalman filtering for object tracking
class KalmanFilter:
    def __init__(self, initial_state):
        # Initialize Kalman filter parameters
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
        
        # Initialize state using the detected bounding box
        x, y, w, h = initial_state
        self.kalman.statePost = np.array([x + w / 2, y + h / 2, 0, 0], dtype=np.float32).reshape(-1, 1)

    def predict(self):
        # Predict the next state
        self.kalman.predict()

    def correct(self, measurement):
        # Update the state based on the measurement
        self.kalman.correct(measurement)

    def get_state(self):
        # Get the current estimated state
        return self.kalman.statePost[:2].flatten()

# Video capture
cap = cv2.VideoCapture('C:\\Users\\Rachit\\OneDrive\\Desktop\\Git in One Video\\YOLO\\road.mp4')

# Initialize variables
predicted_objects = []
kalman_filters = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLOv4
    # (Replace this with your YOLOv4 detection logic)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layer_names)

    # Process YOLOv4 detections and extract bounding boxes
    detected_objects = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x, center_y, w, h = (obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x, y = int(center_x - w/2), int(center_y - h/2)
                detected_objects.append((x, y, x + w, y + h, class_id))

    # Perform object association using the Hungarian algorithm
    assignment = object_association(detected_objects, predicted_objects)

    # Update predicted_objects with the associated detected_objects
    for i, j in enumerate(assignment):
        predicted_objects[j] = detected_objects[i]

    # Update Kalman filters based on new measurements
    for i, j in enumerate(assignment):
        x, y, _, _ = detected_objects[i]
        if j >= len(kalman_filters):
            # Create a new Kalman filter if needed
            kalman_filters.append(KalmanFilter((x, y, 0, 0)))
        else:
            # Update existing Kalman filter with new measurement
            kalman_filters[j].correct(np.array([x, y]).reshape(-1, 1))

    # Draw bounding boxes on the frame
    for x1, y1, x2, y2, class_id in predicted_objects:
        color = (0, 255, 0)  # Green color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('Object Tracking with YOLOv4, Hungarian, and Kalman', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()