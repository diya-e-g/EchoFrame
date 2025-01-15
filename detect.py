import tensorflow as tf
import numpy as np

# Load and preprocess the image
image_path = "p1.jpg"  # Replace with your image path
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)
image = tf.image.resize(image, [512, 512])
input_image = tf.expand_dims(tf.cast(image, tf.uint8), axis=0)  # Add batch dimension

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dynamically resize the input tensor
interpreter.resize_tensor_input(input_details[0]['index'], [1, 512, 512, 3])
interpreter.allocate_tensors()

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# Extract relevant outputs
num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])  # Scalar
detection_boxes = interpreter.get_tensor(output_details[0]['index'])[0][:num_detections]  # Top num_detections boxes
detection_classes = interpreter.get_tensor(output_details[1]['index'])[0][:num_detections]  # Top num_detections classes
detection_scores_raw = interpreter.get_tensor(output_details[4]['index'])[0][:num_detections]  # [num_detections, num_classes]

# Extract the maximum confidence score for each bounding box
detection_scores = np.max(detection_scores_raw, axis=1)  # Shape: [num_detections]

# Load labels from label.txt
label_file = "label.txt"
class_labels = {}
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        class_labels[int(parts[0])] = parts[1]

# Apply Non-Max Suppression (NMS)
iou_threshold = 0.5  # Intersection-over-Union threshold for suppression
score_threshold = 0.7  # Minimum confidence score for detections

selected_indices = tf.image.non_max_suppression(
    detection_boxes,  # Top num_detections bounding boxes
    detection_scores,  # Corresponding scores
    max_output_size=10,  # Maximum number of boxes to keep
    iou_threshold=iou_threshold,
    score_threshold=score_threshold
)

selected_boxes = tf.gather(detection_boxes, selected_indices).numpy()
selected_classes = tf.gather(detection_classes, selected_indices).numpy()
selected_scores = tf.gather(detection_scores, selected_indices).numpy()

# Display filtered detections
print("Detected Objects:")
for i, box in enumerate(selected_boxes):
    label = class_labels.get(int(selected_classes[i]), f"Unknown ({int(selected_classes[i])})")
    ymin, xmin, ymax, xmax = box  # Bounding box coordinates
    print(f"- {label} (Confidence: {selected_scores[i]:.2f})")
