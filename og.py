import tensorflow as tf
import numpy as np

# Load the original TensorFlow SavedModel
saved_model_path = "ssd_mobilenet_v2_saved_model"  # Update with the correct path
model = tf.saved_model.load(saved_model_path)

# Load and preprocess the image
image_path = "sample1.jpg"  # Replace with your image path
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)
image = tf.image.resize(image, [512, 512])

# Convert the image to uint8 and add a batch dimension
input_image = tf.expand_dims(tf.cast(image, tf.uint8), axis=0)  # Change to uint8

# Perform inference
inference_function = model.signatures["serving_default"]
results = inference_function(input_image)

# Extract detection results
detection_boxes = results["detection_boxes"].numpy()
detection_scores = results["detection_scores"].numpy()
detection_classes = results["detection_classes"].numpy()

# Load COCO class labels
label_file = "label.txt"  # Ensure this has the correct COCO labels
class_labels = {}
with open(label_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        class_labels[int(parts[0])] = parts[1]

# Process the detections
threshold = 0.7  # Confidence threshold
print("Detected Objects:")
for i in range(len(detection_scores[0])):
    score = detection_scores[0][i]
    if score > threshold:
        class_id = int(detection_classes[0][i])
        label = class_labels.get(class_id, f"Unknown ({class_id})")
        print(f"- {label} (Confidence: {score:.2f})")