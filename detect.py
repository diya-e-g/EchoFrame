import tensorflow as tf
import numpy as np

# Load and preprocess the image
image_path = "sample3.jpg"  # Replace with your image path
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
detection_boxes = interpreter.get_tensor(output_details[0]['index'])  # [1, 1917, 4]
detection_classes = interpreter.get_tensor(output_details[1]['index'])  # [1, 100]
detection_scores = interpreter.get_tensor(output_details[4]['index'])  # [1, 100, 4]

# Class labels
class_labels = {
    1: "person",
    2: "car",
    3: "bicycle",
    # Add more class mappings if available
}

# Process the detections
threshold = 0.5  # Confidence threshold
print("Detected Objects:")
for i in range(num_detections):
    # Extract detection info
    scores_for_detection = detection_scores[0][i]  # Shape: [4]
    max_score_index = np.argmax(scores_for_detection)  # Index of the highest score
    score = float(scores_for_detection[max_score_index])  # Confidence for the detected class
    class_id = max_score_index  # Class ID from the highest score
    box = detection_boxes[0][i]  # [ymin, xmin, ymax, xmax]

    # Validate and filter detections
    if score > threshold and all(0 <= coord <= 1 for coord in box):
        label = class_labels.get(class_id, f"Unknown ({class_id})")
        ymin, xmin, ymax, xmax = box  # Normalized coordinates
        print(f"Class: {label}, Confidence: {score:.2f}, Box: [ymin={ymin:.2f}, xmin={xmin:.2f}, ymax={ymax:.2f}, xmax={xmax:.2f}]")
