import tensorflow as tf
import numpy as np
import cv2

# Load and preprocess the image
image_path = "image.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Resize image to match input shape requirement (300x300 for SSD models)
input_image = cv2.resize(image, (300, 300))
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = np.expand_dims(input_image, axis=0).astype(np.uint8)  # Add batch dimension

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v2.tflite")  # Replace with your model path
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Resize tensor input to fixed shape if necessary
interpreter.resize_tensor_input(input_details[0]['index'], [1, 300, 300, 3])
interpreter.allocate_tensors()

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# Extract outputs
raw_boxes = interpreter.get_tensor(output_details[0]['index'])  # [1, 1917, 4]
raw_scores = interpreter.get_tensor(output_details[1]['index'])  # [1, 1917, 91]
detection_scores = np.array(interpreter.get_tensor(output_details[4]['index'])).squeeze()  # Ensure array format
detection_classes = np.array(interpreter.get_tensor(output_details[2]['index'])).squeeze()  # Ensure array format

# Handle potential edge cases for empty or incorrect output shapes
if detection_scores.ndim == 0:
    detection_scores = np.expand_dims(detection_scores, axis=0)
if detection_classes.ndim == 0:
    detection_classes = np.expand_dims(detection_classes, axis=0)

# Process detections (set a threshold for detection)
detection_threshold = 0.3

print("Detected Objects:")
for score, class_id in zip(np.ravel(detection_scores), np.ravel(detection_classes)):
    if float(score) > detection_threshold:
        print(f"Class ID: {int(class_id)}, Confidence: {score:.2f}")

