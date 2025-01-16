import tensorflow as tf
import numpy as np

# Load the TFLite model
tflite_model_path = "model.tflite"  # Update with your model path
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the expected input shape of the model
input_shape = input_details[0]['shape']  # Example: [1, 300, 300, 3]
input_height, input_width = input_shape[1], input_shape[2]

# Load and preprocess the image
image_path = "sample1.jpg"  # Replace with your image path
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)

# Resize and normalize the image
image = tf.image.resize(image, [input_height, input_width])
input_image = tf.expand_dims(tf.cast(image, tf.uint8), axis=0)  # Add batch dimension

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)

# Perform inference
interpreter.invoke()

# Extract detection results
detection_boxes = interpreter.get_tensor(output_details[0]['index'])  # [1, num_detections, 4]
detection_classes = interpreter.get_tensor(output_details[1]['index'])  # [1, num_detections]
detection_scores = interpreter.get_tensor(output_details[2]['index'])  # [1, num_detections]
num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Scalar

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
for i in range(num_detections):
    score = detection_scores[0][i]
    if score > threshold:
        class_id = int(detection_classes[0][i])
        label = class_labels.get(class_id, f"Unknown ({class_id})")
        ymin, xmin, ymax, xmax = detection_boxes[0][i]
        print(f"- {label} (Confidence: {score:.2f}), Box: [ymin={ymin:.2f}, xmin={xmin:.2f}, ymax={ymax:.2f}, xmax={xmax:.2f}]")
