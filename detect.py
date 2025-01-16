import tensorflow as tf
import numpy as np

# Load the TFLite model
tflite_model_path = "model.tflite"  # Update with your model path
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print output details for debugging
print("Output Details:")
for output in output_details:
    print(output)

# Load the label map file
def load_labels(label_path):
    """Load label map from a text file."""
    with open(label_path, 'r') as f:
        return {int(line.split(':')[0]): line.split(':')[1].strip().replace('"', '') for line in f.readlines()}

label_map_path = "label.txt"  # Update with your label map file path
label_map = load_labels(label_map_path)

# Load and preprocess the image
image_path = "sample3.jpg"  # Replace with your image path
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3)

# Get the expected input shape of the model
input_shape = input_details[0]['shape']  # Example: [1, 300, 300, 3]
input_height, input_width = input_shape[1], input_shape[2]

# Resize and normalize the image
image = tf.image.resize(image, [input_height, input_width])
input_image = tf.expand_dims(tf.cast(image, tf.uint8), axis=0)  # Add batch dimension
interpreter.set_tensor(input_details[0]['index'], input_image)

# Perform inference
interpreter.invoke()

# Extract detection results
num_detections = int(interpreter.get_tensor(output_details[3]['index']).item())  # Number of detections
detection_boxes = interpreter.get_tensor(output_details[0]['index'])  # Bounding boxes
detection_classes = interpreter.get_tensor(output_details[1]['index'])  # Class IDs
detection_scores = interpreter.get_tensor(output_details[2]['index'])  # Confidence scores

# Post-process and print results
confidence_threshold = 0.5
print(f"Number of detections: {num_detections}")
for i in range(num_detections):
    score = detection_scores[0][i]
    if score > confidence_threshold:
        class_id = int(detection_classes[0][i])
        object_name = label_map.get(class_id, "Unknown")
        ymin, xmin, ymax, xmax = detection_boxes[0][i]
        print(f"Detection {i + 1}:")
        print(f"  Name: {object_name}")
        print(f"  Class ID: {class_id}")
        print(f"  Score: {score:.2f}")
        print(f"  Box: [{ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f}]")

# Optional: Visualize detections
import cv2
import matplotlib.pyplot as plt

# Convert tensor to a NumPy array for visualization
image_np = input_image[0].numpy()
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

for i in range(num_detections):
    score = detection_scores[0][i]
    if score > confidence_threshold:
        class_id = int(detection_classes[0][i])
        object_name = label_map.get(class_id, "Unknown")
        ymin, xmin, ymax, xmax = detection_boxes[0][i]
        (start_x, start_y) = (int(xmin * input_width), int(ymin * input_height))
        (end_x, end_y) = (int(xmax * input_width), int(ymax * input_height))
        cv2.rectangle(image_np, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(image_np, f"{object_name} ({score:.2f})",
                    (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Show the image
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
