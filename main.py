from model_loader import load_model
from image_processor import preprocess_image
from inference import run_inference
from label_utils import load_labels, get_label_from_class_id
from description_generator import generate_description
from audio_feedback import text_to_speech

# Paths to required files
model_dir = r"D:\quad-squad\code"
label_path = r"D:\quad-squad\code\label_map.txt"
image_path = "image.jpg"

# Step 1: Load the model
interpreter = load_model(model_dir)
if interpreter is None:
    print("Failed to load the model. Exiting...")
    exit()

# Step 2: Load labels
labels = load_labels(label_path)
if not labels:
    print("Failed to load labels. Exiting...")
    exit()

# Step 3: Preprocess the image
input_tensor = preprocess_image(image_path)
if input_tensor is None:
    print("Failed to preprocess the image. Exiting...")
    exit()

# Step 4: Run inference
boxes, predicted_classes, confidence_scores = run_inference(interpreter, input_tensor)

# Step 5: Filter high-confidence detections
confidence_threshold = 0.5
detected_objects = []

for i, score in enumerate(confidence_scores):
    if score > confidence_threshold:
        class_id = int(predicted_classes[i])
        label = get_label_from_class_id(class_id, labels)
        print(f"Detected {label} with confidence {score:.2f}")
        detected_objects.append(label)

# Step 6: Generate description and provide audio feedback
description = generate_description(detected_objects)
print("Generated Description:", description)
text_to_speech(description)
