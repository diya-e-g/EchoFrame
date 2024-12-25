from model_loader import load_model
from image_processor import preprocess_image
from inference import run_inference
from description_generator import generate_description
from audio_feedback import text_to_speech

# Use the pre-downloaded model path
model_dir = r"D:\quad-squad\code"
interpreter = load_model(model_dir)
if interpreter is None:
    print("Failed to load the model. Exiting...")
    exit()

# Continue with the workflow...
image_path = "image1.jpg"
input_tensor = preprocess_image(image_path)
output_data = run_inference(interpreter, input_tensor)
detected_objects = ["chair", "table","globe"]  # Replace with actual detection logic
description = generate_description(detected_objects)
print("Generated Description:", description)
text_to_speech(description)
