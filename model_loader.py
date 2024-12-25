import tensorflow as tf
import os

def load_model(model_dir="D:\quad-squad\code"):
    # Ensure the directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return None

    # Find the .tflite model file in the directory
    model_path = None
    for file in os.listdir(model_dir):
        if file.endswith(".tflite"):
            model_path = os.path.join(model_dir, file)
            break

    if model_path is None:
        print(f"No .tflite file found in directory: {model_dir}")
        return None

    # Load the TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
        return interpreter
    except ValueError as e:
        print(f"Error loading model: {e}")
        return None
