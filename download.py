import tensorflow as tf
import tensorflow_hub as hub

# URL of the MobileNetV2 SSD model
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

# Load the model from TensorFlow Hub
model = hub.load(MODEL_URL)
print("Model loaded successfully!")

# Save the model locally as a SavedModel directory
model_save_path = "./ssd_mobilenet_v2_saved_model"
tf.saved_model.save(model, model_save_path)
print(f"Model saved locally at: {model_save_path}")
