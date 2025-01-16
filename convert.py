import tensorflow as tf

# Path to your saved_model directory
saved_model_dir = "ssd_mobilenet_v2_saved_model"

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Optional: Add optimizations for smaller size or faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # For size optimization

# Convert the model
tflite_model = converter.convert()

# Save the .tflite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as 'model.tflite'")
