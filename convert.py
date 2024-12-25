import tensorflow as tf

# Path to the SavedModel directory
saved_model_dir = "C:\\Users\\user\\.cache\\kagglehub\\models\\tensorflow\\ssd-mobilenet-v2\\tensorFlow2\\fpnlite-320x320\\1"

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TFLite model
with open("ssd_mobilenet_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format!")
