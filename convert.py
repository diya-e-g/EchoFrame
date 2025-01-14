
import tensorflow as tf
# Convert the SavedModel to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model("./ssd_mobilenet_v2_saved_model")
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("ssd_mobilenet_v2.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as 'ssd_mobilenet_v2.tflite'")
