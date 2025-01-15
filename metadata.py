import tflite_support.metadata as metadata

# Path to your model
model_path = "ssd_mobilenet_v2.tflite"

# Load metadata
displayer = metadata.MetadataDisplayer.with_model_file(model_path)
print("Associated file(s):", displayer.get_packed_associated_file_list())
print("Metadata JSON:")
print(displayer.get_metadata_json())
