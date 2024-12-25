import numpy as np
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((1, 1))  # Resize to match expected input shape
    image = np.array(image, dtype=np.uint8)  # Ensure the image is of type UINT8
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image