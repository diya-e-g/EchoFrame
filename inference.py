import numpy as np

def stable_softmax(logits):
    """Compute softmax with numerical stability."""
    exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

def run_inference(interpreter, input_tensor):
    """Run inference on the input tensor using the TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    input_index = input_details[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]  # Bounding boxes
    scores = interpreter.get_tensor(output_details[1]["index"])  # Logits for class probabilities
    
    # Convert logits to probabilities using stable softmax
    probabilities = stable_softmax(scores)
    
    # Get predicted classes and confidence scores
    predicted_classes = np.argmax(probabilities, axis=1)  # Class indices
    confidence_scores = np.max(probabilities, axis=1)  # Confidence scores
    
    return boxes, predicted_classes, confidence_scores
