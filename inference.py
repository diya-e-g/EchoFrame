import numpy as np
def run_inference(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    
    # Ensure the input tensor matches the expected shape
    input_tensor = np.resize(input_tensor, input_details[0]['shape'])
    
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data