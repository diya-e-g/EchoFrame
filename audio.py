import pyttsx3
import subprocess

def run_model():
    """Run the test_original_model.py script and capture its output."""
    result = subprocess.run(
        ['python', 'og.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        # Handle errors
        error_message = f"Error running test_original_model.py: {result.stderr}"
        print(error_message)
        return error_message
    return result.stdout

def convert_to_speech(message):
    """Convert text to speech using pyttsx3."""
    text_speech = pyttsx3.init()
    text_speech.say(message)
    text_speech.runAndWait()

if __name__ == "__main__":
    # Run the model script and get its output
    model_output = run_model()
    print("Model Output:")
    print(model_output)

    # Convert the model output to speech
    print("Converting to speech...")
    convert_to_speech(model_output)