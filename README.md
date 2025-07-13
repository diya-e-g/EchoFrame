# Object Detection with Audio Feedback System

## 🎯 Project Overview

This project implements a real-time object detection system designed for deployment on **Xiaesp32s3** microcontrollers. The system captures live video, performs object detection using a **MobileNet SSD** model, and provides audio feedback for detected objects - making it particularly useful for accessibility applications.

While the complete ESP32 deployment wasn't fully implemented, this repository contains all the core components for object detection with TensorFlow Lite optimization and audio feedback generation.

## 🏗️ Architecture

```
Live Video Input → Object Detection (TFLite) → Audio Feedback Output
```

## 📁 Key Files

### Core Implementation
- **[`main.py`](main.py)** - Main orchestration script that integrates all components
- **[`detect.py`](detect.py)** - Core object detection logic using TensorFlow Lite
- **[`model_loader.py`](model_loader.py)** - Model loading and initialization utilities
- **[`audio.py`](audio.py)** - Text-to-speech audio feedback system

### Model Optimization
- **[`og.py`](og.py)** - Original TensorFlow SavedModel implementation
- **[`convert.py`](convert.py)** - Model conversion from SavedModel to TensorFlow Lite
- **[`download.py`](download.py)** - Model acquisition from TensorFlow Hub

### Models & Labels
- **[`model.tflite`](model.tflite)** - Optimized TensorFlow Lite model for inference
- **[`model.h`](model.h)** - C header file for ESP32 deployment
- **[`label.txt`](label.txt)** - Object class labels mapping
- **[`labels_mobilenet_quant_v1_224.txt`](labels_mobilenet_quant_v1_224.txt)** - MobileNet label mappings

## 🚀 Features

- **Real-time Object Detection**: Uses MobileNet SSD v2 for efficient inference
- **TensorFlow Lite Optimization**: Model optimized for embedded systems
- **Audio Feedback**: Text-to-speech conversion of detection results
- **Accessibility Focus**: Designed for visually impaired users
- **ESP32 Ready**: Code prepared for microcontroller deployment

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RohitAnish1/quad_squad.git
   cd quad_squad
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Required packages include**:
   - TensorFlow/TensorFlow Lite
   - OpenCV
   - NumPy
   - pyttsx3 (for text-to-speech)
   - matplotlib (for visualization)

## 💻 Usage

### Basic Object Detection

1. **Run the main pipeline**:
   ```bash
   python main.py
   ```

2. **Test with individual components**:
   ```bash
   # Original TensorFlow model
   python og.py
   
   # TensorFlow Lite optimized version
   python detect.py
   
   # Audio feedback only
   python audio.py
   ```

### Model Conversion

Convert SavedModel to TensorFlow Lite:
```bash
python convert.py
```

### Sample Images

The project includes sample images in the `images/` directory for testing:
- `car.jpg`, `car1.jpg` - Vehicle detection
- `sample.jpg`, `sample1.jpg`, `sample2.jpg`, `sample3.jpg` - Various objects

## 🔧 Configuration

### Model Paths
Update paths in `main.py` and other files:
```python
model_dir = r"path/to/your/model"
label_path = r"path/to/your/labels.txt"
image_path = "path/to/test/image.jpg"
```

### Confidence Threshold
Adjust detection sensitivity in `main.py`:
```python
confidence_threshold = 0.5  # Adjust as needed
```

## 🎛️ Components Breakdown

### 1. Model Loading (`model_loader.py`)
- Discovers and loads TensorFlow Lite models
- Handles model initialization and tensor allocation
- Provides error handling for model files

### 2. Object Detection (`detect.py`)
- Core inference logic using TensorFlow Lite interpreter
- Image preprocessing and postprocessing
- Bounding box and confidence score extraction
- Visualization with OpenCV

### 3. Audio Feedback (`audio.py`)
- Integrates pyttsx3 for text-to-speech conversion
- Processes detection results into natural language
- Provides audio output for accessibility

### 4. Main Pipeline (`main.py`)
- Orchestrates the complete workflow
- Integrates all components seamlessly
- Handles high-level logic and error management

## 🎯 Target Hardware

**Xiaesp32s3 Microcontroller**
- ARM Cortex-M4 processor
- Limited memory and processing power
- Optimized TensorFlow Lite model for deployment
- Real-time inference capabilities

## 🔬 Technical Details

### Model Specifications
- **Base Model**: MobileNet SSD v2
- **Input Size**: 224x224 pixels
- **Output**: Bounding boxes, class predictions, confidence scores
- **Optimization**: TensorFlow Lite with DEFAULT optimizations

### Performance Considerations
- Model size optimized for embedded systems
- Quantization applied for faster inference
- Memory-efficient preprocessing pipeline

## 🚧 Current Status

### ✅ Completed
- Object detection pipeline with TensorFlow Lite
- Audio feedback system
- Model conversion and optimization
- Core inference logic

### 🔄 In Progress / Future Work
- Complete ESP32 deployment
- Live camera integration
- Real-time video processing
- Hardware optimization

## 📊 Sample Results

The system can detect various objects including:
- Vehicles (cars, trucks, motorcycles)
- People
- Animals
- Common objects (bottles, chairs, etc.)

Audio feedback provides descriptions like:
- "Car detected with 85% confidence"
- "Person detected with 92% confidence"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## 👥 Team

**Quad Squad Team**
- Focus on accessibility through computer vision
- Embedded systems and AI integration
- Real-time object detection solutions

## 📞 Contact

For questions or collaboration opportunities, please reach out through the GitHub repository.

---

*This project demonstrates the integration of computer vision, embedded systems, and accessibility technologies for real-world applications.*
