# Real-time Object Detection using TensorFlow Lite and OpenCV

This project demonstrates how to use a TensorFlow Lite model for real-time object detection with a webcam. The results are displayed using Matplotlib.

## Prerequisites

- Python 3.6 or later
- TensorFlow
- OpenCV
- Matplotlib
- A TFLite model file (e.g., `model.tflite`)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv yolov5-env
    yolov5-env\Scripts\activate  # On Windows
    source yolov5-env/bin/activate  # On macOS/Linux
    ```

3. **Install required packages**:

    ```bash
    pip install tensorflow opencv-python matplotlib numpy
    ```

## Running the Script

1. **Ensure you have a TFLite model file** in the project directory (e.g., `model.tflite`).

2. **Ensure you have a camera connected** to your system.

3. **Run the script**:

    ```bash
    python test_tflite_model.py
    ```

## Script Overview

The script captures video frames from the webcam, preprocesses them, runs them through the TensorFlow Lite model, and displays the results in real-time using Matplotlib.

- **`preprocess_image`**: Converts the image from BGR to RGB, resizes it, normalizes it, and adds a batch dimension.
- **`process_output`**: Extracts bounding boxes, class IDs, and scores from the model's output.
- **Main Loop**: Captures video frames, preprocesses them, runs inference, and displays the results.

## Key Files

- `test_tflite_model.py`: The main script for real-time object detection.
- `model.tflite`: Your TensorFlow Lite model file (ensure this file is in the project directory).

## Example Output

The script opens a window displaying the real-time detection results from your camera. Detected objects are highlighted with bounding boxes and class labels.

## License

This project is licensed under the MIT License.
