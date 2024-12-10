# YOLOv8 Object Detection

This project demonstrates how to use the YOLOv8 model for object detection on images, videos, and live webcam feeds using OpenCV.

## Requirements

- Python 3.6+
- OpenCV
- Ultralytics YOLO
- Yolo (V8x model - download from official yolo docs)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/aayush-ojha/yolov8-object-detection.git
    cd yolov8-object-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Script

You can run the script with different sources:

1. **Webcam Feed**:
    Uncomment the following line in `main.py`:
    ```python
    process_source(0, model)
    ```

2. **Image File**:
    Uncomment and modify the following line in `main.py`:
    ```python
    process_source("path/to/image.jpg", model, "output_image.jpg")
    ```

3. **Video File**:
    Uncomment and modify the following line in `main.py`:
    ```python
    process_source("path/to/video.mp4", model, "output_video.mp4")
    ```

### Example

To run object detection on a video file and save the output:
```python
process_source("video.mp4", model, "output_video.mp4")