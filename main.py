from ultralytics import YOLO
import cv2
import os
import numpy as np

def load_model(model_path='yolov8x.pt'):
    try:
        return YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def get_video_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open source {source}")
    return cap

def process_frame(model, frame):
    results = model(frame)[0]
    annotated_frame = frame.copy()
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{results.names[cls]} {conf:.2f}'
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame

def save_image(frame, output_path='output.jpg'):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, frame)

def get_video_writer(cap, output_path='output.mp4'):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def frame_generator(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def process_source(source, model, output_path=None):

    if isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg')):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        frame = cv2.imread(source)
        processed_frame = process_frame(model, frame)
        cv2.imshow('YOLOv8 Detection', processed_frame)
        if output_path:
            save_image(processed_frame, output_path)
        cv2.waitKey(0)
        return


    cap = get_video_capture(source)
    if output_path and not isinstance(source, int):  
        writer = get_video_writer(cap, output_path)
    
    try:
        for frame in frame_generator(cap):
            processed_frame = process_frame(model, frame)
            cv2.imshow('YOLOv8 Detection', processed_frame)
            if output_path and not isinstance(source, int):
                writer.write(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if output_path and not isinstance(source, int):
            writer.release()
        cv2.destroyAllWindows()

def main():
    model = load_model()
    
    # For webcam
    # process_source(0, model)
    
    # For image file with save
    # process_source("image.jpg", model, "output_image.jpg")
    
    # For video file with save
    process_source("video.mp4", model, "output_video.mp4")

if __name__ == "__main__":
    main()

