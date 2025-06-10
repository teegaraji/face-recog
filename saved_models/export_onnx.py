from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load the YOLOv8 model
model.export(format="onnx", imgsz=640)
