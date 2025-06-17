from ultralytics import YOLO

# Load YOLOv5n model (nano)
model = YOLO('yolov5n.pt')  # pretrained weights

# Train the model
model.train(
    data=str(data_yaml_path),  # path to data.yaml
    epochs=80,
    imgsz=640,
    batch=16,
    name='yolov5n-custom'
    verbose=True, show=True
)
