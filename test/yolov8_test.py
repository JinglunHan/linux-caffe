from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model
image_path = '/home/roota/workstation/onnx2caffe/linux-caffe/test/test_data/bus.jpg'
# Predict with the model
results = model(image_path) 

model.predict(image_path, save=True, imgsz=320, conf=0.5)