from ultralytics import YOLO
model = YOLO("/root/taojianwei/projects/ultralytics/runs/detect/runs/train/exp-25/weights/best.pt")
model.export(format="onnx")