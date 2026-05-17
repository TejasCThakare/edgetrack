"""Validate YOLOv8n detector on COCO128 — produces mAP numbers for README."""
from ultralytics import YOLO

print("Loading YOLOv8n...")
model = YOLO("yolov8n.pt")

print("Running validation on COCO128...")
metrics = model.val(data="coco128.yaml", imgsz=640, verbose=True)

print("\n" + "=" * 40)
print("DETECTION ACCURACY (COCO128 val)")
print("=" * 40)
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print("=" * 40)