"""Validate YOLOv8s detector on COCO128 — produces mAP numbers for README."""
from ultralytics import YOLO

DET_WEIGHTS = "yolov8s.pt"

print(f"Loading {DET_WEIGHTS}...")
model = YOLO(DET_WEIGHTS)

print("Running validation on COCO128...")
metrics = model.val(data="coco128.yaml", imgsz=640, verbose=True)

print("\n" + "=" * 40)
print("DETECTION ACCURACY (COCO128 val)")
print("=" * 40)
print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print("=" * 40)