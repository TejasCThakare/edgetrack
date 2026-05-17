"""Export YOLOv8n → ONNX FP32 → ONNX INT8 (dynamic quantization)."""
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

print("Exporting YOLOv8n to ONNX FP32...")
YOLO("yolov8n.pt").export(format="onnx", imgsz=640, simplify=True, opset=12)

print("Applying dynamic INT8 quantization...")
quantize_dynamic(
    model_input="yolov8n.onnx",
    model_output="yolov8n_int8.onnx",
    weight_type=QuantType.QInt8,
)

print("Done.")
print("  yolov8n.onnx       (FP32)")
print("  yolov8n_int8.onnx  (INT8 dynamic)")