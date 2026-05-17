"""Export YOLOv8s → ONNX FP32 → ONNX INT8 (dynamic quantization)."""
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

DET_WEIGHTS = "yolov8s.pt"
ONNX_FP32 = "yolov8s.onnx"
ONNX_INT8 = "yolov8s_int8.onnx"

print(f"Exporting {DET_WEIGHTS} to ONNX FP32...")
YOLO(DET_WEIGHTS).export(format="onnx", imgsz=640, simplify=True, opset=12)

print("Applying dynamic INT8 quantization...")
quantize_dynamic(
    model_input=ONNX_FP32,
    model_output=ONNX_INT8,
    weight_type=QuantType.QInt8,
)

print("Done.")
print(f"  {ONNX_FP32}       (FP32)")
print(f"  {ONNX_INT8}  (INT8 dynamic)")