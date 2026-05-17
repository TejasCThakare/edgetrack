"""Benchmark inference throughput: PyTorch (GPU) vs ONNX FP32 (CPU) vs ONNX INT8 (CPU)."""
import time
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

VIDEO = "input.mp4"
N_FRAMES = 300
IMGSZ = 640
DET_WEIGHTS = "yolov8s.pt"
ONNX_FP32 = "yolov8s.onnx"
ONNX_INT8 = "yolov8s_int8.onnx"


def load_frames(path, n):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (IMGSZ, IMGSZ)))
    cap.release()
    return frames


def bench_pytorch(frames):
    m = YOLO(DET_WEIGHTS)
    for _ in range(5):  # warmup
        m(frames[0], verbose=False)
    t = time.time()
    for f in frames:
        m(f, verbose=False)
    return len(frames) / (time.time() - t)


def bench_onnx(frames, path, providers):
    sess = ort.InferenceSession(path, providers=providers)
    iname = sess.get_inputs()[0].name
    for _ in range(5):  # warmup
        x = frames[0].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        sess.run(None, {iname: x})
    t = time.time()
    for f in frames:
        x = f.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        sess.run(None, {iname: x})
    return len(frames) / (time.time() - t)


print(f"Loading {N_FRAMES} frames from {VIDEO}...")
frames = load_frames(VIDEO, N_FRAMES)
print(f"Got {len(frames)} frames.\n")

print("Available ONNX providers:", ort.get_available_providers())
gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu_available else ["CPUExecutionProvider"]
onnx_hardware = "GPU" if gpu_available else "CPU"
print(f"Using {onnx_hardware} for ONNX runs.\n")

print("Benchmarking PyTorch FP32 (GPU)...")
fps_torch = bench_pytorch(frames)

print(f"Benchmarking ONNX FP32 ({onnx_hardware})...")
fps_onnx_fp32 = bench_onnx(frames, ONNX_FP32, onnx_providers)

print("Benchmarking ONNX INT8 (CPU)...")
fps_onnx_int8 = bench_onnx(frames, ONNX_INT8, ["CPUExecutionProvider"])

print("\n" + "=" * 60)
print(f"INFERENCE THROUGHPUT ({len(frames)} frames @ {IMGSZ}x{IMGSZ})")
print("=" * 60)
print(f"{'Model':<30} {'Hardware':<10} {'FPS':>8} {'ms/frame':>10}")
print("-" * 60)
print(f"{'YOLOv8s PyTorch FP32':<30} {'GPU':<10} {fps_torch:>8.1f} {1000/fps_torch:>10.2f}")
print(f"{'YOLOv8s ONNX FP32':<30} {onnx_hardware:<10} {fps_onnx_fp32:>8.1f} {1000/fps_onnx_fp32:>10.2f}")
print(f"{'YOLOv8s ONNX INT8':<30} {'CPU':<10} {fps_onnx_int8:>8.1f} {1000/fps_onnx_int8:>10.2f}")
print("=" * 60)