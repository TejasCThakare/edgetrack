"""Benchmark inference throughput: PyTorch vs ONNX FP32 vs ONNX INT8."""
import time
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

VIDEO = "input.mp4"
N_FRAMES = 300
IMGSZ = 640


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
    m = YOLO("yolov8n.pt")
    # warmup
    for _ in range(5):
        m(frames[0], verbose=False)
    t = time.time()
    for f in frames:
        m(f, verbose=False)
    return len(frames) / (time.time() - t)


def bench_onnx(frames, path, providers):
    sess = ort.InferenceSession(path, providers=providers)
    iname = sess.get_inputs()[0].name
    # warmup
    for _ in range(5):
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

print("Benchmarking PyTorch FP32 (GPU)...")
fps_torch = bench_pytorch(frames)

print("Benchmarking ONNX FP32 (GPU)...")
fps_onnx_fp32 = bench_onnx(frames, "yolov8n.onnx", ["CUDAExecutionProvider"])

print("Benchmarking ONNX INT8 (CPU)...")
fps_onnx_int8 = bench_onnx(frames, "yolov8n_int8.onnx", ["CPUExecutionProvider"])

print("\n" + "=" * 50)
print(f"INFERENCE THROUGHPUT ({len(frames)} frames @ {IMGSZ}x{IMGSZ})")
print("=" * 50)
print(f"{'Model':<25} {'FPS':>10} {'ms/frame':>12}")
print("-" * 50)
print(f"{'PyTorch FP32 (GPU)':<25} {fps_torch:>10.1f} {1000/fps_torch:>12.2f}")
print(f"{'ONNX FP32 (GPU)':<25} {fps_onnx_fp32:>10.1f} {1000/fps_onnx_fp32:>12.2f}")
print(f"{'ONNX INT8 (CPU)':<25} {fps_onnx_int8:>10.1f} {1000/fps_onnx_int8:>12.2f}")
print("=" * 50)

speedup = (fps_onnx_fp32 - fps_torch) / fps_torch * 100
print(f"\nONNX FP32 vs PyTorch:  {speedup:+.1f}% throughput")