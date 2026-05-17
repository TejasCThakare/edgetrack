"""End-to-end perception pipeline: YOLOv8 detection + tracker + MobileSAM segmentation.

Tracker is configurable via TRACKER env var (default: bytetrack).
Outputs go to outputs/demo_<tracker>.mp4 so both runs are preserved.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

TRACKER = os.environ.get("TRACKER", "bytetrack")  # "bytetrack" or "botsort"
TRACKER_CFG = f"./{TRACKER}.yaml"

VIDEO_IN = "input.mp4"
VIDEO_OUT = f"outputs/demo_{TRACKER}.mp4"
SAM_CKPT = "mobile_sam.pt"
DEVICE = "cuda"
DET_WEIGHTS = "yolov8s.pt"

print(f"Tracker: {TRACKER}  (config: {TRACKER_CFG})")

# Load models
print(f"Loading {DET_WEIGHTS}...")
yolo = YOLO(DET_WEIGHTS)

print("Loading MobileSAM...")
sam = sam_model_registry["vit_t"](checkpoint=SAM_CKPT).to(DEVICE).eval()
predictor = SamPredictor(sam)


def color_for_id(tid):
    np.random.seed(int(tid))
    return tuple(int(x) for x in np.random.randint(50, 255, 3))


cap = cv2.VideoCapture(VIDEO_IN)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

writer = cv2.VideoWriter(
    VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

print(f"Processing {total} frames at {w}x{h} @ {fps:.1f} FPS...")

# Stats for comparison
total_dets = 0
unique_ids = set()

for i, r in enumerate(
    yolo.track(
        source=VIDEO_IN,
        tracker=TRACKER_CFG,
        stream=True,
        persist=True,
        classes=[0],
        verbose=False,
    )
):
    frame = r.orig_img.copy()

    if r.boxes.id is None:
        writer.write(frame)
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy().astype(int)

    total_dets += len(ids)
    unique_ids.update(ids.tolist())

    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for box, tid in zip(boxes, ids):
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0]
        color = color_for_id(tid)

        overlay = frame.copy()
        overlay[mask] = color
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"ID {tid}", (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

    writer.write(frame)

    if (i + 1) % 30 == 0:
        print(f"  frame {i + 1}/{total}")

writer.release()
print(f"\nDone. Output: {VIDEO_OUT}")
print(f"Total detections: {total_dets}")
print(f"Unique track IDs assigned: {len(unique_ids)}")
print(f"Avg detections per unique ID: {total_dets / max(len(unique_ids), 1):.1f}")