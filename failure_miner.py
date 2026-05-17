"""Auto-mine failure cases. Tracker via TRACKER env var (default: bytetrack)."""
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO

TRACKER = os.environ.get("TRACKER", "bytetrack")
TRACKER_CFG = f"./{TRACKER}.yaml"

VIDEO = "input.mp4"
OUT_DIR = f"assets/failures_{TRACKER}"
CONF_THRESHOLD = 0.45
MIN_LOW_CONF = 2
DET_WEIGHTS = "yolov8s.pt"

os.makedirs(OUT_DIR, exist_ok=True)
yolo = YOLO(DET_WEIGHTS)

prev_ids = set()
flagged = []
id_lifetimes = defaultdict(int)

print(f"Mining failures from {VIDEO} with tracker={TRACKER}...")

for i, r in enumerate(
    yolo.track(
        source=VIDEO,
        tracker=TRACKER_CFG,
        stream=True,
        classes=[0],
        verbose=False,
    )
):
    if r.boxes.id is None:
        continue

    confs = r.boxes.conf.cpu().numpy()
    ids = set(r.boxes.id.cpu().numpy().astype(int))

    for tid in ids:
        id_lifetimes[tid] += 1

    low_conf_count = int((confs < CONF_THRESHOLD).sum())
    lost_ids = prev_ids - ids
    real_lost = sum(1 for tid in lost_ids if id_lifetimes[tid] >= 3)

    reasons = []
    if low_conf_count >= MIN_LOW_CONF:
        reasons.append(f"{low_conf_count} low-conf detections")
    if real_lost >= 1:
        reasons.append(f"{real_lost} lost track IDs")

    if reasons:
        out_path = f"{OUT_DIR}/frame_{i:05d}.jpg"
        cv2.imwrite(out_path, r.plot())
        flagged.append((i, " | ".join(reasons)))

    prev_ids = ids

print(f"\nFlagged {len(flagged)} failure frames in {OUT_DIR}/")
print("\nFirst 15:")
for fr, reason in flagged[:15]:
    print(f"  frame {fr:5d}: {reason}")