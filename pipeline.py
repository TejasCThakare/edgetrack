"""End-to-end perception pipeline: YOLOv8 detection + ByteTrack tracking + MobileSAM segmentation.

Reads input.mp4, runs detection + tracking + per-track segmentation,
writes annotated video to outputs/demo.mp4.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

VIDEO_IN = "input.mp4"
VIDEO_OUT = "outputs/demo.mp4"
SAM_CKPT = "mobile_sam.pt"
DEVICE = "cuda"

# Load models
print("Loading YOLOv8n...")
yolo = YOLO("yolov8n.pt")

print("Loading MobileSAM...")
sam = sam_model_registry["vit_t"](checkpoint=SAM_CKPT).to(DEVICE).eval()
predictor = SamPredictor(sam)


def color_for_id(tid):
    """Deterministic color per track ID."""
    np.random.seed(int(tid))
    return tuple(int(x) for x in np.random.randint(50, 255, 3))


# Setup video writer
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

# Stream through video with built-in ByteTrack tracker
# classes=[0] = person; remove to track all COCO classes
for i, r in enumerate(
    yolo.track(
        source=VIDEO_IN,
        tracker="bytetrack.yaml",
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

    # Prompt MobileSAM with each tracked box
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for box, tid in zip(boxes, ids):
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0]
        color = color_for_id(tid)

        # Overlay mask
        overlay = frame.copy()
        overlay[mask] = color
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Box + ID label
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"ID {tid}", (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    writer.write(frame)

    if (i + 1) % 30 == 0:
        print(f"  frame {i + 1}/{total}")

writer.release()
print(f"Done. Output: {VIDEO_OUT}")