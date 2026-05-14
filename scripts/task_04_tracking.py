#!/usr/bin/env python3
"""Task-04: Object Tracking using ByteTrack (human and car)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


PROJECT_ROOT = Path("/Users/mdibrahimkhalil/projects/drone-human-detection")
DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "task_02_training" / "runs" / "human_car_yolov8n" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "VisDrone_Dataset" / "VisDrone2019-DET-test-dev" / "images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task_04_tracking"

CLASS_NAMES = {0: "human", 1: "car"}
CLASS_COLORS = {0: (40, 220, 40), 1: (40, 170, 255)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-04 tracking with ByteTrack")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=80)
    return parser.parse_args()


def get_images(source: Path, limit: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in source.iterdir() if p.suffix.lower() in exts])
    return images[:limit]


def main() -> None:
    args = parse_args()

    if YOLO is None:
        raise RuntimeError(f"ultralytics is required: {ULTRALYTICS_IMPORT_ERROR}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    tracks_dir = OUTPUT_ROOT / "tracked_images"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    images = get_images(args.source, args.max_images)
    if not images:
        raise RuntimeError("No images found to process")

    unique_humans = set()
    unique_cars = set()
    per_frame = []
    total_time = 0.0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        result = model.track(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]
        dt = time.perf_counter() - t0
        total_time += dt
        fps = 1.0 / dt if dt > 0 else 0.0

        vis = frame.copy()
        frame_humans = 0
        frame_cars = 0

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else [-1] * len(classes)
            )

            for box, conf, cls_id, track_id in zip(xyxy, confs, classes, track_ids):
                if cls_id not in (0, 1):
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label = f"ID {track_id} {CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                if cls_id == 0:
                    frame_humans += 1
                    if track_id >= 0:
                        unique_humans.add(int(track_id))
                elif cls_id == 1:
                    frame_cars += 1
                    if track_id >= 0:
                        unique_cars.add(int(track_id))

        cv2.putText(vis, f"Unique Humans: {len(unique_humans)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Unique Cars: {len(unique_cars)}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.2f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)

        cv2.imwrite(str(tracks_dir / img_path.name), vis)

        per_frame.append({
            "file": img_path.name,
            "frame_humans": frame_humans,
            "frame_cars": frame_cars,
            "unique_humans_so_far": len(unique_humans),
            "unique_cars_so_far": len(unique_cars),
            "fps": fps,
        })

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(images)} images")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "processed_images": len(per_frame),
        "unique_humans": len(unique_humans),
        "unique_cars": len(unique_cars),
        "average_fps": (len(per_frame) / total_time) if total_time > 0 else 0.0,
        "weights": str(args.weights),
        "source": str(args.source),
        "tracked_images": str(tracks_dir),
        "frames": per_frame,
    }

    summary_path = OUTPUT_ROOT / "task_04_tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Task-04 complete")
    print(f"Tracked images: {tracks_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
