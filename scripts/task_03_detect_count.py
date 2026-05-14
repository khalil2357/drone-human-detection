#!/usr/bin/env python3
"""Task-03: Human & Car Detection with Human Counting."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task_03_detection_count"

CLASS_NAMES = {0: "human", 1: "car"}
CLASS_COLORS = {0: (40, 220, 40), 1: (40, 170, 255)}  # BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-03 detection and counting")
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

    prediction_dir = OUTPUT_ROOT / "prediction_outputs"
    counting_dir = OUTPUT_ROOT / "counting_visualization"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    counting_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    images = get_images(args.source, args.max_images)
    if not images:
        raise RuntimeError("No images found to process")

    stats = []
    total_time = 0.0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        result = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dt = time.perf_counter() - t0
        total_time += dt
        fps = 1.0 / dt if dt > 0 else 0.0

        detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, classes):
                if cls_id not in (0, 1):
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append((cls_id, float(conf), (x1, y1, x2, y2)))

        human_count = sum(1 for cls, _, _ in detections if cls == 0)
        car_count = sum(1 for cls, _, _ in detections if cls == 1)

        vis = frame.copy()
        for cls_id, conf, (x1, y1, x2, y2) in detections:
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        cv2.putText(vis, f"Total Humans: {human_count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Cars: {car_count}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.2f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)

        cv2.imwrite(str(prediction_dir / img_path.name), vis)
        cv2.imwrite(str(counting_dir / img_path.name), vis)

        stats.append({
            "file": img_path.name,
            "human_count": human_count,
            "car_count": car_count,
            "detections": len(detections),
            "fps": fps,
        })

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(images)} images")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "processed_images": len(stats),
        "average_fps": (len(stats) / total_time) if total_time > 0 else 0.0,
        "average_human_count": (sum(s["human_count"] for s in stats) / len(stats)) if stats else 0.0,
        "average_car_count": (sum(s["car_count"] for s in stats) / len(stats)) if stats else 0.0,
        "weights": str(args.weights),
        "source": str(args.source),
        "prediction_outputs": str(prediction_dir),
        "counting_visualization": str(counting_dir),
        "frames": stats,
    }

    summary_path = OUTPUT_ROOT / "task_03_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Task-03 complete")
    print(f"Prediction outputs: {prediction_dir}")
    print(f"Counting visualization: {counting_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
