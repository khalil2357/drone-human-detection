#!/usr/bin/env python3
"""Task-05: Evaluation and visualization for Task-03 and Task-04 outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


PROJECT_ROOT = Path("/Users/mdibrahimkhalil/projects/drone-human-detection")
TASK03_ROOT = PROJECT_ROOT / "outputs" / "task_03_detection_count"
TASK04_ROOT = PROJECT_ROOT / "outputs" / "task_04_tracking"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task_05_evaluation"

DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "task_02_training" / "runs" / "human_car_yolov8n" / "weights" / "best.pt"
DEFAULT_DATA_YAML = PROJECT_ROOT / "outputs" / "task_02_processed_data" / "human_car_yolo" / "human_car.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-05 evaluation and visualization")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--max-preview", type=int, default=6)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gather_images(folder: Path, limit: int) -> List[Path]:
    if not folder.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    return imgs[:limit]


def make_grid(images: List[Path], output_path: Path, cell: int = 420, cols: int = 3) -> None:
    tiles = []
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (cell, cell))
        tiles.append(img)

    if not tiles:
        return

    rows = (len(tiles) + cols - 1) // cols
    canvas = np.ones((rows * cell, cols * cell, 3), dtype=np.uint8) * 255

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile

    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    args = parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    processed_dir = OUTPUT_ROOT / "processed_images_results"
    processed_dir.mkdir(parents=True, exist_ok=True)

    task03_summary = read_json(TASK03_ROOT / "task_03_summary.json")
    task04_summary = read_json(TASK04_ROOT / "task_04_tracking_summary.json")

    prediction_imgs = gather_images(TASK03_ROOT / "prediction_outputs", args.max_preview)
    counting_imgs = gather_images(TASK03_ROOT / "counting_visualization", args.max_preview)
    tracking_imgs = gather_images(TASK04_ROOT / "tracked_images", args.max_preview)

    pred_grid = OUTPUT_ROOT / "prediction_outputs_grid.jpg"
    count_grid = OUTPUT_ROOT / "counting_visualization_grid.jpg"
    track_grid = OUTPUT_ROOT / "tracking_outputs_grid.jpg"

    make_grid(prediction_imgs, pred_grid)
    make_grid(counting_imgs, count_grid)
    make_grid(tracking_imgs, track_grid)

    for p in prediction_imgs + counting_imgs + tracking_imgs:
        dst = processed_dir / p.name
        if not dst.exists():
            dst.write_bytes(p.read_bytes())

    metrics = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map5095": None,
        "fps": task03_summary.get("average_fps"),
    }

    if YOLO is not None and args.weights.exists() and args.data_yaml.exists():
        try:
            model = YOLO(str(args.weights))
            val = model.val(data=str(args.data_yaml), imgsz=640, batch=8, device="cpu", verbose=False)
            metrics.update(
                {
                    "precision": float(getattr(val.box, "mp", 0.0)),
                    "recall": float(getattr(val.box, "mr", 0.0)),
                    "map50": float(getattr(val.box, "map50", 0.0)),
                    "map5095": float(getattr(val.box, "map", 0.0)),
                }
            )
        except Exception:
            pass

    strengths = [
        "Strong integration with trained human/car detector from Task-02.",
        "Clear counting overlays make scene-level understanding easy.",
        "Tracking adds temporal consistency and unique-object insights.",
    ]
    limitations = [
        "Per-frame counting can fluctuate in crowded/occluded scenes.",
        "Tracking IDs may switch under heavy occlusion.",
        "CPU-only runs reduce throughput for large-scale inference.",
    ]
    challenges = [
        "Small distant humans in aerial views are hard to detect.",
        "Vehicle appearance diversity can cause class confusion.",
        "Motion blur and perspective distortion lower confidence.",
    ]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "task03_summary": task03_summary,
        "task04_summary": task04_summary,
        "metrics": metrics,
        "prediction_outputs_grid": str(pred_grid),
        "counting_visualization_grid": str(count_grid),
        "tracking_outputs_grid": str(track_grid),
        "processed_images_results": str(processed_dir),
        "strengths": strengths,
        "limitations": limitations,
        "challenges": challenges,
    }

    summary_path = OUTPUT_ROOT / "task_05_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "=" * 80,
        "TASK-05 EVALUATION & VISUALIZATION REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Prediction outputs:",
        f"- {TASK03_ROOT / 'prediction_outputs'}",
        f"- Grid: {pred_grid}",
        "",
        "Counting visualization:",
        f"- {TASK03_ROOT / 'counting_visualization'}",
        f"- Grid: {count_grid}",
        "",
        "Processed images/results:",
        f"- {processed_dir}",
        f"- Tracking grid: {track_grid}",
        "",
        "Optional metrics:",
        f"- Precision: {metrics['precision']}",
        f"- Recall: {metrics['recall']}",
        f"- mAP@0.5: {metrics['map50']}",
        f"- mAP@0.5:0.95: {metrics['map5095']}",
        f"- FPS: {metrics['fps']}",
        "",
        "Strengths:",
    ]
    report_lines.extend([f"- {s}" for s in strengths])
    report_lines.append("")
    report_lines.append("Limitations:")
    report_lines.extend([f"- {l}" for l in limitations])
    report_lines.append("")
    report_lines.append("Challenges faced:")
    report_lines.extend([f"- {c}" for c in challenges])
    report_lines.append("=" * 80)

    report_path = OUTPUT_ROOT / "task_05_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Task-05 complete")
    print(f"Metrics summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
