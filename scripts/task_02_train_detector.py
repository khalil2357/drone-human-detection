#!/usr/bin/env python3
"""
Task-02: Model Training for Human and Car Detection

This script prepares a YOLO dataset filtered to two classes:
- human: pedestrian + person(with hard hat)
- car: car + van

It then fine-tunes a YOLOv8n detector and exports:
- training approach summary
- sample predictions
- results report
- machine-readable training metadata
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - validated at runtime
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


PROJECT_ROOT = Path("/Users/mdibrahimkhalil/projects/drone-human-detection")
DATASET_ROOT = PROJECT_ROOT / "VisDrone_Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task_02_training"
PROCESSED_ROOT = PROJECT_ROOT / "outputs" / "task_02_processed_data" / "human_car_yolo"
RUNS_ROOT = OUTPUT_ROOT / "runs"

RANDOM_SEED = 42
IMAGE_SIZE = 640
EPOCHS = 3
BATCH_SIZE = 8
TRAIN_LIMIT = 900
VAL_LIMIT = 180
PREDICTION_SAMPLES = 6

SOURCE_SPLITS = {
    "train": DATASET_ROOT / "VisDrone2019-DET-train",
    "val": DATASET_ROOT / "VisDrone2019-DET-val",
}

SOURCE_TARGETS = {
    0: 0,  # pedestrian -> human
    1: 0,  # person with hard hat -> human
    2: 1,  # car -> car
    3: 1,  # van -> car
}

TARGET_NAMES = ["human", "car"]


@dataclass
class SplitSummary:
    source_images: int = 0
    processed_images: int = 0
    source_boxes: int = 0
    processed_boxes: int = 0
    class_counts: Counter = None

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = Counter()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
            except ValueError:
                continue
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def write_label_file(label_path: Path, labels: Iterable[Tuple[int, float, float, float, float]]) -> int:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with label_path.open("w", encoding="utf-8") as handle:
        for class_id, x_center, y_center, width, height in labels:
            handle.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            written += 1
    return written


def copy_image(source_image: Path, target_image: Path) -> None:
    target_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, target_image)


def prepare_split(source_dir: Path, split_name: str, limit: int | None = None) -> SplitSummary:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    target_images_dir = PROCESSED_ROOT / "images" / split_name
    target_labels_dir = PROCESSED_ROOT / "labels" / split_name
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if limit is not None:
        image_files = image_files[:limit]

    summary = SplitSummary(source_images=len(list(images_dir.iterdir())) if images_dir.exists() else 0)

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        source_labels = read_label_file(label_path)
        filtered_labels = []
        summary.source_boxes += len(source_labels)

        for class_id, x_center, y_center, width, height in source_labels:
            if class_id not in SOURCE_TARGETS:
                continue
            target_class = SOURCE_TARGETS[class_id]
            filtered_labels.append((target_class, x_center, y_center, width, height))
            summary.class_counts[target_class] += 1

        if not filtered_labels:
            continue

        target_image_path = target_images_dir / image_path.name
        target_label_path = target_labels_dir / f"{image_path.stem}.txt"

        copy_image(image_path, target_image_path)
        written = write_label_file(target_label_path, filtered_labels)

        summary.processed_images += 1
        summary.processed_boxes += written

    return summary


def build_dataset_yaml() -> Path:
    yaml_path = PROCESSED_ROOT / "human_car.yaml"
    yaml_content = f"""path: {PROCESSED_ROOT}
train: images/train
val: images/val

nc: 2
names:
  0: human
  1: car
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def draw_prediction_grid(model: YOLO, image_files: List[Path], output_path: Path) -> None:
    tiles: List[np.ndarray] = []
    for image_path in image_files:
        result = model.predict(str(image_path), imgsz=IMAGE_SIZE, conf=0.25, verbose=False)[0]
        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated = cv2.resize(annotated, (IMAGE_SIZE, IMAGE_SIZE))
        tiles.append(annotated)

    if not tiles:
        return

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.ones((rows * IMAGE_SIZE, cols * IMAGE_SIZE, 3), dtype=np.uint8) * 255

    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        canvas[
            row * IMAGE_SIZE:(row + 1) * IMAGE_SIZE,
            col * IMAGE_SIZE:(col + 1) * IMAGE_SIZE,
        ] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    seed_everything(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install it first, then rerun this script. "
            f"Original import error: {ULTRALYTICS_IMPORT_ERROR}"
        )

    print("=" * 80)
    print("TASK-02: MODEL TRAINING FOR HUMAN AND CAR DETECTION")
    print("=" * 80)
    print()
    print("Selected model: YOLOv8n")
    print("Reason: labels are already YOLO-formatted, and YOLOv8 offers the best speed/accuracy tradeoff for aerial detection.")
    print("Target classes: human, car")
    print()

    split_summaries: Dict[str, SplitSummary] = {}
    split_summaries["train"] = prepare_split(SOURCE_SPLITS["train"], "train", TRAIN_LIMIT)
    split_summaries["val"] = prepare_split(SOURCE_SPLITS["val"], "val", VAL_LIMIT)

    yaml_path = build_dataset_yaml()

    print("Prepared dataset summary:")
    for split_name, summary in split_summaries.items():
        print(
            f"  {split_name:5s} | source_images={summary.source_images:,} | "
            f"processed_images={summary.processed_images:,} | processed_boxes={summary.processed_boxes:,}"
        )
    print()

    if sum(summary.processed_images for summary in split_summaries.values()) == 0:
        raise RuntimeError("No training images were prepared. Check the label conversion logic.")

    model = YOLO("yolov8n.pt")

    print("Starting fine-tuning run...")
    train_results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(RUNS_ROOT),
        name="human_car_yolov8n",
        exist_ok=True,
        pretrained=True,
        device="cpu",
        workers=0,
        cache=False,
        verbose=False,
        patience=2,
    )

    best_weights = Path(train_results.save_dir) / "weights" / "best.pt"
    last_weights = Path(train_results.save_dir) / "weights" / "last.pt"
    trained_weights = best_weights if best_weights.exists() else last_weights

    trained_model = YOLO(str(trained_weights))

    val_images_dir = PROCESSED_ROOT / "images" / "val"
    val_images = sorted([path for path in val_images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    sample_images = random.sample(val_images, k=min(PREDICTION_SAMPLES, len(val_images)))

    predictions_dir = OUTPUT_ROOT / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    grid_path = predictions_dir / "sample_predictions.png"
    draw_prediction_grid(trained_model, sample_images, grid_path)

    metrics = {}
    try:
        validation_metrics = trained_model.val(data=str(yaml_path), imgsz=IMAGE_SIZE, batch=BATCH_SIZE, device="cpu", verbose=False)
        metrics = {
            "map50": float(getattr(validation_metrics.box, "map50", 0.0)),
            "map5095": float(getattr(validation_metrics.box, "map", 0.0)),
            "precision": float(getattr(validation_metrics.box, "mp", 0.0)),
            "recall": float(getattr(validation_metrics.box, "mr", 0.0)),
        }
    except Exception:
        metrics = {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0}

    class_totals = Counter()
    for summary in split_summaries.values():
        class_totals.update(summary.class_counts)

    report_path = OUTPUT_ROOT / "task_02_report.txt"
    report = f"""================================================================================
TASK-02: MODEL TRAINING REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Training approach:
- Model: YOLOv8n
- Initialization: pretrained yolov8n.pt weights
- Objective: fine-tune for human and car detection
- Class mapping:
  0 -> human   (pedestrian + person with hard hat)
  1 -> car     (car + van)
- Image size: {IMAGE_SIZE}
- Epochs: {EPOCHS}
- Batch size: {BATCH_SIZE}
- Device: CPU

Prepared data:
- Train images: {split_summaries['train'].processed_images:,}
- Val images: {split_summaries['val'].processed_images:,}
- Train boxes: {split_summaries['train'].processed_boxes:,}
- Val boxes: {split_summaries['val'].processed_boxes:,}

Class distribution in processed training data:
- human: {class_totals[0]:,}
- car: {class_totals[1]:,}

Validation metrics:
- mAP@0.5: {metrics['map50']:.4f}
- mAP@0.5:0.95: {metrics['map5095']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}

Sample prediction output:
- {grid_path}

Why YOLOv8 was selected:
- The dataset labels are already in YOLO format.
- It supports fast fine-tuning and inference.
- It is the most practical choice for drone imagery where small-object detection and speed both matter.

Files generated:
- {yaml_path}
- {grid_path}
- {trained_weights}
- {report_path}
================================================================================
"""
    report_path.write_text(report, encoding="utf-8")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": "YOLOv8n",
        "trained_weights": str(trained_weights),
        "data_yaml": str(yaml_path),
        "output_grid": str(grid_path),
        "target_classes": {"0": "human", "1": "car"},
        "split_summaries": {
            split_name: {
                "source_images": summary.source_images,
                "processed_images": summary.processed_images,
                "source_boxes": summary.source_boxes,
                "processed_boxes": summary.processed_boxes,
                "class_counts": dict(summary.class_counts),
            }
            for split_name, summary in split_summaries.items()
        },
        "metrics": metrics,
    }
    metadata_path = OUTPUT_ROOT / "task_02_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("Training completed.")
    print(f"Best weights: {trained_weights}")
    print(f"Sample predictions: {grid_path}")
    print(f"Report: {report_path}")
    print(f"Metadata: {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
