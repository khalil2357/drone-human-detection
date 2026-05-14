# Drone-Based Human and Car Detection System

A complete computer vision pipeline for aerial object detection on the VisDrone dataset, built as an internship project and delivered across five tasks:

- Task-01: Dataset understanding and analysis
- Task-02: Model training (human/car detector)
- Task-03: Detection with human counting
- Task-04: Object tracking
- Task-05: Evaluation and visualization

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Final Task Status](#final-task-status)
- [Measured Results](#measured-results)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [How to Run Each Task](#how-to-run-each-task)
- [Generated Outputs](#generated-outputs)
- [Task-05 Discussion](#task-05-discussion)
- [Tech Stack](#tech-stack)
- [Notes](#notes)

## Project Overview

This repository implements a full workflow for human and car detection in aerial scenes:

1. Analyze and understand VisDrone data characteristics
2. Build a focused 2-class dataset (human, car)
3. Fine-tune a YOLOv8 detector
4. Run detection and human counting on unseen frames
5. Add object tracking with ByteTrack
6. Produce evaluation artifacts and a concise report

Why YOLOv8n was selected:

- VisDrone labels in this project are already YOLO-compatible
- Fast iteration and good speed/accuracy tradeoff
- Easy deployment path for inference and tracking

## Dataset

Source: VisDrone 2019 Object Detection Dataset

Kaggle reference:
https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

Dataset used in this workspace:

- Train: 6,471 images
- Validation: 548 images
- Test-dev: 1,610 images
- Test-challenge: 1,580 images available in this local copy

For Task-02, classes were remapped to a focused 2-class problem:

- human: pedestrian + person-with-hard-hat
- car: car + van

## Final Task Status

| Task | Title | Status |
|---|---|---|
| Task-01 | Dataset Understanding and Analysis | Completed |
| Task-02 | Model Training for Human/Car | Completed |
| Task-03 | Detection + Human Counting | Completed |
| Task-04 | Object Tracking (ByteTrack) | Completed |
| Task-05 | Evaluation + Visualization | Completed |

## Measured Results

### Task-02 Training Metrics

From outputs/task_02_training/task_02_report.txt:

- Model: YOLOv8n (fine-tuned)
- Train images: 899
- Validation images: 180
- mAP@0.5: 0.3832
- mAP@0.5:0.95: 0.1897
- Precision: 0.5357
- Recall: 0.3925

### Task-03 Inference and Counting Summary

From outputs/task_03_detection_count/task_03_summary.json:

- Processed images: 80
- Average FPS: 54.80
- Average human count/frame: 20.79
- Average car count/frame: 2.44

### Task-04 Tracking Summary

From outputs/task_04_tracking/task_04_tracking_summary.json:

- Processed images: 80
- Tracker: ByteTrack
- Unique humans tracked: 38
- Unique cars tracked: 25
- Average FPS: 40.14

### Task-05 Evaluation Summary

From outputs/task_05_evaluation/task_05_metrics_summary.json:

- Precision: 0.5357440618797931
- Recall: 0.392458807110761
- mAP@0.5: 0.38318931831195185
- mAP@0.5:0.95: 0.1897166111257797
- FPS (from Task-03 run): 54.79721983647497

## Project Structure

```text
drone-human-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── task_01_dataset_analysis.py
│   ├── task_02_train_detector.py
│   ├── task_03_detect_count.py
│   ├── task_04_tracking.py
│   ├── task_05_evaluation_visualization.py
│   └── visdrone_analysis.py
├── VisDrone_Dataset/
│   ├── visdrone.yaml
│   ├── VisDrone2019-DET-train/
│   ├── VisDrone2019-DET-val/
│   ├── VisDrone2019-DET-test-dev/
│   └── VisDrone2019-DET-test-challenge/
└── outputs/
    ├── task_01_analysis/
    ├── task_02_processed_data/
    ├── task_02_training/
    ├── task_03_detection_count/
    ├── task_04_tracking/
    └── task_05_evaluation/
```

## Environment Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Confirm dataset folders

Expected base path:

```text
VisDrone_Dataset/
```

## How to Run Each Task

Run all commands from scripts/ with the venv activated.

### Task-01

```bash
python3 task_01_dataset_analysis.py
```

### Task-02

```bash
python3 task_02_train_detector.py
```

### Task-03

```bash
python3 task_03_detect_count.py --max-images 80
```

### Task-04

```bash
python3 task_04_tracking.py --max-images 80
```

### Task-05

```bash
python3 task_05_evaluation_visualization.py
```

## Generated Outputs

### Task-01

- outputs/task_01_analysis/01_dataset_overview.png
- outputs/task_01_analysis/02_sample_annotations.png
- outputs/task_01_analysis/03_challenges_summary.png
- outputs/task_01_analysis/task_01_report.txt
- outputs/task_01_analysis/dataset_statistics.json

### Task-02

- outputs/task_02_processed_data/human_car_yolo/human_car.yaml
- outputs/task_02_training/predictions/sample_predictions.png
- outputs/task_02_training/runs/human_car_yolov8n/weights/best.pt
- outputs/task_02_training/task_02_report.txt
- outputs/task_02_training/task_02_metadata.json

### Task-03

- outputs/task_03_detection_count/prediction_outputs/
- outputs/task_03_detection_count/counting_visualization/
- outputs/task_03_detection_count/task_03_summary.json

### Task-04

- outputs/task_04_tracking/tracked_images/
- outputs/task_04_tracking/task_04_tracking_summary.json

### Task-05

- outputs/task_05_evaluation/prediction_outputs_grid.jpg
- outputs/task_05_evaluation/counting_visualization_grid.jpg
- outputs/task_05_evaluation/tracking_outputs_grid.jpg
- outputs/task_05_evaluation/processed_images_results/
- outputs/task_05_evaluation/task_05_metrics_summary.json
- outputs/task_05_evaluation/task_05_report.txt

## Task-05 Discussion

### Strengths

- Focused 2-class setup improved practical learning speed
- Inference and counting pipeline is simple and easy to interpret
- ByteTrack adds temporal continuity and unique-object insights
- End-to-end outputs are reproducible and organized by task

### Limitations

- Per-frame counting can fluctuate in crowded scenes
- Tracking IDs may switch during heavy occlusion
- CPU inference limits throughput for larger production workloads

### Challenges Faced

- Small, far-away humans in aerial perspective are difficult to detect reliably
- Motion blur and camera angle changes reduce confidence on some frames
- Vehicle appearance diversity introduces occasional class confusion

## Tech Stack

- Python 3.9
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib

## Notes

This README reflects completed implementation results from the current repository state and output artifacts.

If you retrain with different epochs, confidence thresholds, or image limits, the metrics and summary files will update accordingly.
