#!/usr/bin/env python3
"""
TASK-01: Dataset Understanding & Preprocessing
VisDrone Object Detection Dataset Analysis

Generates comprehensive analysis, visualizations, and recommendations for the
VisDrone 2019 dataset used for drone-based human detection.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_ROOT = "/Users/mdibrahimkhalil/projects/drone-human-detection/VisDrone_Dataset"
OUTPUT_DIR = "/Users/mdibrahimkhalil/projects/drone-human-detection/outputs"
REPORT_DIR = os.path.join(OUTPUT_DIR, "task_01_analysis")
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 80)
print("TASK-01: DATASET UNDERSTANDING & PREPROCESSING")
print("=" * 80)
print("\n[1/4] Analyzing Dataset Structure...\n")

# Dataset structure analysis
splits = {
    "train": "VisDrone2019-DET-train",
    "validation": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
    "test-challenge": "VisDrone2019-DET-test-challenge"
}

dataset_info = {}
for split_name, split_folder in splits.items():
    split_path = os.path.join(DATASET_ROOT, split_folder)
    if os.path.exists(split_path):
        images_dir = os.path.join(split_path, "images")
        labels_dir = os.path.join(split_path, "labels")
        num_images = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
        num_labels = len(os.listdir(labels_dir)) if os.path.exists(labels_dir) else 0
        dataset_info[split_name] = {
            "num_images": num_images,
            "num_labels": num_labels,
            "images_path": images_dir,
            "labels_path": labels_dir
        }

print("Dataset Structure Overview:")
print("-" * 80)
for split_name, info in dataset_info.items():
    print(f"  {split_name.upper():20} | Images: {info['num_images']:5d} | Labels: {info['num_labels']:5d}")
print()

# Object analysis
print("[2/4] Analyzing Annotation Format & Objects...\n")

class_names = {
    0: "Pedestrian", 1: "Person (with hard hat)", 2: "Car", 3: "Van", 4: "Truck",
    5: "Tricycle", 6: "Awning-tricycle", 7: "Bus", 8: "Motorcycle", 9: "Bicycle"
}

object_stats = Counter()
bbox_sizes = []

testdev_labels_dir = dataset_info.get("test-dev", {}).get("labels_path", "")
if os.path.exists(testdev_labels_dir):
    for label_file in os.listdir(testdev_labels_dir):
        label_path = os.path.join(testdev_labels_dir, label_file)
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            obj_category = int(parts[0])
                            x_norm, y_norm, w_norm, h_norm = map(float, parts[1:5])
                            w_pixels = w_norm * 640
                            h_pixels = h_norm * 480
                            area = w_pixels * h_pixels
                            object_stats[class_names.get(obj_category, f"Class_{obj_category}")] += 1
                            bbox_sizes.append(area)
                        except (ValueError, IndexError):
                            pass
        except Exception as e:
            pass

print(f"Total objects analyzed: {sum(object_stats.values()):,}")
print(f"Object categories: {len(object_stats)}")
print("\nObject Distribution:")
print("-" * 80)
for obj_class, count in object_stats.most_common():
    percentage = (count / sum(object_stats.values())) * 100 if object_stats else 0
    print(f"  {obj_class:30} | Count: {count:6d} | {percentage:5.1f}%")
print()

# Challenges
print("[3/4] Identifying Dataset Challenges & Characteristics...\n")

challenges = [
    {"name": "Class Imbalance", "severity": "HIGH", 
     "description": "Significant imbalance in object categories",
     "mitigation": "Weighted loss functions, data augmentation"},
    {"name": "Small Object Detection", "severity": "HIGH",
     "description": "Many objects < 20x20 pixels", 
     "mitigation": "Multi-scale FPN, higher resolution"},
    {"name": "Aerial Perspective", "severity": "HIGH",
     "description": "Extreme viewpoint variation from drone angles",
     "mitigation": "Perspective augmentation, transfer learning"},
    {"name": "Scale Variation", "severity": "HIGH",
     "description": "Object sizes vary dramatically",
     "mitigation": "Feature pyramids, multi-scale heads"},
    {"name": "Occlusion & Truncation", "severity": "MEDIUM",
     "description": "Real-world occlusion and boundary truncation",
     "mitigation": "Context learning, ensemble methods"},
    {"name": "Real-World Variability", "severity": "MEDIUM",
     "description": "Complex backgrounds, weather, lighting",
     "mitigation": "Extensive augmentation, domain randomization"}
]

print("Identified Challenges:")
print("-" * 80)
for i, challenge in enumerate(challenges, 1):
    print(f"\n{i}. {challenge['name']} [{challenge['severity']}]")
    print(f"   {challenge['description']}")
    print(f"   → {challenge['mitigation']}")
print()

# Visualizations
print("[4/4] Generating Visualizations...\n")

# Figure 1
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('VisDrone Dataset Overview', fontsize=16, fontweight='bold')

ax = axes[0, 0]
splits_names = list(dataset_info.keys())
splits_counts = [dataset_info[s]["num_images"] for s in splits_names]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
ax.bar(splits_names, splits_counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
ax.set_title('Dataset Split Distribution', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(splits_counts):
    ax.text(i, v + 100, str(v), ha='center', fontweight='bold')

ax = axes[0, 1]
if object_stats:
    top_classes = object_stats.most_common(6)
    class_names_list = [c[0][:20] for c in top_classes]
    class_counts = [c[1] for c in top_classes]
    ax.barh(class_names_list, class_counts, color='#95E1D3', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Top 6 Object Classes', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(class_counts):
        ax.text(v + 50, i, str(v), va='center', fontweight='bold')

ax = axes[1, 0]
if bbox_sizes:
    ax.hist(bbox_sizes, bins=50, color='#A8E6CF', edgecolor='black', linewidth=1)
    ax.set_xlabel('Bounding Box Area (pixels²)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Bounding Box Size Distribution', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

ax = axes[1, 1]
challenge_names = [c['name'][:15] for c in challenges[:5]]
severity_colors = {'HIGH': '#FF6B6B', 'MEDIUM': '#FFA07A', 'LOW': '#FFD93D'}
colors_list = [severity_colors[c['severity']] for c in challenges[:5]]
y_pos = np.arange(len(challenge_names))
ax.barh(y_pos, [1]*len(challenge_names), color=colors_list, edgecolor='black', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(challenge_names, fontsize=9)
ax.set_xlim(0, 1.2)
ax.set_title('Top Challenges', fontsize=12, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "01_dataset_overview.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: 01_dataset_overview.png")
plt.close()

# Figure 2: Sample images
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Sample Annotated Images from Test-Dev Set', fontsize=16, fontweight='bold')

testdev_images_dir = dataset_info.get("test-dev", {}).get("images_path", "")
testdev_labels_dir = dataset_info.get("test-dev", {}).get("labels_path", "")

if os.path.exists(testdev_images_dir):
    image_files = [f for f in os.listdir(testdev_images_dir) if f.endswith(('.jpg', '.png'))]
    sample_images = np.random.choice(image_files, min(6, len(image_files)), replace=False)

    for idx, (ax, img_file) in enumerate(zip(axes.flat, sample_images)):
        img_path = os.path.join(testdev_images_dir, img_file)
        label_path = os.path.join(testdev_labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            ax.imshow(img)
            ax.set_title(f'{img_file[:20]}...', fontsize=10, fontweight='bold')
            ax.axis('off')
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                x_norm, y_norm, w_norm, h_norm = map(float, parts[1:5])
                                x = int(x_norm * w)
                                y = int(y_norm * h)
                                box_w = int(w_norm * w)
                                box_h = int(h_norm * h)
                                rect = patches.Rectangle((x - box_w//2, y - box_h//2), box_w, box_h,
                                                        linewidth=2, edgecolor='lime', facecolor='none')
                                ax.add_patch(rect)
                            except:
                                pass
        except Exception as e:
            pass

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "02_sample_annotations.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: 02_sample_annotations.png")
plt.close()

# Figure 3
fig, ax = plt.subplots(figsize=(12, 8))
challenge_names = [c['name'] for c in challenges]
severity_colors = {'HIGH': '#FF6B6B', 'MEDIUM': '#FFA07A', 'LOW': '#FFD93D'}
colors = [severity_colors[c['severity']] for c in challenges]
y_pos = np.arange(len(challenge_names))
ax.barh(y_pos, [1] * len(challenge_names), color=colors, edgecolor='black', linewidth=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(challenge_names, fontsize=11, fontweight='bold')
ax.set_xlim(0, 1.3)
ax.set_title('Dataset Challenges Summary', fontsize=14, fontweight='bold')
ax.axis('off')

for i, challenge in enumerate(challenges):
    severity = challenge['severity']
    ax.text(1.05, i, severity, ha='left', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=severity_colors[severity], alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "03_challenges_summary.png"), dpi=300, bbox_inches='tight')
print("✓ Saved: 03_challenges_summary.png")
plt.close()

# Report
print("Generating comprehensive report...\n")

report = f"""
================================================================================
        TASK-01: DATASET UNDERSTANDING & PREPROCESSING REPORT
                  VisDrone Object Detection Dataset
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
1. DATASET STRUCTURE
================================================================================

DATASET SPLIT OVERVIEW:
"""

for split_name, info in dataset_info.items():
    report += f"  {split_name.upper():20} | Images: {info['num_images']:5,} | Labels: {info['num_labels']:5,}\n"

total_images = sum(info['num_images'] for info in dataset_info.values())
total_labels = sum(info['num_labels'] for info in dataset_info.values())

report += f"""
TOTAL: {total_images:,} images | {total_labels:,} annotation files

ANNOTATION FORMAT:
  YOLO Format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

OBJECT CATEGORIES (10 total):
  0 = Pedestrian                    5 = Tricycle
  1 = Person (with hard hat)        6 = Awning-tricycle
  2 = Car                           7 = Bus
  3 = Van                           8 = Motorcycle
  4 = Truck                         9 = Bicycle

================================================================================
2. PREPROCESSING & AUGMENTATION STRATEGY
================================================================================

RECOMMENDED PREPROCESSING:
  • Image Resize: 416×416 or 640×640
  • Normalization: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
  • Filter invalid boxes and boundary violations

DATA AUGMENTATION:
  • Horizontal Flip (50%)
  • Vertical Flip (10%)
  • Random Rotation (±5-15 degrees)
  • Brightness/Contrast (±20%)
  • Mosaic Augmentation (YOLOv4 style)
  • CutMix and Mixup

================================================================================
3. KEY CHALLENGES & MITIGATION
================================================================================
"""

for i, challenge in enumerate(challenges, 1):
    report += f"""
{i}. {challenge['name']} [{challenge['severity']}]
   Description: {challenge['description']}
   Mitigation: {challenge['mitigation']}
"""

report += f"""
================================================================================
4. OBJECT STATISTICS
================================================================================

Total Objects: {sum(object_stats.values()):,}

"""

for obj_class, count in object_stats.most_common():
    percentage = (count / sum(object_stats.values())) * 100 if object_stats else 0
    report += f"  {obj_class:30} {count:>7,} ({percentage:5.1f}%)\n"

if bbox_sizes:
    report += f"""
BOUNDING BOX STATISTICS:
  Min:     {min(bbox_sizes):>8.0f} px²
  Max:     {max(bbox_sizes):>8.0f} px²
  Mean:    {np.mean(bbox_sizes):>8.0f} px²
  Median:  {np.median(bbox_sizes):>8.0f} px²

================================================================================
5. TRAINING RECOMMENDATIONS
================================================================================

Loss Function: Weighted CrossEntropyLoss or Focal Loss
Backbone: YOLOv8, EfficientDet, or Faster R-CNN
Batch Size: 16-32
Learning Rate: 0.001 (cosine annealing)
Epochs: 100-200
Optimizer: SGD (momentum=0.9) or Adam

================================================================================
6. OUTPUT FILES
================================================================================

Generated Files:
  ├─ 01_dataset_overview.png       (Dataset statistics)
  ├─ 02_sample_annotations.png     (Sample annotated images)
  ├─ 03_challenges_summary.png     (Challenge visualization)
  ├─ task_01_report.txt            (This report)
  └─ dataset_statistics.json       (Statistics in JSON)

Location: {REPORT_DIR}/

================================================================================
END OF REPORT
================================================================================
"""

report_path = os.path.join(REPORT_DIR, "task_01_report.txt")
with open(report_path, 'w') as f:
    f.write(report)
print(f"✓ Saved: task_01_report.txt\n")

# JSON stats
stats_data = {
    "timestamp": datetime.now().isoformat(),
    "dataset_splits": dataset_info,
    "object_statistics": dict(object_stats),
    "bbox_statistics": {
        "min": float(min(bbox_sizes)) if bbox_sizes else 0,
        "max": float(max(bbox_sizes)) if bbox_sizes else 0,
        "mean": float(np.mean(bbox_sizes)) if bbox_sizes else 0,
        "median": float(np.median(bbox_sizes)) if bbox_sizes else 0,
        "total_count": len(bbox_sizes)
    }
}

stats_path = os.path.join(REPORT_DIR, "dataset_statistics.json")
with open(stats_path, 'w') as f:
    json.dump(stats_data, f, indent=2)
print(f"✓ Saved: dataset_statistics.json\n")

# Summary
print("=" * 80)
print("✓ TASK-01 COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput Directory: {REPORT_DIR}")
print("\nGenerated Files:")
print("  ├─ 01_dataset_overview.png")
print("  ├─ 02_sample_annotations.png")
print("  ├─ 03_challenges_summary.png")
print("  ├─ task_01_report.txt")
print("  └─ dataset_statistics.json")
print("\n" + "=" * 80)
