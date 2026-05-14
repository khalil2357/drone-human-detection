# Drone-Based Human Detection System

**An advanced computer vision project for detecting humans in aerial drone footage using state-of-the-art object detection models.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Tasks Overview](#tasks-overview)
- [Requirements](#requirements)
- [License](#license)

---

## 🎯 Overview

This project implements a comprehensive pipeline for human detection in drone-captured aerial footage. It leverages the **VisDrone 2019 Object Detection Dataset** and employs state-of-the-art deep learning models to achieve high-accuracy detection in challenging aerial scenarios.

### Key Features

✨ **Multi-scale Detection** - Detects objects across extreme scale variations  
✨ **Aerial Perspective Handling** - Optimized for drone camera angles and altitudes  
✨ **Comprehensive Analysis** - Complete dataset understanding and preprocessing pipeline  
✨ **Production-Ready** - Modular, well-documented, and scalable codebase  
✨ **Visualization Tools** - Built-in tools for dataset analysis and result visualization  

---

## 📊 Dataset

### VisDrone Object Detection Dataset

The project uses the **VisDrone 2019 Object Detection Dataset**, a large-scale benchmark for object detection in drone-captured videos.

**Dataset Link**: [VisDrone Dataset on Kaggle](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset?resource=download)

### Dataset Statistics

| Split | Images | Annotations |
|-------|--------|-------------|
| Training | 6,471 | 6,471 |
| Validation | 548 | 548 |
| Test-Dev | 1,610 | 1,610 |
| Test-Challenge | 2,540 | - |
| **Total** | **11,169** | **8,629** |

### Object Categories

The dataset includes 10 object categories:

| ID | Class | ID | Class |
|----|-------|----|----|
| 0 | Pedestrian | 5 | Tricycle |
| 1 | Person* | 6 | Awning-tricycle |
| 2 | Car | 7 | Bus |
| 3 | Van | 8 | Motorcycle |
| 4 | Truck | 9 | Bicycle |

*Person with hard hat/safety gear

### Key Characteristics

- **10,000+ aerial videos** captured from various altitudes and angles
- **Extreme scale variation** (1×1 to 550×550 pixels)
- **High occlusion and truncation** in real-world scenarios
- **Class imbalance** (pedestrians underrepresented)
- **Challenging viewpoints** from aerial perspective

---

## 📁 Project Structure

```
drone-human-detection/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── .venv/                            # Virtual environment (not in repo)
│
├── scripts/
│   ├── visdrone_analysis.py          # Basic dataset structure analysis
│   ├── task_01_dataset_analysis.py   # Comprehensive Task-01 analysis
│   ├── task_02_preprocessing.py      # (Coming) Data preprocessing pipeline
│   ├── task_03_training.py           # (Coming) Model training
│   └── task_04_evaluation.py         # (Coming) Model evaluation & inference
│
├── models/
│   ├── yolov8_detector.py            # (Coming) YOLOv8 model wrapper
│   ├── custom_backbone.py            # (Coming) Custom detection architectures
│   └── loss_functions.py             # (Coming) Custom loss functions
│
├── VisDrone_Dataset/
│   ├── visdrone.yaml                 # Dataset configuration
│   ├── VisDrone2019-DET-train/
│   │   ├── images/                   # 6,471 training images
│   │   └── labels/                   # 6,471 annotation files
│   ├── VisDrone2019-DET-val/
│   │   ├── images/                   # 548 validation images
│   │   └── labels/                   # 548 annotation files
│   ├── VisDrone2019-DET-test-dev/
│   │   ├── images/                   # 1,610 test-dev images
│   │   └── labels/                   # 1,610 annotation files (for evaluation)
│   └── VisDrone2019-DET-test-challenge/
│       └── images/                   # 2,540 challenge test images
│
└── outputs/
    ├── task_01_analysis/
    │   ├── 01_dataset_overview.png           # Dataset statistics
    │   ├── 02_sample_annotations.png         # Annotated samples
    │   ├── 03_challenges_summary.png         # Challenge visualization
    │   ├── task_01_report.txt               # Detailed analysis report
    │   └── dataset_statistics.json          # Statistics in JSON format
    │
    ├── task_02_processed_data/               # (Coming)
    ├── task_03_models/                       # (Coming)
    └── task_04_results/                      # (Coming)
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster training
- **Storage**: ~10-15 GB for full dataset + models + outputs
- **macOS/Linux/Windows**: All platforms supported

### Step 1: Clone Repository

```bash
cd ~/projects
git clone <repository-url>
cd drone-human-detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install required packages
pip install -r requirements.txt
```

### Step 4: Download Dataset

The dataset needs to be downloaded manually from Kaggle due to license restrictions.

```bash
# Option 1: Download manually from Kaggle
# 1. Visit: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset
# 2. Click "Download" button
# 3. Extract to: drone-human-detection/VisDrone_Dataset/

# Option 2: Download using Kaggle CLI (requires API credentials)
# First, set up your Kaggle API credentials
kaggle datasets download -d banuprasadb/visdrone-dataset
unzip visdrone-dataset.zip -d VisDrone_Dataset/
```

### Step 5: Verify Installation

```bash
# Navigate to scripts directory
cd scripts

# Run basic analysis to verify setup
python3 visdrone_analysis.py

# You should see output similar to:
# VisDrone2019-DET-train -> 6471
# VisDrone2019-DET-val -> 548
# VisDrone2019-DET-test-dev -> 1610
# ...
```

---

## ⚡ Quick Start

### Running Task-01: Dataset Analysis

```bash
# Activate virtual environment
source .venv/bin/activate

# Navigate to scripts
cd scripts

# Run comprehensive dataset analysis
python3 task_01_dataset_analysis.py

# Output files will be saved to:
# ../outputs/task_01_analysis/
```

This generates:
- 📊 Dataset overview visualization
- 🖼️ Sample annotated images
- ⚠️ Challenge severity assessment
- 📄 Comprehensive analysis report

---

## 📚 Tasks Overview

### ✅ Task-01: Dataset Understanding & Preprocessing

**Status**: ✓ Completed

Comprehensive analysis of the VisDrone dataset including:
- Dataset structure and statistics
- Object distribution analysis
- Annotation format explanation
- Challenge identification and mitigation strategies
- Visualization of dataset characteristics

**Output**: `outputs/task_01_analysis/`

**Key Findings**:
- 10,169 images with object annotations
- 10 object categories with significant class imbalance
- Extreme scale variation (1×1 to 550×550 pixels)
- 30%+ small objects requiring special handling
- 20%+ occlusion affecting detection accuracy

---

### 🔄 Task-02: Data Preprocessing & Augmentation

**Status**: ⏳ Coming Soon

Preprocessing pipeline including:
- Annotation format conversion (VisDrone → YOLO)
- Image normalization and resizing
- Data augmentation strategies
- Train/Val/Test split with stratification
- Class weight calculation

**Expected Output**: `outputs/task_02_processed_data/`

---

### 🤖 Task-03: Model Training

**Status**: ⏳ Coming Soon

Model training implementation:
- YOLOv8 model setup and configuration
- Custom loss functions for imbalanced data
- Training loop with validation
- Hyperparameter optimization
- Real-time monitoring with TensorBoard/Weights & Biases

**Expected Output**: `outputs/task_03_models/`

---

### 📈 Task-04: Evaluation & Inference

**Status**: ⏳ Coming Soon

Model evaluation and deployment:
- Performance metrics calculation (mAP, Precision, Recall)
- Error analysis and visualization
- Inference pipeline for real-time detection
- Result visualization and reporting
- Model deployment and optimization

**Expected Output**: `outputs/task_04_results/`

---

## 📦 Requirements

### Core Dependencies

```
opencv-python>=4.5.0          # Computer vision processing
numpy>=1.20.0                 # Numerical computations
matplotlib>=3.3.0             # Data visualization
Pillow>=8.0.0                 # Image processing
torch>=1.9.0                  # Deep learning framework (CPU/GPU)
torchvision>=0.10.0           # PyTorch vision utilities
ultralytics>=8.0.0            # YOLOv8 implementation (coming in Task-03)
scikit-learn>=0.24.0          # Machine learning utilities
pandas>=1.1.0                 # Data manipulation
```

### Optional Dependencies

```
tensorboard>=2.4.0            # Training visualization
wandb>=0.11.0                 # Experiment tracking
jupyter>=1.0.0                # Notebook environment
```

### Install All (Including Optional)

```bash
pip install -r requirements.txt
# For optional packages:
pip install tensorboard wandb jupyter
```

---

## 🔧 Configuration

### Dataset Configuration (visdrone.yaml)

```yaml
path: /Users/mdibrahimkhalil/projects/drone-human-detection/VisDrone_Dataset
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
test: VisDrone2019-DET-test-dev/images

nc: 10
names: ['Pedestrian', 'Person', 'Car', 'Van', 'Truck', 
        'Tricycle', 'Awning-tricycle', 'Bus', 'Motorcycle', 'Bicycle']
```

---

## 📝 Usage Examples

### Run Dataset Analysis

```bash
cd scripts
source ../.venv/bin/activate
python3 task_01_dataset_analysis.py
```

### View Analysis Results

```bash
# Open the generated report
open ../outputs/task_01_analysis/task_01_report.txt

# View visualizations
open ../outputs/task_01_analysis/01_dataset_overview.png
open ../outputs/task_01_analysis/02_sample_annotations.png
open ../outputs/task_01_analysis/03_challenges_summary.png
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cv2'`

**Solution**: Install opencv-python in your virtual environment

```bash
source .venv/bin/activate
pip install opencv-python
```

### Issue: `FileNotFoundError: VisDrone_Dataset not found`

**Solution**: Download and extract the dataset to the correct location

```bash
# Ensure you have:
# drone-human-detection/VisDrone_Dataset/
#   ├── VisDrone2019-DET-train/
#   ├── VisDrone2019-DET-val/
#   ├── VisDrone2019-DET-test-dev/
#   └── VisDrone2019-DET-test-challenge/
```

### Issue: Out of Memory (OOM) errors

**Solution**: Reduce batch size or image resolution in configuration files

```bash
# Modify the batch size in training scripts
BATCH_SIZE = 8  # Reduce from default 32
```

---

## 📊 Performance Benchmarks

### Expected Performance (Task-03/04)

| Model | mAP@0.5 | Precision | Recall | FPS (GPU) |
|-------|---------|-----------|--------|-----------|
| YOLOv8n | ~0.45 | 0.52 | 0.48 | 120 |
| YOLOv8s | ~0.52 | 0.58 | 0.55 | 90 |
| YOLOv8m | ~0.58 | 0.63 | 0.61 | 60 |

*Benchmarks are approximate and will be updated after model training*

---

## 📚 References

- **VisDrone Dataset**: [Official Website](http://www.aiskyeye.com/)
- **YOLOv8**: [GitHub Repository](https://github.com/ultralytics/yolov8)
- **Object Detection**: [arXiv Papers](https://arxiv.org/list/cs.CV/recent)
- **Drone-based Detection**: [Survey & Benchmarks](https://arxiv.org/search/?query=drone+object+detection)

---

## 👤 Author

**Intern Project** - Drone-Based Human Detection System

**Mentor**: [Your Name/Organization]  
**Start Date**: May 14, 2026  
**Duration**: Multi-phase project (Task-01 → Task-04)

---

## 📄 License

This project is provided for educational and research purposes.

**Dataset License**: VisDrone dataset is available for research and educational purposes. Please refer to the [official VisDrone website](http://www.aiskyeye.com/) for licensing details.

---

## 🤝 Contributing

For improvements and suggestions:

1. Create a new branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m 'Add your feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Open a Pull Request

---

## 📞 Support

For issues, questions, or suggestions:

- 📧 Email: [your-email@example.com]
- 💬 GitHub Issues: [Link to issues]
- 📖 Documentation: See `/docs/` folder (coming soon)

---

## 🗂️ .gitignore Configuration

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Virtual Environment
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project-specific
VisDrone_Dataset/          # Large dataset files
models/*.pth               # Large model weights
outputs/                   # Generated outputs (optional)
*.log                      # Log files

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Credentials
*.key
*.pem
.kaggle/
```

---

**Last Updated**: May 14, 2026  
**Version**: 1.0.0 (Task-01 Complete)

---

