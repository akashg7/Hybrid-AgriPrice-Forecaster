# 🌿 Crop Disease Detection Using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12+](https://img.shields.io/badge/tensorflow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automated plant disease detection from leaf images using **EfficientNet-B0** with two-phase transfer learning. Achieves high accuracy across 38 disease classes with model interpretability via Grad-CAM.

---

## 📋 Problem Statement

Plant diseases cause 20–40% annual crop losses globally. Traditional diagnosis by expert pathologists is slow and inaccessible to smallholder farmers. This project builds an automated detection system using deep learning on leaf images.

## 🔬 Approach

1. **Dataset**: Merged two Kaggle datasets (87K+ images, 38 classes) with perceptual hash–based deduplication
2. **Model**: EfficientNet-B0 (pretrained on ImageNet) with custom classification head
3. **Training**: Two-phase transfer learning — frozen feature extraction → fine-tuning top layers
4. **Interpretability**: Grad-CAM visualizations showing model attention on diseased regions
5. **Analysis**: Systematic failure analysis of misclassified images

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | Run notebook for results |
| Precision (weighted) | Run notebook for results |
| Recall (weighted) | Run notebook for results |
| F1 Score (weighted) | Run notebook for results |

## 🏗️ Project Structure

```
plantDiseasesDetection/
├── data/                  # Datasets (downloaded via kagglehub)
├── notebooks/
│   └── crop_disease_detection.ipynb  # Complete pipeline
├── src/
│   ├── data_loader.py     # Data loading, merging, deduplication
│   ├── augmentation.py    # Data augmentation pipeline
│   ├── model.py           # EfficientNet-B0 architecture
│   ├── train.py           # Training with callbacks
│   ├── evaluate.py        # Metrics & visualization
│   └── gradcam.py         # Grad-CAM implementation
├── utils/
│   └── helpers.py         # Utility functions
├── models/                # Saved model weights
├── outputs/               # Generated figures & reports
├── report/
│   └── report.tex         # LaTeX academic report
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### Option 1: Google Colab / Kaggle (Recommended)

1. Upload the `notebooks/crop_disease_detection.ipynb` notebook
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells — datasets are automatically downloaded via kagglehub

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/plantDiseasesDetection.git
cd plantDiseasesDetection

# Install dependencies
pip install -r requirements.txt

# Set up Kaggle API credentials
# Place kaggle.json in ~/.kaggle/

# Run notebook
jupyter notebook notebooks/crop_disease_detection.ipynb
```

## 🔑 Key Features

- **Data Deduplication**: Perceptual hashing prevents data leakage across merged datasets
- **Two-Phase Training**: Feature extraction (frozen base) → Fine-tuning (unfrozen top layers)
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1, confusion matrix, training curves
- **Grad-CAM**: Visual explanations of model predictions
- **Failure Analysis**: Systematic analysis of misclassified images with confusion pair identification

## 📚 References

1. Mohanty et al. (2016) – Using deep learning for image-based plant disease detection
2. Ferentinos (2018) – Deep learning models for plant disease detection
3. Too et al. (2019) – Comparative study of fine-tuning for plant disease identification
4. Tan & Le (2019) – EfficientNet: Rethinking model scaling for CNNs
5. Selvaraju et al. (2017) – Grad-CAM: Visual explanations from deep networks

## 📄 License

This project is for academic purposes. Datasets are sourced from Kaggle under their respective licenses.
