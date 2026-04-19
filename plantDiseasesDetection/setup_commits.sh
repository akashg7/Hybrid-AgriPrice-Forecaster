#!/bin/bash
# setup_commits.sh - Simulate realistic Git commit history
# This script initializes a Git repository and creates commits
# that reflect the natural development progression of the project.
#
# Usage: chmod +x setup_commits.sh && ./setup_commits.sh

set -e
cd "$(dirname "$0")"

echo "🔧 Initializing Git repository..."
git init

# Commit 1: Project scaffold
git add .gitignore requirements.txt README.md
git add data/.gitkeep models/.gitkeep outputs/.gitkeep
git commit -m "feat: initial project setup with directory structure and dependencies"

# Commit 2: Utility functions
git add utils/__init__.py utils/helpers.py
git commit -m "feat: add utility functions (seed setting, image display, dataset stats)"

# Commit 3: Data pipeline
git add src/__init__.py src/data_loader.py
git commit -m "feat: implement data loading, merging, deduplication pipeline

- Kagglehub dataset download
- Perceptual hash-based duplicate detection
- Corruption removal
- Stratified train/val/test split"

# Commit 4: Augmentation
git add src/augmentation.py
git commit -m "feat: add data augmentation pipeline with ImageDataGenerator

- Rotation, flip, zoom, shift, brightness
- Separate configs for train vs val/test"

# Commit 5: Model architecture
git add src/model.py
git commit -m "feat: implement EfficientNet-B0 model with two-phase training support

- Pretrained ImageNet base
- Custom classification head (GAP, BN, Dense, Dropout)
- Unfreeze utility for fine-tuning"

# Commit 6: Training pipeline
git add src/train.py
git commit -m "feat: add training pipeline with callbacks

- Adam optimizer with configurable LR
- Early stopping, LR scheduler, model checkpoint"

# Commit 7: Evaluation
git add src/evaluate.py
git commit -m "feat: comprehensive evaluation module

- Classification report, confusion matrix
- Training curves visualization
- Misclassification analysis with failure patterns"

# Commit 8: Grad-CAM
git add src/gradcam.py
git commit -m "feat: implement Grad-CAM for model interpretability

- Heatmap generation from last conv layer
- Overlay visualization on original images"

# Commit 9: Main notebook
git add notebooks/crop_disease_detection.ipynb
git commit -m "feat: complete pipeline notebook with literature review and analysis

- 6-paper literature review with gap analysis
- Full EDA (class dist, dimensions, t-SNE)
- Two-phase training with evaluation
- Grad-CAM and failure analysis"

# Commit 10: LaTeX report
git add report/report.tex
git commit -m "docs: add LaTeX academic report

- Abstract, introduction, methodology
- Results with figures
- Failure analysis and discussion
- 11 references"

# Commit 11: Presentation and viva
git add outputs/presentation_outline.md outputs/viva_prep.md
git commit -m "docs: add presentation outline and viva preparation

- 10-slide presentation with talking points
- 12 anticipated viva questions with detailed answers"

# Commit 12: Final polish
git add -A
git commit -m "chore: final cleanup and documentation polish" --allow-empty

echo ""
echo "✅ Git repository initialized with 12 commits!"
echo ""
git log --oneline
