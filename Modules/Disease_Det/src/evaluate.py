"""
evaluate.py - Evaluation metrics and visualization for crop disease detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from PIL import Image
from collections import Counter


def plot_training_curves(history, save_dir='outputs'):
    """Plot training/validation accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_gen, class_names, save_dir='outputs'):
    """Full evaluation: accuracy, precision, recall, F1, classification report."""
    os.makedirs(save_dir, exist_ok=True)
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes[:len(y_pred)]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}\n  TEST RESULTS\n{'='*50}")
    print(f"  Accuracy:  {acc:.4f}\n  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}\n  F1 Score:  {f1:.4f}\n{'='*50}\n")

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n\n{report}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1_score': f1, 'y_true': y_true, 'y_pred': y_pred, 'y_pred_probs': y_pred_probs}


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='outputs', figsize=(20, 16)):
    """Plot and save confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right', fontsize=7); plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()


def analyze_misclassifications(model, test_gen, class_names,
                                y_true=None, y_pred=None,
                                num_examples=9, save_dir='outputs'):
    """Visualize misclassified images and identify top confusion pairs."""
    os.makedirs(save_dir, exist_ok=True)
    if y_true is None or y_pred is None:
        test_gen.reset()
        y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
        y_true = test_gen.classes[:len(y_pred)]

    misclassified = np.where(y_true != y_pred)[0]
    print(f"[INFO] Misclassified: {len(misclassified)}/{len(y_true)} ({len(misclassified)/len(y_true)*100:.1f}%)")
    if len(misclassified) == 0:
        return

    sample_idx = np.random.choice(misclassified, min(num_examples, len(misclassified)), replace=False)
    filepaths = test_gen.filepaths
    cols, rows = 3, (num_examples + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= len(sample_idx):
            ax.axis('off'); continue
        idx = sample_idx[i]
        try:
            img = Image.open(filepaths[idx]).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.set_title(f"True: {class_names[y_true[idx]][:25]}\nPred: {class_names[y_pred[idx]][:25]}",
                     fontsize=8, color='red', fontweight='bold')
        ax.axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold', color='darkred')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'misclassified_examples.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Top confusion pairs
    pairs = Counter()
    for i in misclassified:
        pairs[(class_names[y_true[i]], class_names[y_pred[i]])] += 1
    print("\nTop 10 Confusion Pairs:")
    for (t, p), c in pairs.most_common(10):
        print(f"  {t} → {p}: {c} times")
