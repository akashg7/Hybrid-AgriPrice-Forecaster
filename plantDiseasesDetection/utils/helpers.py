"""
helpers.py - Utility functions for the crop disease detection project.

Provides reproducibility setup, image display utilities, and directory helpers.
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Deterministic operations (may reduce performance)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"[INFO] Random seeds set to {seed} for reproducibility.")


def display_sample_images(image_dir: str, class_names: list, samples_per_class: int = 3,
                          figsize: tuple = (15, 10)) -> None:
    """Display sample images from each class for visual inspection.
    
    Args:
        image_dir: Root directory containing class subdirectories.
        class_names: List of class names (subdirectory names).
        samples_per_class: Number of images to show per class.
        figsize: Figure size for matplotlib.
    """
    n_classes = min(len(class_names), 6)  # Show max 6 classes
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=figsize)
    
    for i, cls in enumerate(class_names[:n_classes]):
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        images = os.listdir(cls_dir)[:samples_per_class]
        for j, img_name in enumerate(images):
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i, j].imshow(img)
                axes[i, j].set_title(cls[:25], fontsize=8)
                axes[i, j].axis('off')
            except Exception:
                axes[i, j].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[i, j].axis('off')
    
    plt.suptitle('Sample Images by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()


def get_class_distribution(data_dir: str) -> dict:
    """Count images per class in a directory structure.
    
    Args:
        data_dir: Root directory with class subdirectories.
    
    Returns:
        Dictionary mapping class names to image counts.
    """
    distribution = {}
    if not os.path.isdir(data_dir):
        return distribution
    
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ])
            distribution[class_name] = count
    
    return distribution


def get_image_dimensions(data_dir: str, sample_size: int = 500) -> list:
    """Sample image dimensions from the dataset for analysis.
    
    Args:
        data_dir: Root directory with class subdirectories.
        sample_size: Number of images to sample.
    
    Returns:
        List of (width, height) tuples.
    """
    all_images = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_images.append(os.path.join(class_path, img_name))
    
    # Random sample
    sampled = random.sample(all_images, min(sample_size, len(all_images)))
    dimensions = []
    for img_path in sampled:
        try:
            with Image.open(img_path) as img:
                dimensions.append(img.size)  # (width, height)
        except Exception:
            continue
    
    return dimensions


def count_total_images(data_dir: str) -> int:
    """Count total number of images in a dataset directory.
    
    Args:
        data_dir: Root directory with class subdirectories.
    
    Returns:
        Total image count.
    """
    total = 0
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            total += len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ])
    return total
