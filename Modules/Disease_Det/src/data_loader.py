"""
data_loader.py - Dataset loading, merging, deduplication, and cleaning.

Handles downloading datasets via kagglehub, merging PlantVillage and
New Plant Diseases datasets, detecting duplicates using perceptual hashing,
and removing corrupted images to prevent data leakage.
"""

import os
import shutil
import hashlib
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    imagehash = None
    print("[WARNING] imagehash not installed. Duplicate detection will use MD5 only.")


# ──────────────────────────────────────────────────────────────
# 1. Dataset Download
# ──────────────────────────────────────────────────────────────

def download_datasets() -> Tuple[str, str]:
    """Download both datasets using kagglehub.
    
    Returns:
        Tuple of (new_plant_diseases_path, plantvillage_path).
    """
    import kagglehub
    
    print("[INFO] Downloading 'vipoooool/new-plant-diseases-dataset'...")
    ds1_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print(f"  → Downloaded to: {ds1_path}")
    
    print("[INFO] Downloading 'abdallahalidev/plantvillage-dataset'...")
    ds2_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print(f"  → Downloaded to: {ds2_path}")
    
    return ds1_path, ds2_path


def find_image_root(base_path: str) -> str:
    """Recursively find the directory containing class subdirectories with images.
    
    Args:
        base_path: Root path to search from.
    
    Returns:
        Path to the directory containing class folders.
    """
    # Check if current directory has subdirectories with images
    if not os.path.isdir(base_path):
        return base_path
    
    subdirs = [d for d in os.listdir(base_path) 
               if os.path.isdir(os.path.join(base_path, d))]
    
    if not subdirs:
        return base_path
    
    # Check if subdirectories contain images (class folders)
    sample_dir = os.path.join(base_path, subdirs[0])
    has_images = any(
        f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))
    )
    
    if has_images:
        return base_path
    
    # Recurse into subdirectories
    for subdir in subdirs:
        result = find_image_root(os.path.join(base_path, subdir))
        if result != os.path.join(base_path, subdir):
            return result
        # Check this subdir
        sub_subdirs = [d for d in os.listdir(os.path.join(base_path, subdir))
                       if os.path.isdir(os.path.join(base_path, subdir, d))]
        if sub_subdirs:
            sample = os.path.join(base_path, subdir, sub_subdirs[0])
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                   for f in os.listdir(sample) if os.path.isfile(os.path.join(sample, f))):
                return os.path.join(base_path, subdir)
    
    return base_path


# ──────────────────────────────────────────────────────────────
# 2. Class Name Normalization
# ──────────────────────────────────────────────────────────────

def normalize_class_name(name: str) -> str:
    """Normalize class directory names for consistent merging.
    
    Strips whitespace, converts to lowercase with underscores,
    and standardizes common naming variations.
    
    Args:
        name: Original class directory name.
    
    Returns:
        Normalized class name string.
    """
    normalized = name.strip()
    normalized = normalized.replace('___', '_').replace('__', '_')
    normalized = normalized.replace(' ', '_')
    # Remove trailing/leading underscores
    normalized = normalized.strip('_')
    return normalized


def get_class_mapping(dir1: str, dir2: str) -> Dict[str, str]:
    """Create mapping from original class names to normalized names.
    
    Args:
        dir1: First dataset directory.
        dir2: Second dataset directory.
    
    Returns:
        Dictionary mapping original names to normalized names.
    """
    mapping = {}
    for d in [dir1, dir2]:
        if not os.path.isdir(d):
            continue
        for cls in os.listdir(d):
            if os.path.isdir(os.path.join(d, cls)):
                mapping[cls] = normalize_class_name(cls)
    return mapping


# ──────────────────────────────────────────────────────────────
# 3. Duplicate Detection
# ──────────────────────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file contents.
    
    Args:
        filepath: Path to image file.
    
    Returns:
        Hex digest of MD5 hash.
    """
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_perceptual_hash(filepath: str) -> str:
    """Compute perceptual hash using average hashing.
    
    Perceptual hashes are robust to minor resizing and compression,
    making them ideal for detecting near-duplicate images across datasets.
    
    Args:
        filepath: Path to image file.
    
    Returns:
        String representation of perceptual hash.
    """
    if imagehash is None:
        return compute_file_hash(filepath)
    
    try:
        img = Image.open(filepath).convert('RGB')
        return str(imagehash.average_hash(img, hash_size=16))
    except Exception:
        return compute_file_hash(filepath)


def detect_duplicates(data_dir: str, use_perceptual: bool = True) -> List[str]:
    """Detect duplicate images in a dataset directory.
    
    Uses perceptual hashing (if available) to detect near-duplicates
    that may exist across merged datasets, preventing data leakage.
    
    Args:
        data_dir: Root directory containing class subdirectories.
        use_perceptual: Whether to use perceptual hashing (recommended).
    
    Returns:
        List of duplicate file paths to remove.
    """
    hash_fn = compute_perceptual_hash if use_perceptual else compute_file_hash
    hash_to_files = defaultdict(list)
    duplicates = []
    
    print("[INFO] Computing image hashes for duplicate detection...")
    all_images = []
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(cls_path, img_name))
    
    for img_path in tqdm(all_images, desc="Hashing images"):
        h = hash_fn(img_path)
        hash_to_files[h].append(img_path)
    
    # Collect duplicates (keep first occurrence, mark rest)
    for h, files in hash_to_files.items():
        if len(files) > 1:
            duplicates.extend(files[1:])  # Keep first, remove rest
    
    print(f"[INFO] Found {len(duplicates)} duplicate images.")
    return duplicates


# ──────────────────────────────────────────────────────────────
# 4. Corruption Detection
# ──────────────────────────────────────────────────────────────

def detect_corrupted_images(data_dir: str) -> List[str]:
    """Identify corrupted or unreadable images.
    
    Attempts to open and verify each image. Images that fail
    are flagged for removal.
    
    Args:
        data_dir: Root directory containing class subdirectories.
    
    Returns:
        List of corrupted file paths.
    """
    corrupted = []
    print("[INFO] Scanning for corrupted images...")
    
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                corrupted.append(img_path)
    
    print(f"[INFO] Found {len(corrupted)} corrupted images.")
    return corrupted


# ──────────────────────────────────────────────────────────────
# 5. Dataset Merging
# ──────────────────────────────────────────────────────────────

def merge_datasets(ds1_root: str, ds2_root: str, output_dir: str,
                   remove_duplicates: bool = True) -> Dict[str, int]:
    """Merge two datasets into a unified directory with deduplication.
    
    Strategy:
    1. Find image roots in both datasets
    2. Normalize class names
    3. Copy images to output directory
    4. Detect and remove duplicates (prevents data leakage)
    5. Remove corrupted images
    
    Args:
        ds1_root: Path to first dataset (new-plant-diseases).
        ds2_root: Path to second dataset (plantvillage).
        output_dir: Output directory for merged dataset.
        remove_duplicates: Whether to run deduplication.
    
    Returns:
        Dictionary of class name to image count after merging.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find actual image directories
    ds1_img = find_image_root(ds1_root)
    ds2_img = find_image_root(ds2_root)
    
    print(f"[INFO] Dataset 1 image root: {ds1_img}")
    print(f"[INFO] Dataset 2 image root: {ds2_img}")
    
    # Copy images from both datasets
    copy_count = 0
    for ds_path, ds_label in [(ds1_img, "ds1"), (ds2_img, "ds2")]:
        if not os.path.isdir(ds_path):
            print(f"[WARNING] {ds_path} is not a valid directory, skipping.")
            continue
        for cls in os.listdir(ds_path):
            cls_src = os.path.join(ds_path, cls)
            if not os.path.isdir(cls_src):
                continue
            norm_cls = normalize_class_name(cls)
            cls_dst = os.path.join(output_dir, norm_cls)
            os.makedirs(cls_dst, exist_ok=True)
            
            for img_name in os.listdir(cls_src):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src = os.path.join(cls_src, img_name)
                    # Prefix with dataset label to avoid filename collisions
                    dst = os.path.join(cls_dst, f"{ds_label}_{img_name}")
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copy_count += 1
    
    print(f"[INFO] Copied {copy_count} images to merged directory.")
    
    # Remove duplicates
    if remove_duplicates:
        duplicates = detect_duplicates(output_dir)
        for dup in duplicates:
            os.remove(dup)
        print(f"[INFO] Removed {len(duplicates)} duplicate images.")
    
    # Remove corrupted images
    corrupted = detect_corrupted_images(output_dir)
    for c in corrupted:
        os.remove(c)
    print(f"[INFO] Removed {len(corrupted)} corrupted images.")
    
    # Final counts
    class_counts = {}
    for cls in sorted(os.listdir(output_dir)):
        cls_path = os.path.join(output_dir, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[cls] = count
    
    total = sum(class_counts.values())
    print(f"[INFO] Merged dataset: {len(class_counts)} classes, {total} images total.")
    return class_counts


# ──────────────────────────────────────────────────────────────
# 6. Train/Val/Test Split
# ──────────────────────────────────────────────────────────────

def split_dataset(data_dir: str, output_base: str,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[str, str, str]:
    """Split merged dataset into train/validation/test sets.
    
    Performs stratified splitting to maintain class proportions.
    
    Args:
        data_dir: Merged dataset directory.
        output_base: Base directory for split outputs.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.
    
    Returns:
        Tuple of (train_dir, val_dir, test_dir) paths.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    
    train_dir = os.path.join(output_base, 'train')
    val_dir = os.path.join(output_base, 'val')
    test_dir = os.path.join(output_base, 'test')
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    print("[INFO] Splitting dataset into train/val/test...")
    for cls in tqdm(sorted(os.listdir(data_dir)), desc="Splitting classes"):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        np.random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            train_dir: images[:n_train],
            val_dir: images[n_train:n_train + n_val],
            test_dir: images[n_train + n_val:]
        }
        
        for split_dir, split_images in splits.items():
            cls_split_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_split_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(cls_split_dir, img_name)
                shutil.copy2(src, dst)
    
    print(f"[INFO] Split complete:")
    for label, d in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        total = sum(
            len(os.listdir(os.path.join(d, c)))
            for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))
        )
        print(f"  {label}: {total} images")
    
    return train_dir, val_dir, test_dir
