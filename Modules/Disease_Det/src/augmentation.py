"""
augmentation.py - Data augmentation pipeline for crop disease detection.

Configures training augmentation (rotation, flip, zoom, shift, brightness)
and validation/test preprocessing (normalization only). Uses Keras
ImageDataGenerator for memory-efficient on-the-fly augmentation.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ──────────────────────────────────────────────────────────────
# Augmentation Configuration
# ──────────────────────────────────────────────────────────────

# Training augmentation rationale:
# - Rotation: Leaves can appear at any angle in real-world capture
# - Horizontal/Vertical flip: Leaf orientation shouldn't affect disease detection
# - Zoom: Simulates varying camera distances
# - Width/Height shift: Handles off-center leaf positioning
# - Brightness: Accounts for different lighting conditions
# - Rescale 1/255: Normalizes pixel values to [0, 1] for faster convergence

TRAIN_AUGMENTATION = {
    'rescale': 1.0 / 255,
    'rotation_range': 30,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'reflect',
}

# Validation/Test: Only normalization, no augmentation
# This ensures evaluation reflects true model performance
VAL_TEST_CONFIG = {
    'rescale': 1.0 / 255,
}


def get_train_generator(train_dir: str, target_size: tuple = (224, 224),
                        batch_size: int = 32, seed: int = 42):
    """Create augmented training data generator.
    
    Applies comprehensive data augmentation to increase effective
    training set size and improve model generalization.
    
    Args:
        train_dir: Path to training data directory.
        target_size: Image resize dimensions (H, W).
        batch_size: Batch size for training.
        seed: Random seed for reproducibility.
    
    Returns:
        DirectoryIterator yielding augmented batches.
    """
    train_datagen = ImageDataGenerator(**TRAIN_AUGMENTATION)
    
    generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed,
    )
    
    print(f"[INFO] Training generator: {generator.samples} images, "
          f"{generator.num_classes} classes, batch_size={batch_size}")
    return generator


def get_val_generator(val_dir: str, target_size: tuple = (224, 224),
                      batch_size: int = 32, seed: int = 42):
    """Create validation data generator (no augmentation).
    
    Only applies rescaling to ensure unbiased evaluation.
    
    Args:
        val_dir: Path to validation data directory.
        target_size: Image resize dimensions (H, W).
        batch_size: Batch size.
        seed: Random seed.
    
    Returns:
        DirectoryIterator yielding normalized batches.
    """
    val_datagen = ImageDataGenerator(**VAL_TEST_CONFIG)
    
    generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed,
    )
    
    print(f"[INFO] Validation generator: {generator.samples} images, "
          f"{generator.num_classes} classes")
    return generator


def get_test_generator(test_dir: str, target_size: tuple = (224, 224),
                       batch_size: int = 32, seed: int = 42):
    """Create test data generator (no augmentation).
    
    Identical to validation generator. Shuffle is disabled to enable
    proper alignment of predictions with ground truth labels.
    
    Args:
        test_dir: Path to test data directory.
        target_size: Image resize dimensions (H, W).
        batch_size: Batch size.
        seed: Random seed.
    
    Returns:
        DirectoryIterator yielding normalized batches.
    """
    test_datagen = ImageDataGenerator(**VAL_TEST_CONFIG)
    
    generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed,
    )
    
    print(f"[INFO] Test generator: {generator.samples} images, "
          f"{generator.num_classes} classes")
    return generator
