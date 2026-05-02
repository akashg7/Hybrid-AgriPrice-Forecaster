"""
train.py - Training pipeline with callbacks for crop disease detection.

Implements a training loop with:
- EarlyStopping: Prevents overfitting by monitoring validation loss
- ReduceLROnPlateau: Adaptively reduces learning rate when progress stalls
- ModelCheckpoint: Saves best model based on validation accuracy
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam


def compile_model(model, learning_rate: float = 1e-3):
    """Compile model with Adam optimizer and categorical crossentropy.
    
    Adam chosen because:
    - Adaptive learning rates per-parameter
    - Momentum + RMSProp combination
    - Works well with sparse gradients (common in image tasks)
    - Default β1=0.9, β2=0.999 are robust choices
    
    Args:
        model: Keras Model to compile.
        learning_rate: Initial learning rate.
    
    Returns:
        Compiled model.
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    print(f"[INFO] Model compiled with Adam (lr={learning_rate})")
    return model


def get_callbacks(model_save_path: str = 'models/best_model.keras',
                  patience_es: int = 5,
                  patience_lr: int = 3) -> list:
    """Create training callbacks.
    
    Callbacks strategy:
    1. EarlyStopping (patience=5): Stop if val_loss doesn't improve for 5 epochs.
       Restores best weights automatically.
    2. ReduceLROnPlateau (patience=3, factor=0.5): Halve LR after 3 epochs of stagnation.
       Allows model to escape local minima with smaller steps.
    3. ModelCheckpoint: Save best model by val_accuracy.
    
    Args:
        model_save_path: Path to save best model weights.
        patience_es: Patience for early stopping.
        patience_lr: Patience for LR reduction.
    
    Returns:
        List of Keras callbacks.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]
    
    print("[INFO] Callbacks configured: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
    return callbacks


def train_model(model, train_gen, val_gen, epochs: int = 15,
                model_save_path: str = 'models/best_model.keras'):
    """Train the model with configured callbacks.
    
    Args:
        model: Compiled Keras model.
        train_gen: Training data generator.
        val_gen: Validation data generator.
        epochs: Maximum number of training epochs.
        model_save_path: Path to save best model.
    
    Returns:
        Training history object.
    """
    callbacks = get_callbacks(model_save_path)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING STARTED")
    print(f"  Epochs: {epochs} | Batch size: {train_gen.batch_size}")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    
    print(f"\n[INFO] Training complete. Best model saved to: {model_save_path}")
    return history
