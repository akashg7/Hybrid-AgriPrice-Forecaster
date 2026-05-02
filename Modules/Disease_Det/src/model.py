"""
model.py - EfficientNet-B0 model architecture for crop disease classification.

Implements transfer learning with a two-phase training strategy:
Phase 1: Freeze base, train custom classification head
Phase 2: Unfreeze top layers, fine-tune at reduced learning rate

EfficientNet-B0 is chosen over ResNet50 because:
1. Compound scaling (depth + width + resolution) yields better accuracy/FLOPs
2. 5.3M parameters vs ResNet50's 25.6M — more efficient
3. State-of-the-art ImageNet accuracy at comparable compute
(Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3),
                dropout_rate: float = 0.3, freeze_base: bool = True) -> Model:
    """Build EfficientNet-B0 with custom classification head.
    
    Architecture:
        EfficientNet-B0 (pretrained on ImageNet)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu)
        → Dropout(0.3)
        → Dense(num_classes, softmax)
    
    Args:
        num_classes: Number of output classes.
        input_shape: Input image dimensions (H, W, C).
        dropout_rate: Dropout rate for regularization.
        freeze_base: Whether to freeze the base model weights.
    
    Returns:
        Compiled Keras Model.
    """
    # Load pretrained EfficientNet-B0 (exclude top classification layer)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )
    
    # Freeze base model for Phase 1 training
    base_model.trainable = not freeze_base
    
    # Custom classification head
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # EfficientNet-B0 forward pass
    x = base_model(inputs, training=False)
    
    # Global Average Pooling reduces spatial dimensions
    # Preferred over Flatten to reduce parameters and prevent overfitting
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Batch Normalization for stable training
    x = layers.BatchNormalization(name='batch_norm')(x)
    
    # Dense layer for feature transformation
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    
    # Dropout for regularization (prevents co-adaptation of neurons)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Output layer with softmax for multi-class classification
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CropDiseaseDetector')
    
    print(f"[INFO] Model built: {model.count_params():,} total parameters")
    print(f"[INFO] Base model frozen: {freeze_base}")
    print(f"[INFO] Output classes: {num_classes}")
    
    return model


def unfreeze_model(model: Model, num_layers_to_unfreeze: int = 20,
                   learning_rate: float = 1e-5) -> Model:
    """Unfreeze top layers of base model for fine-tuning (Phase 2).
    
    Fine-tuning strategy:
    - Unfreeze only the top N layers to avoid catastrophic forgetting
    - Use a very low learning rate (1e-5) to make small updates
    - Bottom layers retain generic features (edges, textures)
    - Top layers are adapted to domain-specific features (leaf patterns)
    
    Args:
        model: Trained model from Phase 1.
        num_layers_to_unfreeze: Number of base model layers to unfreeze.
        learning_rate: Reduced learning rate for fine-tuning.
    
    Returns:
        Recompiled model ready for Phase 2 training.
    """
    # Access the base model (first layer in our architecture)
    base_model = model.layers[1]  # EfficientNetB0 layer
    base_model.trainable = True
    
    # Freeze all layers except the top N
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    trainable = sum(1 for layer in model.layers if layer.trainable)
    total = len(model.layers)
    print(f"[INFO] Fine-tuning mode: {trainable}/{total} layers trainable")
    print(f"[INFO] Learning rate reduced to {learning_rate}")
    
    return model


def get_model_summary(model: Model) -> str:
    """Get model summary as a string.
    
    Args:
        model: Keras Model.
    
    Returns:
        String representation of model summary.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)
