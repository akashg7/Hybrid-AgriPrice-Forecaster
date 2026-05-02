"""
gradcam.py - Gradient-weighted Class Activation Mapping (Grad-CAM).

Implements Grad-CAM (Selvaraju et al., 2017) to visualize which regions
of an input image the model focuses on for its prediction. This provides
interpretability and trust in the model's decision-making process.

Theory:
- Compute gradients of the predicted class score w.r.t. the last conv layer
- Global average pool the gradients to get importance weights per channel
- Weighted combination of feature maps produces a heatmap
- Heatmap highlights discriminative regions (e.g., diseased leaf areas)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def get_last_conv_layer_name(model):
    """Find the name of the last convolutional layer in the model.

    Args:
        model: Keras model.

    Returns:
        Name of the last Conv2D layer.
    """
    # For our architecture, the base model is the second layer
    base = model.layers[1]  # EfficientNetB0
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Fallback: search in full model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image.

    Args:
        img_array: Preprocessed image array (1, H, W, 3).
        model: Trained Keras model.
        last_conv_layer_name: Name of the target conv layer.
        pred_index: Class index to explain (None = predicted class).

    Returns:
        Numpy array heatmap of shape (H, W).
    """
    # Build a sub-model that outputs both conv layer output and predictions
    base = model.layers[1]
    conv_layer = base.get_layer(last_conv_layer_name)

    # Create gradient model
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4, target_size=(224, 224)):
    """Overlay Grad-CAM heatmap on original image.

    Args:
        img_path: Path to original image.
        heatmap: Grad-CAM heatmap array.
        alpha: Overlay transparency.
        target_size: Image resize target.

    Returns:
        Superimposed image as numpy array.
    """
    img = Image.open(img_path).convert('RGB').resize(target_size)
    img_array = np.array(img)

    # Resize heatmap to image dimensions
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(
        Image.fromarray(heatmap_resized).resize(target_size)
    )

    # Apply colormap
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Superimpose
    superimposed = (jet_heatmap * alpha + img_array * (1 - alpha)).astype(np.uint8)
    return superimposed


def visualize_gradcam(model, img_paths, class_names, preprocess_fn=None,
                      save_dir='outputs', num_images=6):
    """Generate and display Grad-CAM visualizations for multiple images.

    Args:
        model: Trained model.
        img_paths: List of image file paths.
        class_names: List of class names.
        preprocess_fn: Optional preprocessing function.
        save_dir: Output directory.
        num_images: Number of images to visualize.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    last_conv = get_last_conv_layer_name(model)
    print(f"[INFO] Grad-CAM target layer: {last_conv}")

    cols = 3
    rows = min(num_images, len(img_paths))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes]

    for i in range(min(num_images, len(img_paths))):
        img_path = img_paths[i]

        # Load and preprocess
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_input, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_class = class_names[pred_idx]
        confidence = preds[0][pred_idx]

        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_input, model, last_conv, pred_idx)
        superimposed = overlay_gradcam(img_path, heatmap)

        # Plot: Original | Heatmap | Overlay
        axes[i][0].imshow(img)
        axes[i][0].set_title('Original', fontsize=10)
        axes[i][0].axis('off')

        axes[i][1].imshow(heatmap, cmap='jet')
        axes[i][1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[i][1].axis('off')

        axes[i][2].imshow(superimposed)
        axes[i][2].set_title(f'{pred_class}\n({confidence:.2%})', fontsize=9)
        axes[i][2].axis('off')

    plt.suptitle('Grad-CAM: Model Attention Visualization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gradcam_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Grad-CAM saved to {save_path}")
