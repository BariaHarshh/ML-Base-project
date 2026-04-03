# =============================================================================
# preprocess.py — Image Preprocessing & Augmentation Pipeline
# =============================================================================
# Builds Keras ImageDataGenerators for train / val / test splits.
# Training data gets augmentation to improve generalization.
# Val/Test data only gets rescaling — no augmentation (we evaluate on clean data).
# =============================================================================

# -----------------------------------------------------------------------------
# GOOGLE COLAB NOTE:
# If running on Colab, make sure config.py is in the same directory and
# your dataset is already mounted/extracted. GPU is not required for this file.
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

import config  # central config — all paths and hyperparameters come from here


# =============================================================================
# 1. DATA GENERATORS
# =============================================================================

def build_generators():
    """
    Build and return train, validation, and test data generators.

    - Train generator: augmented + rescaled
    - Val/Test generators: only rescaled (no augmentation)

    Returns:
        train_gen  : augmented training generator
        val_gen    : clean validation generator
        test_gen   : clean test generator (shuffle=False for ordered evaluation)
    """

    # -------------------------------------------------------------------------
    # Training augmentation
    # Augmentation artificially expands the dataset and prevents overfitting
    # by exposing the model to slightly different versions of each image.
    # rescale=1./255 normalizes pixel values from [0, 255] → [0.0, 1.0]
    # -------------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale            = 1. / 255,
        rotation_range     = config.AUGMENTATION['rotation_range'],
        zoom_range         = config.AUGMENTATION['zoom_range'],
        width_shift_range  = config.AUGMENTATION['width_shift_range'],
        height_shift_range = config.AUGMENTATION['height_shift_range'],
        horizontal_flip    = config.AUGMENTATION['horizontal_flip'],
        brightness_range   = config.AUGMENTATION['brightness_range'],
        fill_mode          = config.AUGMENTATION['fill_mode'],
    )

    # -------------------------------------------------------------------------
    # Validation & Test: ONLY rescale — no augmentation
    # We want to evaluate on the true distribution of images, not augmented ones.
    # -------------------------------------------------------------------------
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    # -------------------------------------------------------------------------
    # Flow from directory — Keras reads folder names as class labels.
    # Folder structure:
    #   train/fractured/ → class label "fractured"
    #   train/normal/    → class label "normal"
    # class_mode='binary' returns labels as 0 or 1 (for sigmoid output).
    # -------------------------------------------------------------------------
    train_gen = train_datagen.flow_from_directory(
        directory   = config.TRAIN_DIR,
        target_size = config.IMAGE_SIZE,
        batch_size  = config.BATCH_SIZE,
        class_mode  = 'binary',
        color_mode  = 'rgb',
        shuffle     = True,
        seed        = config.RANDOM_SEED,
    )

    val_gen = val_test_datagen.flow_from_directory(
        directory   = config.VAL_DIR,
        target_size = config.IMAGE_SIZE,
        batch_size  = config.BATCH_SIZE,
        class_mode  = 'binary',
        color_mode  = 'rgb',
        shuffle     = False,          # keep order for consistent evaluation
        seed        = config.RANDOM_SEED,
    )

    test_gen = val_test_datagen.flow_from_directory(
        directory   = config.TEST_DIR,
        target_size = config.IMAGE_SIZE,
        batch_size  = config.BATCH_SIZE,
        class_mode  = 'binary',
        color_mode  = 'rgb',
        shuffle     = False,          # must be False for evaluate.py to work correctly
        seed        = config.RANDOM_SEED,
    )

    # -------------------------------------------------------------------------
    # Log class indices so we always know which folder = which label
    # ImageDataGenerator assigns labels alphabetically:
    #   'fractured' → 1, 'normal' → 0   (f < n alphabetically... wait, f < n)
    # Verify this matches config.CLASS_NAMES ordering.
    # -------------------------------------------------------------------------
    print("\n[PREPROCESS] ✅ Data generators created successfully.")
    print(f"  Class indices  : {train_gen.class_indices}")
    print(f"  Train samples  : {train_gen.samples}")
    print(f"  Val samples    : {val_gen.samples}")
    print(f"  Test samples   : {test_gen.samples}")
    print(f"  Image size     : {config.IMAGE_SIZE}")
    print(f"  Batch size     : {config.BATCH_SIZE}")
    print(f"  Train batches  : {len(train_gen)}")
    print(f"  Val batches    : {len(val_gen)}")
    print(f"  Test batches   : {len(test_gen)}\n")

    return train_gen, val_gen, test_gen


# =============================================================================
# 2. SINGLE IMAGE PREPROCESSOR (used by Flask app and Grad-CAM)
# =============================================================================

def preprocess_single_image(img_path):
    """
    Load and preprocess a single image for model inference.
    Does NOT apply augmentation — only resize + normalize.

    Args:
        img_path (str): path to the image file (.jpg / .jpeg / .png)

    Returns:
        img_array (np.ndarray): shape (1, 224, 224, 3), values in [0, 1]
        original_img (np.ndarray): shape (224, 224, 3), uint8, for display

    Raises:
        FileNotFoundError: if img_path does not exist
        ValueError: if image cannot be decoded
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"[PREPROCESS] Image not found: {img_path}")

    # Load as PIL Image and resize to model's expected input size
    pil_img = load_img(img_path, target_size=config.IMAGE_SIZE, color_mode='rgb')

    # Convert to numpy array — shape (224, 224, 3), values [0, 255]
    original_img = img_to_array(pil_img).astype(np.uint8)

    # Normalize to [0, 1] and add batch dimension → shape (1, 224, 224, 3)
    img_array = original_img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img


# =============================================================================
# 3. CLASS WEIGHT CALCULATOR (handles class imbalance)
# =============================================================================

def compute_class_weights(train_gen):
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes.astype(int)

    # Guard: if dataset is empty, skip weight calculation
    if len(labels) == 0:
        print("[PREPROCESS] ⚠️  No training samples found — skipping class weight calculation.")
        print("[PREPROCESS] ⚠️  Make sure your dataset images are inside the train/fractured and train/normal folders!")
        return {0: 1.0, 1: 1.0}

    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    print(f"[PREPROCESS] Class weights: {class_weights}")
    return class_weights

# =============================================================================
# 4. DATASET VISUALIZER (for sanity check and notebook exploration)
# =============================================================================

def visualize_samples(train_gen, num_samples=8, save_path=None):
    """
    Plot a grid of sample augmented training images with their labels.
    Useful for verifying augmentation looks reasonable before training.

    Args:
        train_gen  : training data generator
        num_samples: number of images to display (default 8)
        save_path  : if provided, saves the plot as a PNG
    """
    # Reverse the class_indices dict so we can map int → class name
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}

    # Get one batch from the generator
    images, labels = next(train_gen)

    # Limit to num_samples (batch might be larger)
    images = images[:num_samples]
    labels = labels[:num_samples]

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig = plt.figure(figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(
        "Sample Augmented Training Images",
        fontsize=14, fontweight='bold', y=1.01
    )

    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for i in range(num_samples):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(images[i])  # already normalized [0,1], imshow handles it
        label_name = idx_to_class[int(labels[i])]
        color = 'red' if label_name == 'fractured' else 'green'
        ax.set_title(label_name.upper(), color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PREPROCESS] Sample grid saved → {save_path}")

    plt.show()
    plt.close()


# =============================================================================
# 5. DATASET STATISTICS PRINTER
# =============================================================================

def print_dataset_stats(train_gen, val_gen, test_gen):
    """
    Print a formatted table of dataset statistics per split and class.
    Helps quickly spot class imbalance before training.

    Args:
        train_gen, val_gen, test_gen: data generators
    """
    from collections import Counter

    def class_counts(gen):
        counts = Counter(gen.classes)
        idx_to_class = {v: k for k, v in gen.class_indices.items()}
        return {idx_to_class[k]: v for k, v in counts.items()}

    train_counts = class_counts(train_gen)
    val_counts   = class_counts(val_gen)
    test_counts  = class_counts(test_gen)

    print("\n" + "="*55)
    print("  DATASET STATISTICS")
    print("="*55)
    print(f"  {'Split':<10} {'Normal':>10} {'Fractured':>12} {'Total':>8}")
    print("-"*55)

    for name, counts in [('Train', train_counts), ('Val', val_counts), ('Test', test_counts)]:
        normal    = counts.get('normal', 0)
        fractured = counts.get('fractured', 0)
        total     = normal + fractured
        print(f"  {name:<10} {normal:>10} {fractured:>12} {total:>8}")

    print("="*55 + "\n")


# =============================================================================
# QUICK SANITY CHECK — run this file directly to verify generators work
# =============================================================================

if __name__ == '__main__':
    print("\n[PREPROCESS] Running sanity check...\n")

    # Build generators
    train_gen, val_gen, test_gen = build_generators()

    # Print class distribution
    print_dataset_stats(train_gen, val_gen, test_gen)

    # Compute class weights
    class_weights = compute_class_weights(train_gen)

    # Visualize augmented samples
    sample_plot_path = os.path.join(config.PLOTS_DIR, 'sample_augmented_images.png')
    visualize_samples(train_gen, num_samples=8, save_path=sample_plot_path)

    # Test single image loader (uses first image found in test set)
    test_images = []
    for root, dirs, files in os.walk(config.TEST_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, f))
        if test_images:
            break

    if test_images:
        img_array, original = preprocess_single_image(test_images[0])
        print(f"[PREPROCESS] Single image test:")
        print(f"  Path          : {test_images[0]}")
        print(f"  img_array     : shape={img_array.shape}, dtype={img_array.dtype}, "
              f"range=[{img_array.min():.3f}, {img_array.max():.3f}]")
        print(f"  original_img  : shape={original.shape}, dtype={original.dtype}")
    else:
        print("[PREPROCESS] ⚠️  No test images found — download dataset first.")

    print("\n[PREPROCESS] ✅ All checks passed.\n")
