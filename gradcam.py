# =============================================================================
# gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)
# =============================================================================
# Grad-CAM produces a heatmap highlighting which regions in the X-ray most
# influenced the model's fracture/normal prediction.
#
# Algorithm:
#   1. Forward pass → record activations of last conv layer (conv5_block3_out)
#   2. Compute gradients of predicted class score w.r.t. those activations
#   3. Global-average-pool gradients → per-channel importance weights
#   4. Weight each channel by its importance, sum across channels
#   5. Apply ReLU → normalize to [0,1] → resize to 224x224
#   6. Apply jet colormap → blend over original X-ray
#
# Model architecture handled:
#   Input → ResNet50 (nested sub-model) → GlobalAvgPool → Dense(256)
#        → Dense(128) → Dense(1, sigmoid)
#
# Usage:
#   python gradcam.py                         ← runs on test set images
#   from gradcam import predict_and_gradcam   ← called by app.py
# =============================================================================

import os
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (safe for Flask & scripts)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image

import config
from preprocess import preprocess_single_image
from model import load_model


# =============================================================================
# 1. GRAD-CAM CORE COMPUTATION
# =============================================================================

def compute_gradcam(model, img_array, layer_name=None):
    """
    Compute the Grad-CAM heatmap for a single preprocessed image.

    Strategy for nested-model architecture
    (Input → resnet50 sub-model → custom head):
      - Build a sub-model from ResNet50's input to the target conv layer.
      - Use GradientTape to record operations from that conv output through
        the remaining custom-head layers to the final sigmoid score.
      - Compute d(score) / d(conv_outputs) and pool spatially.

    Args:
        model      (tf.keras.Model) : full BoneFractureDetector model
        img_array  (np.ndarray)     : shape (1, 224, 224, 3), values in [0, 1]
        layer_name (str)            : target conv layer inside ResNet50
                                      (default: config.GRADCAM_LAYER)

    Returns:
        heatmap   (np.ndarray) : shape (H_conv, W_conv), values in [0, 1]
        pred_prob (float)      : model's P(fractured)

    Raises:
        ValueError  : if layer_name is not found in ResNet50
        RuntimeError: if gradient computation fails
    """
    if layer_name is None:
        layer_name = config.GRADCAM_LAYER

    # -------------------------------------------------------------------------
    # Locate the ResNet50 sub-model and the target conv layer inside it.
    # In model.py: model.layers[0] = InputLayer
    #              model.layers[1] = ResNet50 (named 'resnet50')
    #              model.layers[2:] = GlobalAvgPool → Dense → ... → Output
    # -------------------------------------------------------------------------
    try:
        base_model = model.get_layer('resnet50')
    except ValueError:
        # Fallback: find the first sub-model layer
        base_model = next(
            (l for l in model.layers if isinstance(l, tf.keras.Model)), None
        )
        if base_model is None:
            raise RuntimeError(
                "[GRADCAM] Could not locate ResNet50 sub-model inside the loaded model."
            )

    try:
        base_model.get_layer(layer_name)
    except ValueError:
        available = [l.name for l in base_model.layers
                     if 'conv' in l.name.lower()][-5:]
        raise ValueError(
            f"[GRADCAM] Layer '{layer_name}' not found in ResNet50.\n"
            f"  Last available conv layers: {available}\n"
            f"  Update GRADCAM_LAYER in config.py to a valid layer name."
        )

    # -------------------------------------------------------------------------
    # Build a sub-model: ResNet50.input → target conv layer output
    # This gives us access to the intermediate conv activations as a tensor
    # that lives in the same computation graph as the outer model layers.
    # -------------------------------------------------------------------------
    conv_sub_model = tf.keras.Model(
        inputs  = base_model.input,
        outputs = base_model.get_layer(layer_name).output,
        name    = 'conv_feature_extractor',
    )

    # Collect the outer model's layers that come AFTER the ResNet50 sub-model
    # (index 0 = InputLayer, index 1 = resnet50, index 2+ = custom head)
    outer_head_layers = model.layers[2:]

    # -------------------------------------------------------------------------
    # Forward pass with GradientTape
    #
    # Key points:
    #  - We watch `img_tensor` so the tape records ALL downstream ops.
    #  - tape.watch(conv_outputs) is called AFTER computing conv_outputs;
    #    subsequent operations on conv_outputs will be recorded, making
    #    d(class_score)/d(conv_outputs) computable.
    #  - Dropout / BatchNorm layers are called with training=False.
    # -------------------------------------------------------------------------
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        tape.watch(img_tensor)

        # 1. Get conv layer activations from ResNet50 sub-model
        conv_outputs = conv_sub_model(img_tensor, training=False)

        # 2. Explicitly watch conv_outputs so tape records ops from here onward
        tape.watch(conv_outputs)

        # 3. Apply custom classification head layers sequentially
        x = conv_outputs
        for layer in outer_head_layers:
            x = layer(x, training=False)

        # x is now the sigmoid output, shape (1, 1)
        pred_prob = float(x[0][0])

        # Class score: raw output neuron (before any thresholding)
        class_score = x[:, 0]

    # -------------------------------------------------------------------------
    # Compute gradients: d(class_score) / d(conv_outputs)
    # Shape: same as conv_outputs → (1, H, W, C)
    # -------------------------------------------------------------------------
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "[GRADCAM] Gradient computation returned None.\n"
            "  Possible causes:\n"
            "  1. Layer name not in the forward-pass computation graph.\n"
            "  2. Model was loaded without the proper layer connections.\n"
            f"  Target layer: '{layer_name}'"
        )

    # -------------------------------------------------------------------------
    # Pool gradients globally across spatial dims → importance weight per channel
    # Shape: (C,) — one scalar weight per feature map channel
    # -------------------------------------------------------------------------
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # → (C,)

    # -------------------------------------------------------------------------
    # Weight feature maps by their importance and sum across channels
    # conv_outputs[0] shape: (H, W, C)
    # -------------------------------------------------------------------------
    conv_np   = conv_outputs[0].numpy()    # (H, W, C)
    grads_np  = pooled_grads.numpy()       # (C,)

    # Multiply each channel by its pooled gradient weight
    weighted  = conv_np * grads_np[np.newaxis, np.newaxis, :]  # (H, W, C)

    # Sum across channels → (H, W)
    heatmap = np.sum(weighted, axis=-1)

    # -------------------------------------------------------------------------
    # ReLU: keep only positive contributions to the class score
    # Negative values would indicate regions that SUPPRESS the prediction.
    # -------------------------------------------------------------------------
    heatmap = np.maximum(heatmap, 0)

    # Normalize to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap, pred_prob


# =============================================================================
# 2. HEATMAP → COLOR OVERLAY
# =============================================================================

def overlay_heatmap(heatmap, original_img, alpha=None):
    """
    Resize the raw Grad-CAM heatmap to match the original image size,
    apply a jet colormap, and blend with the original X-ray.

    Args:
        heatmap      (np.ndarray) : shape (H_conv, W_conv), values in [0, 1]
        original_img (np.ndarray) : shape (224, 224, 3), dtype uint8
        alpha        (float)      : heatmap blend weight  (default: config.GRADCAM_ALPHA)
                                    0 = original only, 1 = heatmap only

    Returns:
        superimposed_img (np.ndarray) : shape (224, 224, 3), uint8  — blended result
        heatmap_colored  (np.ndarray) : shape (224, 224, 3), uint8  — colorized heatmap
    """
    if alpha is None:
        alpha = config.GRADCAM_ALPHA

    h, w = original_img.shape[:2]

    # Resize heatmap to original image dimensions using bilinear interpolation
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_LINEAR)

    # Apply jet colormap: 0 = blue (low activation), 1 = red (high activation)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_bgr     = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)  # → RGB

    # Blend: result = alpha * heatmap_colored + (1 - alpha) * original
    original_f = original_img.astype(np.float32)
    heatmap_f  = heatmap_colored.astype(np.float32)

    superimposed = alpha * heatmap_f + (1.0 - alpha) * original_f
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed, heatmap_colored


# =============================================================================
# 3. SINGLE-IMAGE GRAD-CAM PIPELINE
# =============================================================================

def generate_gradcam(img_path, model=None, save_dir=None, show=False):
    """
    Full Grad-CAM pipeline for one image:
      1. Load & preprocess image
      2. Compute Grad-CAM heatmap
      3. Create heatmap overlay
      4. Build 3-panel visualization (original | heatmap | overlay)
      5. Save PNG and optionally display

    Args:
        img_path (str)          : path to input X-ray image (.jpg / .jpeg / .png)
        model  (tf.keras.Model) : loaded Keras model (loaded from disk if None)
        save_dir (str)          : directory to save PNG  (default: config.HEATMAPS_DIR)
        show     (bool)         : call plt.show() for interactive display

    Returns:
        result (dict):
            'pred_prob'        → float   : P(fractured)
            'pred_label'       → str     : 'Fractured' or 'Normal'
            'confidence'       → float   : confidence in the predicted label
            'heatmap_path'     → str     : path to saved PNG
            'superimposed_img' → ndarray : blended heatmap image (224, 224, 3)
            'heatmap'          → ndarray : raw normalized heatmap (H, W)
    """
    if save_dir is None:
        save_dir = config.HEATMAPS_DIR
    os.makedirs(save_dir, exist_ok=True)

    if model is None:
        model = load_model()

    # ------------------------------------------------------------------
    # 1. Preprocess
    # ------------------------------------------------------------------
    img_array, original_img = preprocess_single_image(img_path)

    # ------------------------------------------------------------------
    # 2. Compute Grad-CAM
    # ------------------------------------------------------------------
    heatmap, pred_prob = compute_gradcam(model, img_array)

    # ------------------------------------------------------------------
    # 3. Overlay
    # ------------------------------------------------------------------
    superimposed_img, heatmap_colored = overlay_heatmap(heatmap, original_img)

    # ------------------------------------------------------------------
    # 4. Interpretation
    # ------------------------------------------------------------------
    pred_label  = 'Fractured' if pred_prob >= config.THRESHOLD else 'Normal'
    confidence  = pred_prob if pred_label == 'Fractured' else (1.0 - pred_prob)
    label_color = 'red' if pred_label == 'Fractured' else 'green'

    # ------------------------------------------------------------------
    # 5. Build 3-panel figure
    #    Panel 1: Original X-ray
    #    Panel 2: Colorized Grad-CAM heatmap (jet)
    #    Panel 3: Heatmap blended on X-ray
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Grad-CAM Analysis  |  Prediction: {pred_label}  '
        f'(P(fracture) = {pred_prob:.3f},  confidence = {confidence * 100:.1f}%)',
        fontsize=13, fontweight='bold', color=label_color,
    )

    axes[0].imshow(original_img)
    axes[0].set_title('Original X-Ray', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(heatmap_colored)
    axes[1].set_title('Grad-CAM Heatmap\n(red = high activation)', fontsize=11)
    axes[1].axis('off')

    # Colorbar on heatmap panel
    sm = plt.cm.ScalarMappable(
        cmap='jet', norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(superimposed_img)
    axes[2].set_title('Overlay (Heatmap + X-Ray)', fontsize=11)
    axes[2].axis('off')

    # Medical disclaimer as a small footnote
    fig.text(
        0.5, -0.01,
        config.MEDICAL_DISCLAIMER,
        ha='center', fontsize=7, color='gray', wrap=True,
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    img_stem  = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f'gradcam_{img_stem}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[GRADCAM] Heatmap saved → {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return {
        'pred_prob'        : pred_prob,
        'pred_label'       : pred_label,
        'confidence'       : confidence,
        'heatmap_path'     : save_path,
        'superimposed_img' : superimposed_img,
        'heatmap'          : heatmap,
    }


# =============================================================================
# 4. BATCH GRAD-CAM RUNNER
# =============================================================================

def run_batch_gradcam(image_paths, model=None, save_dir=None, max_images=10):
    """
    Generate Grad-CAM visualizations for a list of images.
    Loads the model once and processes each image sequentially.

    Args:
        image_paths (list[str]) : list of absolute image file paths
        model                   : loaded Keras model (loaded once if None)
        save_dir   (str)        : output directory (default: config.HEATMAPS_DIR)
        max_images (int)        : cap on the number of images processed

    Returns:
        results (list[dict]) : list of result dicts from generate_gradcam(),
                               with an extra 'path' key and optional 'error' key
    """
    if model is None:
        model = load_model()

    if save_dir is None:
        save_dir = config.HEATMAPS_DIR
    os.makedirs(save_dir, exist_ok=True)

    paths_to_process = image_paths[:max_images]
    total = len(paths_to_process)

    print(f"\n[GRADCAM] Batch processing: {total} image(s)")
    print(f"          Output dir      : {save_dir}\n")

    results = []
    for idx, img_path in enumerate(paths_to_process, 1):
        basename = os.path.basename(img_path)
        print(f"[GRADCAM] [{idx}/{total}] {basename}")
        try:
            result = generate_gradcam(
                img_path = img_path,
                model    = model,
                save_dir = save_dir,
                show     = False,
            )
            results.append({'path': img_path, **result})
            print(f"          → {result['pred_label']}  "
                  f"(P={result['pred_prob']:.3f}, conf={result['confidence']*100:.1f}%)")
        except Exception as exc:
            print(f"[GRADCAM] ⚠️  Failed: {exc}")
            results.append({'path': img_path, 'error': str(exc)})

    # Summary
    good      = [r for r in results if 'error' not in r]
    fractured = sum(1 for r in good if r['pred_label'] == 'Fractured')
    normal    = len(good) - fractured

    print(f"\n[GRADCAM] Batch complete.")
    print(f"          Processed : {len(good)}/{total}")
    print(f"          Fractured : {fractured}")
    print(f"          Normal    : {normal}")
    print(f"          Saved to  : {save_dir}\n")

    return results


# =============================================================================
# 5. SUMMARY GRID VISUALIZER
# =============================================================================

def plot_gradcam_grid(results, save_path=None, cols=4):
    """
    Create a summary grid showing the heatmap overlay for each processed image.
    Prediction label and probability are shown above each panel.

    Args:
        results   (list[dict]) : output of run_batch_gradcam()
        save_path (str)        : PNG save path (default: HEATMAPS_DIR/gradcam_grid.png)
        cols      (int)        : columns in the grid (default 4)
    """
    valid = [r for r in results if 'error' not in r and 'superimposed_img' in r]

    if not valid:
        print("[GRADCAM] ⚠️  No valid results to plot in grid.")
        return

    n    = len(valid)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    fig.suptitle(
        'Grad-CAM Summary Grid',
        fontsize=14, fontweight='bold', y=1.01,
    )

    # Flatten axes array regardless of shape
    if n == 1:
        axes_flat = [axes]
    elif rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for i, result in enumerate(valid):
        ax    = axes_flat[i]
        label = result['pred_label']
        prob  = result['pred_prob']
        color = 'red' if label == 'Fractured' else 'green'
        fname = os.path.basename(result['path'])
        # Truncate long filenames
        display_name = (fname[:18] + '…') if len(fname) > 20 else fname

        ax.imshow(result['superimposed_img'])
        ax.set_title(
            f"{display_name}\n{label}  ({prob:.2f})",
            fontsize=8, color=color, fontweight='bold',
        )
        ax.axis('off')

    # Hide any unused subplot cells
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.HEATMAPS_DIR, 'gradcam_grid.png')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[GRADCAM] Grid saved → {save_path}")
    plt.close(fig)


# =============================================================================
# 6. FLASK-READY SINGLE PREDICTION FUNCTION
# =============================================================================

def predict_and_gradcam(img_path, model, save_static=True):
    """
    Convenience wrapper for the Flask app:
    Run inference + Grad-CAM on one uploaded image, save the heatmap to the
    static directory so Flask can serve it directly.

    Designed to be called with a pre-loaded model (load once at app startup)
    to avoid the overhead of loading the model on every request.

    Args:
        img_path    (str)            : path to the uploaded image file
        model       (tf.keras.Model) : preloaded Keras model
        save_static (bool)           : if True, save to config.STATIC_HEATMAPS;
                                       otherwise save to config.HEATMAPS_DIR

    Returns:
        dict:
            'pred_label'    → 'Fractured' or 'Normal'
            'pred_prob'     → float, probability of fracture  (4 d.p.)
            'confidence'    → float, confidence in prediction (4 d.p.)
            'heatmap_fname' → filename of saved PNG (use in url_for / <img src>)
            'disclaimer'    → medical disclaimer string (from config)
    """
    target_dir    = config.STATIC_HEATMAPS if save_static else config.HEATMAPS_DIR
    img_stem      = os.path.splitext(os.path.basename(img_path))[0]
    heatmap_fname = f'gradcam_{img_stem}.png'

    result = generate_gradcam(
        img_path = img_path,
        model    = model,
        save_dir = target_dir,
        show     = False,
    )

    return {
        'pred_label'    : result['pred_label'],
        'pred_prob'     : round(float(result['pred_prob']), 4),
        'confidence'    : round(float(result['confidence']), 4),
        'heatmap_fname' : heatmap_fname,
        'disclaimer'    : config.MEDICAL_DISCLAIMER,
    }


# =============================================================================
# ENTRY POINT — runs Grad-CAM on a sample of test images for visual inspection
# =============================================================================

if __name__ == '__main__':

    print("\n" + "=" * 60)
    print("  BONE FRACTURE DETECTION — Grad-CAM Visualization")
    print("=" * 60 + "\n")

    # Load model once (prefers finetuned_model.h5, falls back to best_model.h5)
    model = load_model()

    # -------------------------------------------------------------------------
    # Collect test images — up to 5 from each class for a balanced demo
    # -------------------------------------------------------------------------
    fractured_dir = os.path.join(config.TEST_DIR, 'fractured')
    normal_dir    = os.path.join(config.TEST_DIR, 'normal')

    def _collect(directory, n=5):
        imgs = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            imgs.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(imgs)[:n]

    fractured_imgs = _collect(fractured_dir, n=5)
    normal_imgs    = _collect(normal_dir,    n=5)
    all_imgs       = fractured_imgs + normal_imgs

    if not all_imgs:
        print("[GRADCAM] ⚠️  No test images found.")
        print(f"          Expected images in: {config.TEST_DIR}")
        print("          Subfolders needed : fractured/  and  normal/\n")
    else:
        print(f"[GRADCAM] Found {len(fractured_imgs)} fractured + "
              f"{len(normal_imgs)} normal test images.")
        print(f"[GRADCAM] Generating Grad-CAM for up to {len(all_imgs)} images…\n")

        # Run batch Grad-CAM
        results = run_batch_gradcam(
            image_paths = all_imgs,
            model       = model,
            save_dir    = config.HEATMAPS_DIR,
            max_images  = 10,
        )

        # Generate summary grid
        grid_path = os.path.join(config.HEATMAPS_DIR, 'gradcam_grid.png')
        plot_gradcam_grid(results, save_path=grid_path, cols=4)

        print("\n" + "=" * 60)
        print("  GRAD-CAM COMPLETE")
        print("=" * 60)
        print(f"  Heatmaps saved to : {config.HEATMAPS_DIR}")
        print(f"  Summary grid      : {grid_path}")
        print(f"  Next step         : python app.py")
        print("=" * 60 + "\n")
