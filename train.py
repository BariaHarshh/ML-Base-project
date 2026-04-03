# =============================================================================
# train.py — Full 2-Phase Training Pipeline for Bone Fracture Detection
# =============================================================================
# Phase 1: Train custom head only (base frozen)     — 15 epochs
# Phase 2: Fine-tune top ResNet50 layers + head     — 15 more epochs
#
# Callbacks used:
#   - EarlyStopping      : stop if val_loss stagnates (patience=5)
#   - ModelCheckpoint    : save best weights automatically
#   - ReduceLROnPlateau  : halve LR if val_loss plateaus (patience=3)
#
# Outputs:
#   models/best_model.h5          ← best Phase 1 weights
#   models/finetuned_model.h5     ← best Phase 2 weights  (use this for inference)
#   outputs/plots/phase1_history.png
#   outputs/plots/phase2_history.png
#   outputs/plots/combined_history.png
# =============================================================================

# -----------------------------------------------------------------------------
# GOOGLE COLAB SETUP — uncomment and run first if using Colab:
# -----------------------------------------------------------------------------
# import subprocess
# subprocess.run(['pip', 'install', '-q', 'tensorflow', 'scikit-learn', 'opencv-python'])
# from google.colab import drive
# drive.mount('/content/drive')
# import sys; sys.path.append('/content/drive/MyDrive/bone_fracture_detection')
#
# GPU CHECK:
# import tensorflow as tf
# print("GPU:", tf.config.list_physical_devices('GPU'))
# If empty → Runtime → Change runtime type → T4 GPU
# -----------------------------------------------------------------------------

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate slightly corrupted images
import config
from preprocess import build_generators, compute_class_weights, print_dataset_stats
from model import build_model, fine_tune_model, configure_gpu, print_model_summary


# =============================================================================
# 1. CALLBACKS
# =============================================================================

def build_callbacks(phase=1):
    """
    Build the list of Keras callbacks for a given training phase.

    Args:
        phase (int): 1 or 2 — determines which model path to save to

    Returns:
        list of tf.keras.callbacks
    """

    # Save path depends on phase
    checkpoint_path = (
        config.BEST_MODEL_PATH if phase == 1
        else config.FINETUNED_MODEL_PATH
    )

    # -------------------------------------------------------------------------
    # ModelCheckpoint — saves model only when val_loss improves
    # save_best_only=True means we always keep the best weights, not the last
    # -------------------------------------------------------------------------
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath        = checkpoint_path,
        monitor         = config.MONITOR_METRIC,
        save_best_only  = True,
        save_weights_only = False,     # save full model (architecture + weights)
        mode            = 'min',       # lower val_loss is better
        verbose         = 1,
    )

    # -------------------------------------------------------------------------
    # EarlyStopping — halt training if val_loss doesn't improve for N epochs
    # restore_best_weights=True rolls back to the best epoch automatically
    # -------------------------------------------------------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor              = config.MONITOR_METRIC,
        patience             = config.EARLY_STOPPING_PATIENCE,
        restore_best_weights = True,
        mode                 = 'min',
        verbose              = 1,
    )

    # -------------------------------------------------------------------------
    # ReduceLROnPlateau — reduce LR when val_loss is stuck
    # Helps escape local minima without manually scheduling LR
    # -------------------------------------------------------------------------
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor   = config.MONITOR_METRIC,
        factor    = config.REDUCE_LR_FACTOR,
        patience  = config.REDUCE_LR_PATIENCE,
        min_lr    = config.REDUCE_LR_MIN,
        mode      = 'min',
        verbose   = 1,
    )

    # -------------------------------------------------------------------------
    # CSVLogger — logs epoch metrics to a CSV for later analysis
    # -------------------------------------------------------------------------
    log_path = os.path.join(config.OUTPUTS_DIR, f'training_log_phase{phase}.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=False)

    print(f"[TRAIN] Callbacks ready for Phase {phase}:")
    print(f"        Checkpoint → {checkpoint_path}")
    print(f"        CSV log    → {log_path}")

    return [checkpoint, early_stop, reduce_lr, csv_logger]


# =============================================================================
# 2. HISTORY PLOTTER
# =============================================================================

def plot_history(history, phase, save=True):
    """
    Plot and save training/validation accuracy and loss curves for one phase.

    Args:
        history  : Keras History object returned by model.fit()
        phase    : 1 or 2 (used in title and filename)
        save     : if True, saves PNG to config.PLOTS_DIR
    """

    metrics = history.history
    epochs  = range(1, len(metrics['loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Phase {phase} Training History — '
        f'{"Frozen Base" if phase == 1 else "Fine-Tuning"}',
        fontsize=14, fontweight='bold'
    )

    # --- Loss curve ---
    axes[0].plot(epochs, metrics['loss'],     'b-o', markersize=4, label='Train Loss')
    axes[0].plot(epochs, metrics['val_loss'], 'r-o', markersize=4, label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Crossentropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy curve ---
    axes[1].plot(epochs, metrics['accuracy'],     'b-o', markersize=4, label='Train Acc')
    axes[1].plot(epochs, metrics['val_accuracy'], 'r-o', markersize=4, label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = os.path.join(config.PLOTS_DIR, f'phase{phase}_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[TRAIN] Phase {phase} plot saved → {save_path}")

    plt.show()
    plt.close()


def plot_combined_history(history1, history2, save=True):
    """
    Plot combined training history across both phases side by side.
    Draws a vertical dashed line at the Phase 1/2 boundary.

    Args:
        history1, history2 : Keras History objects from Phase 1 and Phase 2
        save               : if True, saves PNG to config.PLOTS_DIR
    """

    # Merge both histories
    def merge(key):
        return history1.history[key] + history2.history[key]

    loss     = merge('loss')
    val_loss = merge('val_loss')
    acc      = merge('accuracy')
    val_acc  = merge('val_accuracy')
    recall   = merge('recall')
    val_rec  = merge('val_recall')
    auc      = merge('auc')
    val_auc  = merge('val_auc')

    phase1_end = len(history1.history['loss'])
    total_eps  = len(loss)
    epochs     = range(1, total_eps + 1)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Combined Training History (Phase 1 + Phase 2 Fine-Tuning)',
                 fontsize=15, fontweight='bold')

    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    axs  = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    plot_data = [
        ('Loss',      loss,   val_loss, 'Train Loss',   'Val Loss'),
        ('Accuracy',  acc,    val_acc,  'Train Acc',    'Val Acc'),
        ('Recall',    recall, val_rec,  'Train Recall', 'Val Recall'),
        ('AUC',       auc,    val_auc,  'Train AUC',    'Val AUC'),
    ]

    for ax, (title, train_vals, val_vals, train_label, val_label) in zip(axs, plot_data):
        ax.plot(epochs, train_vals, 'b-o', markersize=3, label=train_label)
        ax.plot(epochs, val_vals,   'r-o', markersize=3, label=val_label)
        ax.axvline(x=phase1_end, color='green', linestyle='--',
                   linewidth=1.5, label='Fine-tune start')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = os.path.join(config.PLOTS_DIR, 'combined_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[TRAIN] Combined plot saved → {save_path}")

    plt.show()
    plt.close()


# =============================================================================
# 3. METRICS PRINTER
# =============================================================================

def print_phase_results(history, phase):
    """
    Print a clean table of the best epoch metrics from a training phase.

    Args:
        history : Keras History object
        phase   : 1 or 2
    """
    metrics = history.history
    best_epoch = np.argmin(metrics['val_loss'])   # epoch with lowest val_loss

    print(f"\n{'='*55}")
    print(f"  PHASE {phase} RESULTS  (Best Epoch: {best_epoch + 1})")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Train':>10} {'Val':>10}")
    print(f"  {'-'*45}")

    metric_pairs = [
        ('Loss',      'loss',      'val_loss'),
        ('Accuracy',  'accuracy',  'val_accuracy'),
        ('AUC',       'auc',       'val_auc'),
        ('Recall',    'recall',    'val_recall'),
        ('Precision', 'precision', 'val_precision'),
    ]

    for label, train_key, val_key in metric_pairs:
        if train_key in metrics:
            t_val = metrics[train_key][best_epoch]
            v_val = metrics[val_key][best_epoch]
            print(f"  {label:<25} {t_val:>10.4f} {v_val:>10.4f}")

    print(f"{'='*55}\n")


# =============================================================================
# 4. MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """
    Full 2-phase training pipeline:
      Phase 1 → train custom head (frozen base)
      Phase 2 → fine-tune top layers with lower LR

    Saves:
      best_model.h5      — best val_loss model from Phase 1
      finetuned_model.h5 — best val_loss model from Phase 2 (use for inference)
    """

    start_time = time.time()
    print("\n" + "="*60)
    print("  BONE FRACTURE DETECTION — Training Pipeline")
    print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    configure_gpu()
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("[TRAIN] Loading data generators...")
    train_gen, val_gen, test_gen = build_generators()
    print_dataset_stats(train_gen, val_gen, test_gen)

    # Compute class weights for imbalanced dataset handling
    class_weights = compute_class_weights(train_gen)

    # Validate that dataset is not empty before proceeding
    if train_gen.samples == 0:
        raise RuntimeError(
            "[TRAIN] ❌ No training images found!\n"
            f"  Check that images exist in: {config.TRAIN_DIR}\n"
            "  Expected subfolders: fractured/ and normal/"
        )

    # =========================================================================
    # PHASE 1 — Train custom head (base frozen)
    # =========================================================================
    print("\n" + "-"*60)
    print("  PHASE 1: Training Custom Head (Base Frozen)")
    print(f"  Epochs: {config.PHASE1_EPOCHS}  |  LR: {config.LEARNING_RATE}")
    print("-"*60 + "\n")

    model = build_model()
    print_model_summary(model)

    callbacks_p1 = build_callbacks(phase=1)

    phase1_start = time.time()

    history1 = model.fit(
        train_gen,
        epochs          = config.PHASE1_EPOCHS,
        validation_data = val_gen,
        callbacks       = callbacks_p1,
        class_weight    = class_weights,
        verbose         = 1,
    )

    phase1_time = time.time() - phase1_start
    print(f"\n[TRAIN] Phase 1 complete in {phase1_time/60:.1f} minutes")
    print_phase_results(history1, phase=1)
    plot_history(history1, phase=1)

    # =========================================================================
    # PHASE 2 — Fine-tune top ResNet50 layers
    # =========================================================================
    print("\n" + "-"*60)
    print("  PHASE 2: Fine-Tuning (Unfreezing Top Layers)")
    print(f"  Epochs: {config.PHASE2_EPOCHS}  |  LR: {config.FINE_TUNE_LR}")
    print(f"  Unfreezing layers after index: {config.FINE_TUNE_AT}")
    print("-"*60 + "\n")

    # Load best Phase 1 weights before fine-tuning
    # This ensures we start Phase 2 from the best Phase 1 checkpoint
    if os.path.exists(config.BEST_MODEL_PATH):
        model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
        print(f"[TRAIN] Loaded best Phase 1 weights from: {config.BEST_MODEL_PATH}")

    # Apply fine-tuning configuration
    model = fine_tune_model(model)

    callbacks_p2 = build_callbacks(phase=2)

    phase2_start = time.time()

    history2 = model.fit(
        train_gen,
        epochs          = config.PHASE2_EPOCHS,
        validation_data = val_gen,
        callbacks       = callbacks_p2,
        class_weight    = class_weights,
        verbose         = 1,
    )

    phase2_time = time.time() - phase2_start
    print(f"\n[TRAIN] Phase 2 complete in {phase2_time/60:.1f} minutes")
    print_phase_results(history2, phase=2)
    plot_history(history2, phase=2)

    # =========================================================================
    # COMBINED PLOT + SUMMARY
    # =========================================================================
    plot_combined_history(history1, history2)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Total training time  : {total_time/60:.1f} minutes")
    print(f"  Phase 1 time         : {phase1_time/60:.1f} minutes")
    print(f"  Phase 2 time         : {phase2_time/60:.1f} minutes")
    print(f"  Best model saved     : {config.BEST_MODEL_PATH}")
    print(f"  Fine-tuned model     : {config.FINETUNED_MODEL_PATH}")
    print(f"  Plots saved to       : {config.PLOTS_DIR}")
    print("="*60)
    print("\n  Next step: python evaluate.py\n")

    return history1, history2, model


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    history1, history2, model = train()
