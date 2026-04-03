# =============================================================================
# evaluate.py — Complete Model Evaluation for Bone Fracture Detection
# =============================================================================
# Loads best saved model, runs inference on test set, and produces:
#   1. Classification report (precision, recall, F1 per class)
#   2. Confusion matrix
#   3. ROC curve with AUC score
#   4. Precision-Recall curve
#
# Uses THRESHOLD = 0.4 (from config) — lower than default 0.5 to
# prioritize recall (catching fractures) over precision.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
import tensorflow as tf
from PIL import ImageFile

# Allow slightly truncated images to load without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

import config
from preprocess import build_generators
from model import load_model


# =============================================================================
# 1. PREDICTION PIPELINE
# =============================================================================

def get_predictions(model, test_gen):
    """
    Run inference on the entire test set and return raw probabilities,
    binary predictions, and true labels.

    Args:
        model    : loaded Keras model
        test_gen : test data generator (shuffle=False — order is preserved)

    Returns:
        y_true  (np.ndarray): true binary labels  [0=normal, 1=fractured]
        y_probs (np.ndarray): model output probabilities in [0, 1]
        y_pred  (np.ndarray): binary predictions using config.THRESHOLD
    """
    print(f"[EVALUATE] Running inference on {test_gen.samples} test images...")
    print(f"           Using threshold: {config.THRESHOLD} (recall-optimized)\n")

    # Reset generator to start from the beginning (important — never skip this)
    test_gen.reset()

    # Predict in batches — returns shape (N, 1)
    y_probs_raw = model.predict(test_gen, verbose=1)

    # Flatten to 1D array of probabilities
    y_probs = y_probs_raw.flatten()

    # True labels from the generator (in same order as predictions because shuffle=False)
    y_true = test_gen.classes.astype(int)

    # Get class index for "fractured" — depends on alphabetical folder ordering
    # Keras assigns: fractured=0, normal=1 (f < n alphabetically)
    # Our model outputs P(class_index=1) with sigmoid
    # So we need to check which index corresponds to "fractured"
    fractured_idx = test_gen.class_indices.get('fractured', 1)

    # If fractured=0, our sigmoid output P(1) = P(normal), so flip probabilities
    if fractured_idx == 0:
        # Model outputs P(normal), so P(fractured) = 1 - output
        y_probs_fractured = 1.0 - y_probs
        # Also flip true labels so 1=fractured, 0=normal consistently
        y_true = 1 - y_true
    else:
        y_probs_fractured = y_probs

    # Apply threshold — output 1 (fractured) if probability >= threshold
    y_pred = (y_probs_fractured >= config.THRESHOLD).astype(int)

    print(f"[EVALUATE] Predictions complete.")
    print(f"           Test samples    : {len(y_true)}")
    print(f"           Predicted frac  : {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")
    print(f"           True fractured  : {y_true.sum()} ({y_true.mean()*100:.1f}%)\n")

    return y_true, y_probs_fractured, y_pred


# =============================================================================
# 2. CLASSIFICATION REPORT
# =============================================================================

def print_classification_report(y_true, y_pred):
    """
    Print full sklearn classification report with per-class metrics.
    Also prints a summary highlighting recall as the priority metric.

    Args:
        y_true : true binary labels (1=fractured, 0=normal)
        y_pred : predicted binary labels using threshold
    """
    print("=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        y_true,
        y_pred,
        target_names=['Normal', 'Fractured'],
        digits=4,
    )
    print(report)

    # Extract key metrics for summary
    recall_frac    = float(classification_report(y_true, y_pred,
                           target_names=['Normal','Fractured'],
                           output_dict=True)['Fractured']['recall'])
    precision_frac = float(classification_report(y_true, y_pred,
                           target_names=['Normal','Fractured'],
                           output_dict=True)['Fractured']['precision'])
    f1_frac        = float(classification_report(y_true, y_pred,
                           target_names=['Normal','Fractured'],
                           output_dict=True)['Fractured']['f1-score'])
    overall_acc    = float(classification_report(y_true, y_pred,
                           target_names=['Normal','Fractured'],
                           output_dict=True)['accuracy'])

    print("=" * 60)
    print("  SUMMARY (Fractured class — our priority)")
    print("=" * 60)
    print(f"  Overall Accuracy  : {overall_acc*100:.2f}%")
    print(f"  Recall            : {recall_frac*100:.2f}%  ← priority metric")
    print(f"  Precision         : {precision_frac*100:.2f}%")
    print(f"  F1 Score          : {f1_frac*100:.2f}%")
    print(f"  Threshold used    : {config.THRESHOLD}  (not default 0.5)")
    print("=" * 60)

    # Medical AI note
    print("\n  📋 WHY RECALL IS OUR PRIORITY METRIC:")
    print("  " + "-"*54)
    print("  In medical AI for fracture detection:")
    print("  • False Negative = missed fracture = patient goes untreated")
    print("  • False Positive = extra scan ordered = minor inconvenience")
    print("  • Therefore: High Recall > High Precision")
    print("  • Threshold lowered to 0.4 to catch more fractures")
    print("    (model predicts 'Fractured' more often)")
    print("  " + "-"*54 + "\n")

    return {
        'accuracy'  : overall_acc,
        'recall'    : recall_frac,
        'precision' : precision_frac,
        'f1'        : f1_frac,
    }


# =============================================================================
# 3. CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, save=True):
    """
    Plot and save a styled confusion matrix.

    Args:
        y_true  : true labels
        y_pred  : predicted labels
        save    : save PNG to config.PLOTS_DIR
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages for annotation
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Build annotation labels: count + percentage
    annot = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot     = annot,
        fmt       = '',
        cmap      = 'Blues',
        xticklabels = ['Normal', 'Fractured'],
        yticklabels = ['Normal', 'Fractured'],
        linewidths  = 0.5,
        linecolor   = 'gray',
        cbar        = True,
        ax          = ax,
        annot_kws   = {'size': 13, 'weight': 'bold'},
    )

    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label',      fontsize=12, labelpad=10)

    # Color code: TN=blue, TP=blue, FN/FP=red border
    tn, fp, fn, tp = cm.ravel()
    fig.text(0.5, 0.01,
             f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  |  "
             f"Missed fractures (FN): {fn}",
             ha='center', fontsize=10, color='darkred')

    plt.tight_layout()

    if save:
        save_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVALUATE] Confusion matrix saved → {save_path}")

    plt.show()
    plt.close()

    # Print raw matrix values
    print(f"\n  Confusion Matrix Breakdown:")
    print(f"  True Negatives  (TN): {tn}  — correctly predicted Normal")
    print(f"  False Positives (FP): {fp}  — Normal predicted as Fractured")
    print(f"  False Negatives (FN): {fn}  — Fractured MISSED (most dangerous!)")
    print(f"  True Positives  (TP): {tp}  — correctly predicted Fractured\n")


# =============================================================================
# 4. ROC CURVE
# =============================================================================

def plot_roc_curve(y_true, y_probs, save=True):
    """
    Plot and save ROC curve with AUC score.

    Args:
        y_true  : true binary labels
        y_probs : predicted probabilities (not binary)
        save    : save PNG to config.PLOTS_DIR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    # Find point on curve closest to our operating threshold
    thresh_idx = np.argmin(np.abs(thresholds - config.THRESHOLD))

    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2.5,
            label=f'ROC Curve (AUC = {auc_score:.4f})')

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    # Mark operating threshold point
    ax.scatter(fpr[thresh_idx], tpr[thresh_idx],
               color='red', s=120, zorder=5,
               label=f'Threshold = {config.THRESHOLD} '
                     f'(TPR={tpr[thresh_idx]:.2f}, FPR={fpr[thresh_idx]:.2f})')

    ax.fill_between(fpr, tpr, alpha=0.1, color='blue')

    ax.set_title(f'ROC Curve  (AUC = {auc_score:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if save:
        save_path = os.path.join(config.PLOTS_DIR, 'roc_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVALUATE] ROC curve saved → {save_path}")

    plt.show()
    plt.close()

    print(f"  AUC Score: {auc_score:.4f}  "
          f"({'Excellent' if auc_score > 0.9 else 'Good' if auc_score > 0.8 else 'Fair'})\n")

    return auc_score


# =============================================================================
# 5. PRECISION-RECALL CURVE
# =============================================================================

def plot_precision_recall_curve(y_true, y_probs, save=True):
    """
    Plot and save Precision-Recall curve.
    More informative than ROC for imbalanced datasets.

    Args:
        y_true  : true binary labels
        y_probs : predicted probabilities
        save    : save PNG to config.PLOTS_DIR
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)

    # Baseline: random classifier on imbalanced data
    baseline = y_true.sum() / len(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall, precision, 'g-', linewidth=2.5,
            label=f'PR Curve (AP = {ap_score:.4f})')
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Baseline (random) = {baseline:.2f}')

    # Mark our operating threshold
    if len(thresholds) > 0:
        thresh_idx = np.argmin(np.abs(thresholds - config.THRESHOLD))
        ax.scatter(recall[thresh_idx], precision[thresh_idx],
                   color='red', s=120, zorder=5,
                   label=f'Threshold = {config.THRESHOLD}')

    ax.fill_between(recall, precision, alpha=0.1, color='green')

    ax.set_title(f'Precision-Recall Curve  (AP = {ap_score:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if save:
        save_path = os.path.join(config.PLOTS_DIR, 'precision_recall_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[EVALUATE] PR curve saved → {save_path}")

    plt.show()
    plt.close()

    return ap_score


# =============================================================================
# 6. FULL EVALUATION PIPELINE
# =============================================================================

def evaluate():
    """
    Run the complete evaluation pipeline:
    1. Load best saved model
    2. Build test generator
    3. Get predictions with threshold=0.4
    4. Print classification report
    5. Plot confusion matrix, ROC curve, PR curve
    6. Save all plots to outputs/plots/
    """

    print("\n" + "="*60)
    print("  BONE FRACTURE DETECTION — Model Evaluation")
    print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # Load model — prefers finetuned, falls back to best Phase 1
    # -------------------------------------------------------------------------
    model = load_model()

    # -------------------------------------------------------------------------
    # Build test generator (shuffle=False is critical here)
    # -------------------------------------------------------------------------
    print("[EVALUATE] Building test generator...")
    _, _, test_gen = build_generators()

    if test_gen.samples == 0:
        raise RuntimeError(
            "[EVALUATE] ❌ No test images found!\n"
            f"  Check: {config.TEST_DIR}\n"
            "  Expected subfolders: fractured/ and normal/"
        )

    print(f"[EVALUATE] Test set: {test_gen.samples} images")
    print(f"           Classes : {test_gen.class_indices}\n")

    # -------------------------------------------------------------------------
    # Get predictions
    # -------------------------------------------------------------------------
    y_true, y_probs, y_pred = get_predictions(model, test_gen)

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------
    metrics = print_classification_report(y_true, y_pred)

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------
    print("[EVALUATE] Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred)

    # -------------------------------------------------------------------------
    # ROC curve
    # -------------------------------------------------------------------------
    print("[EVALUATE] Plotting ROC curve...")
    auc_score = plot_roc_curve(y_true, y_probs)

    # -------------------------------------------------------------------------
    # Precision-Recall curve
    # -------------------------------------------------------------------------
    print("[EVALUATE] Plotting Precision-Recall curve...")
    ap_score = plot_precision_recall_curve(y_true, y_probs)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("  FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"  Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  Recall (Fracture) : {metrics['recall']*100:.2f}%  ← priority")
    print(f"  Precision (Frac.) : {metrics['precision']*100:.2f}%")
    print(f"  F1 (Fracture)     : {metrics['f1']*100:.2f}%")
    print(f"  AUC-ROC           : {auc_score:.4f}")
    print(f"  Avg Precision     : {ap_score:.4f}")
    print(f"  Threshold used    : {config.THRESHOLD}")
    print("="*60)

    # Performance verdict
    recall = metrics['recall']
    acc    = metrics['accuracy']
    if recall >= 0.90 and acc >= 0.85:
        verdict = "🟢 EXCELLENT — Ready for deployment demo"
    elif recall >= 0.80 and acc >= 0.75:
        verdict = "🟡 GOOD — Consider more training data for improvement"
    else:
        verdict = "🔴 NEEDS IMPROVEMENT — Try retraining with more fractured images"

    print(f"\n  Verdict: {verdict}")
    print(f"\n  Plots saved to: {config.PLOTS_DIR}")
    print("\n  Next step: python gradcam.py\n")

    return metrics, auc_score


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    evaluate()
