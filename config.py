# =============================================================================
# config.py — Central Configuration for Bone Fracture Detection System
# =============================================================================
# All hyperparameters, paths, and settings live here.
# Import this file in every other module — never hardcode values elsewhere.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# GOOGLE COLAB SETUP (uncomment if running on Colab)
# -----------------------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')
# BASE_DIR = '/content/drive/MyDrive/bone_fracture_detection'
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# BASE DIRECTORY — change this to your project root
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# DATASET PATHS
# Expected folder structure:
#   dataset/
#     train/
#       fractured/
#       normal/
#     val/
#       fractured/
#       normal/
#     test/
#       fractured/
#       normal/
# -----------------------------------------------------------------------------
DATASET_DIR   = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR     = os.path.join(DATASET_DIR, 'train')
VAL_DIR       = os.path.join(DATASET_DIR, 'val')
TEST_DIR      = os.path.join(DATASET_DIR, 'test')

# -----------------------------------------------------------------------------
# MODEL SAVE PATHS
# -----------------------------------------------------------------------------
MODELS_DIR          = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH     = os.path.join(MODELS_DIR, 'best_model.h5')       # saved during training (phase 1)
FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, 'finetuned_model.h5') # saved after fine-tuning (phase 2)

# -----------------------------------------------------------------------------
# OUTPUT PATHS (plots, heatmaps, evaluation results)
# -----------------------------------------------------------------------------
OUTPUTS_DIR     = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR       = os.path.join(OUTPUTS_DIR, 'plots')
HEATMAPS_DIR    = os.path.join(OUTPUTS_DIR, 'heatmaps')

# Flask static folder for serving heatmap images
STATIC_DIR      = os.path.join(BASE_DIR, 'static')
STATIC_UPLOADS  = os.path.join(STATIC_DIR, 'uploads')
STATIC_HEATMAPS = os.path.join(STATIC_DIR, 'heatmaps')

# -----------------------------------------------------------------------------
# IMAGE SETTINGS
# ResNet50 expects 224x224 RGB images
# -----------------------------------------------------------------------------
IMAGE_SIZE    = (224, 224)          # (height, width) — ResNet50 input size
IMAGE_CHANNELS = 3                  # RGB
INPUT_SHAPE   = (*IMAGE_SIZE, IMAGE_CHANNELS)  # (224, 224, 3)

# -----------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
BATCH_SIZE    = 32      # number of images per gradient update
EPOCHS        = 30      # total epochs (split across phase 1 & 2)
PHASE1_EPOCHS = 15      # phase 1: train with frozen base
PHASE2_EPOCHS = 15      # phase 2: fine-tune unfrozen layers

# Learning rates
LEARNING_RATE   = 1e-4  # initial LR for phase 1 (Adam optimizer)
FINE_TUNE_LR    = 1e-5  # lower LR for fine-tuning (phase 2)

# ResNet50 layer fine-tuning
# Layers after index FINE_TUNE_AT will be unfrozen during phase 2
# ResNet50 has 175 layers total; unfreezing ~75 last layers
FINE_TUNE_AT  = 100

# Regularization
DROPOUT       = 0.3     # dropout rate applied after Dense layers

# -----------------------------------------------------------------------------
# CLASSIFICATION THRESHOLD
# Using 0.4 instead of 0.5 to PRIORITIZE RECALL over Precision.
# In medical AI, a missed fracture (False Negative) is more dangerous
# than a false alarm (False Positive). Lower threshold → more "Fractured" 
# predictions → higher sensitivity/recall.
# -----------------------------------------------------------------------------
THRESHOLD     = 0.4

# Class labels (match folder names in dataset)
CLASS_NAMES   = ['normal', 'fractured']   # index 0 = normal, index 1 = fractured

# -----------------------------------------------------------------------------
# AUGMENTATION SETTINGS (used in preprocess.py)
# -----------------------------------------------------------------------------
AUGMENTATION = {
    'rotation_range'      : 15,           # rotate ±15 degrees
    'zoom_range'          : 0.10,         # zoom in/out up to 10%
    'width_shift_range'   : 0.10,         # horizontal shift up to 10%
    'height_shift_range'  : 0.10,         # vertical shift up to 10%
    'horizontal_flip'     : True,         # mirror images horizontally
    'brightness_range'    : [0.85, 1.15], # adjust brightness ±15%
    'fill_mode'           : 'nearest',    # fill empty pixels after transforms
}

# -----------------------------------------------------------------------------
# EARLY STOPPING & CALLBACKS
# -----------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE  = 5    # stop if val_loss doesn't improve for 5 epochs
REDUCE_LR_PATIENCE       = 3    # reduce LR if val_loss stagnates for 3 epochs
REDUCE_LR_FACTOR         = 0.5  # multiply LR by this factor on plateau
REDUCE_LR_MIN            = 1e-7 # minimum LR floor

# Monitor metric for ModelCheckpoint and EarlyStopping
MONITOR_METRIC = 'val_loss'

# -----------------------------------------------------------------------------
# GRAD-CAM SETTINGS
# The last convolutional layer in ResNet50 before GlobalAveragePooling
# -----------------------------------------------------------------------------
GRADCAM_LAYER  = 'conv5_block3_out'  # last conv layer name in ResNet50
GRADCAM_ALPHA  = 0.4                 # blending weight: heatmap over original

# -----------------------------------------------------------------------------
# FLASK APP SETTINGS
# -----------------------------------------------------------------------------
FLASK_DEBUG         = True
FLASK_PORT          = 5000
FLASK_HOST          = '0.0.0.0'
MAX_CONTENT_LENGTH  = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS  = {'jpg', 'jpeg', 'png'}

# Medical disclaimer shown on every results page
MEDICAL_DISCLAIMER = (
    "⚠️ DISCLAIMER: This tool is AI-assisted and intended for educational/research "
    "purposes only. It is NOT a substitute for professional medical diagnosis. "
    "Always consult a qualified radiologist or physician for medical decisions."
)

# -----------------------------------------------------------------------------
# SEED — for reproducibility across NumPy, TensorFlow, Python random
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# UTILITY: auto-create required directories on import
# -----------------------------------------------------------------------------
def create_directories():
    """Create all necessary project directories if they don't exist."""
    dirs = [
        MODELS_DIR,
        OUTPUTS_DIR,
        PLOTS_DIR,
        HEATMAPS_DIR,
        STATIC_DIR,
        STATIC_UPLOADS,
        STATIC_HEATMAPS,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[CONFIG] ✅ All directories verified/created under: {BASE_DIR}")


# Run directory creation whenever config is imported
create_directories()


# -----------------------------------------------------------------------------
# SANITY CHECK — run this file directly to verify all paths
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  BONE FRACTURE DETECTION — Configuration Summary")
    print("="*60)
    print(f"  Base Dir         : {BASE_DIR}")
    print(f"  Dataset Dir      : {DATASET_DIR}")
    print(f"  Train Dir        : {TRAIN_DIR}")
    print(f"  Val Dir          : {VAL_DIR}")
    print(f"  Test Dir         : {TEST_DIR}")
    print(f"  Best Model Path  : {BEST_MODEL_PATH}")
    print(f"  Fine-tuned Model : {FINETUNED_MODEL_PATH}")
    print(f"  Plots Dir        : {PLOTS_DIR}")
    print(f"  Heatmaps Dir     : {HEATMAPS_DIR}")
    print("-"*60)
    print(f"  Image Size       : {IMAGE_SIZE}")
    print(f"  Batch Size       : {BATCH_SIZE}")
    print(f"  Phase 1 Epochs   : {PHASE1_EPOCHS}")
    print(f"  Phase 2 Epochs   : {PHASE2_EPOCHS}")
    print(f"  Learning Rate    : {LEARNING_RATE}")
    print(f"  Fine-Tune LR     : {FINE_TUNE_LR}")
    print(f"  Fine-Tune At     : Layer {FINE_TUNE_AT}")
    print(f"  Dropout          : {DROPOUT}")
    print(f"  Threshold        : {THRESHOLD}  ← recall-optimized")
    print(f"  Grad-CAM Layer   : {GRADCAM_LAYER}")
    print(f"  Random Seed      : {RANDOM_SEED}")
    print("="*60 + "\n")

    # Warn if dataset directories don't exist yet
    for name, path in [('Train', TRAIN_DIR), ('Val', VAL_DIR), ('Test', TEST_DIR)]:
        status = "✅ Found" if os.path.isdir(path) else "⚠️  NOT FOUND — download dataset first"
        print(f"  {name:5s} Dir: {status}")
    print()
