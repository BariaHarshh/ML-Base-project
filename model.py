# =============================================================================
# model.py — ResNet50 Transfer Learning Model for Bone Fracture Detection
# =============================================================================
# Phase 1: Load ResNet50 (pretrained on ImageNet), freeze all base layers,
#           train only the custom classification head.
# Phase 2: Unfreeze layers after FINE_TUNE_AT index, retrain with lower LR.
#
# Architecture:
#   ResNet50 (frozen base)
#     → GlobalAveragePooling2D
#     → Dense(256, relu) → Dropout(0.3)
#     → Dense(128, relu) → Dropout(0.3)
#     → Dense(1, sigmoid)   ← binary output: 0=normal, 1=fractured
# =============================================================================

# -----------------------------------------------------------------------------
# GOOGLE COLAB NOTE:
# GPU check — run this at the top of your Colab notebook:
#   import tensorflow as tf
#   print("GPU:", tf.config.list_physical_devices('GPU'))
# If empty, go to Runtime → Change runtime type → GPU
# -----------------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

import config  # all hyperparameters from central config


# =============================================================================
# 1. GPU CONFIGURATION
# =============================================================================

def configure_gpu():
    """
    Configure GPU memory growth to prevent TensorFlow from allocating
    all GPU memory at once. Especially important on shared machines / Colab.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[MODEL] ✅ GPU configured: {len(gpus)} GPU(s) available")
            for g in gpus:
                print(f"         → {g.name}")
        except RuntimeError as e:
            print(f"[MODEL] ⚠️  GPU config error: {e}")
    else:
        print("[MODEL] ⚠️  No GPU found — training on CPU (will be slow)")
        print("         Consider using Google Colab with GPU runtime.")


# =============================================================================
# 2. BUILD MODEL (Phase 1 — frozen base)
# =============================================================================

def build_model():
    """
    Build the full classification model using ResNet50 as a frozen feature extractor.

    Architecture:
        Input(224, 224, 3)
        → ResNet50 (pretrained, all layers frozen)
        → GlobalAveragePooling2D
        → Dense(256, relu) + BatchNorm + Dropout(0.3)
        → Dense(128, relu) + BatchNorm + Dropout(0.3)
        → Dense(1, sigmoid)

    Returns:
        model (tf.keras.Model): compiled Keras model ready for Phase 1 training
    """

    # -------------------------------------------------------------------------
    # Load ResNet50 pretrained on ImageNet
    # include_top=False removes the final classification layers (Dense 1000)
    # so we can attach our own binary classification head.
    # input_shape must match config.INPUT_SHAPE = (224, 224, 3)
    # -------------------------------------------------------------------------
    base_model = ResNet50(
        weights      = 'imagenet',
        include_top  = False,
        input_shape  = config.INPUT_SHAPE,
    )

    # -------------------------------------------------------------------------
    # FREEZE all base model layers for Phase 1
    # We only train our custom head first — the ImageNet weights are good
    # feature extractors already. Training the base too early on a small
    # dataset would destroy those learned features (catastrophic forgetting).
    # -------------------------------------------------------------------------
    base_model.trainable = False

    frozen_count = sum(1 for layer in base_model.layers if not layer.trainable)
    print(f"[MODEL] Base model loaded: ResNet50")
    print(f"        Total layers      : {len(base_model.layers)}")
    print(f"        Frozen layers     : {frozen_count}")
    print(f"        Trainable params  : {base_model.count_params():,} (all frozen for Phase 1)")

    # -------------------------------------------------------------------------
    # Build custom classification head
    # GlobalAveragePooling2D converts feature maps (7x7x2048) → (2048,)
    # This is better than Flatten for preventing overfitting.
    # BatchNormalization stabilizes training and acts as light regularization.
    # -------------------------------------------------------------------------
    inputs = Input(shape=config.INPUT_SHAPE, name='input_layer')

    # Pass through frozen ResNet50 — training=False ensures BatchNorm layers
    # in ResNet50 stay in inference mode even during our training
    x = base_model(inputs, training=False)

    # Pooling: (batch, 7, 7, 2048) → (batch, 2048)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # First dense block
    x = Dense(256, activation='relu', name='dense_256',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn_256')(x)
    x = Dropout(config.DROPOUT, name='dropout_256')(x)

    # Second dense block
    x = Dense(128, activation='relu', name='dense_128',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn_128')(x)
    x = Dropout(config.DROPOUT, name='dropout_128')(x)

    # Output layer — sigmoid for binary classification
    # Output ∈ [0, 1]: probability of being "fractured"
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    # Assemble model
    model = Model(inputs=inputs, outputs=outputs, name='BoneFractureDetector_v1')

    # -------------------------------------------------------------------------
    # Compile the model
    # Loss: binary_crossentropy — standard for sigmoid binary classification
    # Metrics: accuracy + AUC + Recall + Precision for medical evaluation
    # -------------------------------------------------------------------------
    model.compile(
        optimizer = Adam(learning_rate=config.LEARNING_RATE),
        loss      = 'binary_crossentropy',
        metrics   = [
            'accuracy',
            AUC(name='auc'),
            Recall(name='recall'),
            Precision(name='precision'),
        ]
    )

    print(f"\n[MODEL] ✅ Model compiled for Phase 1 (frozen base)")
    print(f"        Learning rate     : {config.LEARNING_RATE}")
    print(f"        Loss              : binary_crossentropy")
    print(f"        Threshold         : {config.THRESHOLD} (recall-optimized)\n")

    return model


# =============================================================================
# 3. FINE-TUNE MODEL (Phase 2 — unfreeze top layers)
# =============================================================================

def fine_tune_model(model):
    """
    Unfreeze ResNet50 layers after FINE_TUNE_AT index and recompile
    with a lower learning rate for Phase 2 fine-tuning.

    Why fine-tune?
    - After Phase 1, our custom head is well-trained.
    - Now we can carefully update the top ResNet50 layers to learn
      fracture-specific features (not just generic ImageNet patterns).
    - Lower LR (1e-5) prevents destroying the pretrained weights.

    Args:
        model (tf.keras.Model): model returned by build_model() after Phase 1

    Returns:
        model (tf.keras.Model): same model, recompiled with unfrozen top layers
    """

    # Get the ResNet50 base model layer (it's the 2nd layer: index 1)
    # layers[0] = input, layers[1] = base_model
    base_model = model.layers[1]

    # Unfreeze the entire base model first
    base_model.trainable = True

    # Re-freeze all layers UP TO FINE_TUNE_AT
    # Only layers AFTER this index will be updated
    for layer in base_model.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen throughout fine-tuning
    # Unfreezing BN layers can destabilize training on small datasets
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Count trainable vs frozen
    trainable   = sum(1 for l in model.layers if l.trainable)
    untrainable = sum(1 for l in model.layers if not l.trainable)
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )

    print(f"[MODEL] Fine-tuning configuration:")
    print(f"        Unfreezing layers after index : {config.FINE_TUNE_AT}")
    print(f"        Trainable params              : {trainable_params:,}")
    print(f"        Frozen params                 : {total_params - trainable_params:,}")
    print(f"        New learning rate             : {config.FINE_TUNE_LR}")

    # Recompile with lower LR — critical, must recompile after changing trainability
    model.compile(
        optimizer = Adam(learning_rate=config.FINE_TUNE_LR),
        loss      = 'binary_crossentropy',
        metrics   = [
            'accuracy',
            AUC(name='auc'),
            Recall(name='recall'),
            Precision(name='precision'),
        ]
    )

    print(f"[MODEL] ✅ Model recompiled for Phase 2 (fine-tuning)\n")
    return model


# =============================================================================
# 4. MODEL SUMMARY PRINTER
# =============================================================================

def print_model_summary(model):
    """
    Print a clean model summary with layer-by-layer breakdown.
    Also prints total vs trainable parameter counts.
    """
    print("\n" + "="*65)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("="*65)
    model.summary(line_length=65)

    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen_params    = total_params - trainable_params

    print("="*65)
    print(f"  Total parameters     : {total_params:>12,}")
    print(f"  Trainable parameters : {trainable_params:>12,}")
    print(f"  Frozen parameters    : {frozen_params:>12,}")
    print("="*65 + "\n")


# =============================================================================
# 5. MODEL LOADER (used by evaluate.py, gradcam.py, app.py)
# =============================================================================

def load_model(model_path=None):
    """
    Load a saved model from disk.
    Defaults to config.FINETUNED_MODEL_PATH (best model after both phases).
    Falls back to config.BEST_MODEL_PATH if fine-tuned version doesn't exist.

    Args:
        model_path (str, optional): explicit path to .h5 file

    Returns:
        model (tf.keras.Model): loaded and ready for inference

    Raises:
        FileNotFoundError: if neither model file exists
    """
    if model_path is None:
        # Prefer the fine-tuned model; fall back to phase 1 best
        if os.path.exists(config.FINETUNED_MODEL_PATH):
            model_path = config.FINETUNED_MODEL_PATH
        elif os.path.exists(config.BEST_MODEL_PATH):
            model_path = config.BEST_MODEL_PATH
            print(f"[MODEL] ⚠️  Fine-tuned model not found, using Phase 1 model.")
        else:
            raise FileNotFoundError(
                f"[MODEL] ❌ No saved model found.\n"
                f"  Expected: {config.FINETUNED_MODEL_PATH}\n"
                f"  Or:       {config.BEST_MODEL_PATH}\n"
                f"  Run train.py first to train and save the model."
            )

    print(f"[MODEL] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"[MODEL] ✅ Model loaded successfully.")
    return model


# =============================================================================
# 6. ARCHITECTURE VISUALIZER (optional — saves model diagram)
# =============================================================================

def plot_model_architecture(model, save_path=None):
    """
    Save a visual diagram of the model architecture as a PNG.
    Requires: pip install pydot graphviz

    Args:
        model     : compiled Keras model
        save_path : path to save PNG (default: outputs/plots/model_architecture.png)
    """
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, 'model_architecture.png')

    try:
        tf.keras.utils.plot_model(
            model,
            to_file    = save_path,
            show_shapes        = True,
            show_layer_names   = True,
            show_dtype         = False,
            expand_nested      = False,
            dpi                = 96,
        )
        print(f"[MODEL] Model diagram saved → {save_path}")
    except Exception as e:
        print(f"[MODEL] ⚠️  Could not save model diagram (install pydot + graphviz): {e}")


# =============================================================================
# QUICK SANITY CHECK — run this file directly to verify model builds correctly
# =============================================================================

if __name__ == '__main__':
    print("\n[MODEL] Running sanity check...\n")

    # Configure GPU
    configure_gpu()

    # Set seed for reproducibility
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Build Phase 1 model
    print("\n--- Phase 1: Building frozen base model ---")
    model = build_model()
    print_model_summary(model)

    # Test forward pass with a dummy batch
    dummy_input = np.random.rand(2, *config.INPUT_SHAPE).astype(np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    print(f"[MODEL] Forward pass test:")
    print(f"        Input shape  : {dummy_input.shape}")
    print(f"        Output shape : {dummy_output.shape}")
    print(f"        Output values: {dummy_output.flatten()}  ← should be in [0, 1]")
    assert dummy_output.shape == (2, 1), "Output shape mismatch!"
    assert np.all((dummy_output >= 0) & (dummy_output <= 1)), "Output not in [0,1]!"

    # Simulate Phase 2 fine-tuning setup
    print("\n--- Phase 2: Configuring fine-tuning ---")
    model = fine_tune_model(model)

    # Optional: save architecture diagram
    plot_model_architecture(model)

    print("\n[MODEL] ✅ All checks passed. Ready to train!\n")
    print("  Next step: python train.py\n")
