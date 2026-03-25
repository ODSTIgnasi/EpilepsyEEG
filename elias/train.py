"""
Training script for EEG Seizure CNN -- CHB-MIT Dataset

Loads train/val splits from the extraction script, trains the model
with early stopping, and saves the best checkpoint.

Usage:
    python train.py
"""

import numpy as np
import os
import keras

from cnn_model import build_model, INPUT_SHAPE

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR  = "/hhome/ricse01/DL/elias/Data"
MODEL_DIR = "/hhome/ricse01/DL/elias/models"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE    = 32
MAX_EPOCHS    = 30
LEARNING_RATE = 1e-3

# Early stopping -- stop training when val_loss doesn't improve for this many epochs
PATIENCE = 5


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

#train = np.load(os.path.join(DATA_DIR, "train.npz"))
#val   = np.load(os.path.join(DATA_DIR, "val.npz"))

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"), allow_pickle=True)
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)

X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"), allow_pickle=True)
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"), allow_pickle=True)


print(f"X_train : {X_train.shape}  |  seizure: {y_train.sum()} ({100*y_train.mean():.1f}%)")
print(f"X_val   : {X_val.shape}    |  seizure: {y_val.sum()} ({100*y_val.mean():.1f}%)")

# Verify shapes match what the model expects
assert X_train.shape[1:] == INPUT_SHAPE, (
    f"Shape mismatch: data is {X_train.shape[1:]} but model expects {INPUT_SHAPE}"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. HANDLE CLASS IMBALANCE
# non-seizure windows will far outnumber seizure windows -- weight them equally
# ══════════════════════════════════════════════════════════════════════════════

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
class_weight = {
    0: 1.0,
    1: n_neg / (n_pos + 1e-8)   # upweight the minority seizure class
}

print(f"\nClass weights: non-seizure={class_weight[0]:.2f}  seizure={class_weight[1]:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

model = build_model(learning_rate=LEARNING_RATE)
model.summary()


# ══════════════════════════════════════════════════════════════════════════════
# 4. CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

callbacks = [

    # Save the best model based on validation AUC
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    # Stop training when val_loss hasn't improved for PATIENCE epochs
    # restore_best_weights=True rewinds the weights to the best epoch on stop
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),

    # Halve the learning rate when val_loss plateaus for 3 epochs
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    # TensorBoard logs -- run: tensorboard --logdir /hhome/ricse/Epilepsy/models/logs
    keras.callbacks.TensorBoard(
        log_dir=os.path.join(MODEL_DIR, "logs"),
        histogram_freq=1
    ),

]


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE FINAL MODEL + HISTORY
# ══════════════════════════════════════════════════════════════════════════════

final_path = os.path.join(MODEL_DIR, "final_model.keras")
model.save(final_path)

import pandas as pd
pd.DataFrame(history.history).to_csv(
    os.path.join(MODEL_DIR, "history.csv"), index=False
)

# ── Training summary ──────────────────────────────────────────────────────────
best_epoch  = int(np.argmax(history.history["val_auc"])) + 1
best_auc    = max(history.history["val_auc"])
best_loss   = min(history.history["val_loss"])
ran_epochs  = len(history.history["loss"])

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Epochs run     : {ran_epochs} / {MAX_EPOCHS}"
      f"{'  (early stop)' if ran_epochs < MAX_EPOCHS else ''}")
print(f"Best epoch     : {best_epoch}")
print(f"Best val AUC   : {best_auc:.4f}")
print(f"Best val loss  : {best_loss:.4f}")
print(f"\nSaved:")
print(f"  Best model   -> {MODEL_DIR}/best_model.keras")
print(f"  Final model  -> {final_path}")
print(f"  History      -> {MODEL_DIR}/history.csv")
print()
print("To load the best model in your test script:")
print("  model = keras.models.load_model('./models/best_model.keras')")