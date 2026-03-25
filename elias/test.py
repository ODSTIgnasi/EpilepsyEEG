"""
Test script for EEG Seizure CNN

Evaluates the trained model on test data using:
- Accuracy
- Precision
- Recall
- F1-score
- False Positive Rate (FPR)
- ROC Curve
- AUC

Usage:
    python test.py
"""

import numpy as np
import os
import keras
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR  = "/hhome/ricse01/DL/elias/Data"
MODEL_DIR = "/hhome/ricse01/DL/elias/models"

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

# Threshold for classification
THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"), allow_pickle=True)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)

print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape} | seizure: {y_test.sum()} ({100*y_test.mean():.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading model...")
model = keras.models.load_model(MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\nRunning predictions...")

y_prob = model.predict(X_test).ravel()  # probabilities
y_pred = (y_prob >= THRESHOLD).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════════════════════

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Confusion matrix → TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# False Positive Rate
fpr_value = fp / (fp + tn + 1e-8)

print("\n" + "=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1-score           : {f1:.4f}")
print(f"False Positive Rate: {fpr_value:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROC + AUC
# ══════════════════════════════════════════════════════════════════════════════

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"AUC                : {roc_auc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOT ROC CURVE
# ══════════════════════════════════════════════════════════════════════════════

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
plt.show()

print(f"\nROC curve saved to: {MODEL_DIR}/roc_curve.png")