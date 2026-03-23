"""
CNN Model for EEG Seizure Detection -- CHB-MIT Dataset

Input shape : (n_windows, 21, 128, 1)
              21 channels, 128 time samples, 1 depth channel
Output      : binary (0 = non-seizure, 1 = seizure)

Usage:
    from model import build_model, INPUT_SHAPE
    model = build_model()
    model.summary()
"""

import keras
from keras import layers
import numpy as np

# ── Input shape matches extracted data: (21 channels, 128 time steps, 1) ──────
INPUT_SHAPE = (21, 128, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SeizureCNN(keras.Model):

    def __init__(self):
        super().__init__()

        # ── Block 1: capture short temporal patterns ───────────────────────
        # kernel (1, 16) slides across time only -- 16 samples = 125ms at 128Hz
        self.conv1 = layers.Conv2D(32, kernel_size=(1, 16), padding="same")
        self.bn1   = layers.BatchNormalization()
        self.act1  = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(pool_size=(1, 4))   # 128 -> 32 time steps
        self.drop1 = layers.Dropout(0.25)

        # ── Block 2: capture longer patterns across more filters ───────────
        self.conv2 = layers.Conv2D(64, kernel_size=(1, 8), padding="same")
        self.bn2   = layers.BatchNormalization()
        self.act2  = layers.ReLU()
        self.pool2 = layers.MaxPooling2D(pool_size=(1, 4))   # 32 -> 8 time steps
        self.drop2 = layers.Dropout(0.25)

        # ── Block 3: capture spatial relationships across channels ─────────
        # kernel (3, 1) slides across channels -- captures inter-channel patterns
        self.conv3 = layers.Conv2D(128, kernel_size=(3, 1), padding="same")
        self.bn3   = layers.BatchNormalization()
        self.act3  = layers.ReLU()
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 1))   # 21 -> 10 channels
        self.drop3 = layers.Dropout(0.25)

        # ── Classifier head ────────────────────────────────────────────────
        # GlobalAveragePooling averages across spatial dims -- more robust than Flatten
        self.gap    = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(64, activation="relu")
        self.drop4  = layers.Dropout(0.5)
        self.out    = layers.Dense(1, activation="sigmoid")  # binary output

    def call(self, x, training=False):
        # training=True/False controls BatchNorm and Dropout behaviour

        # Block 1 -- temporal features
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        # Block 2 -- longer temporal features
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        # Block 3 -- spatial (cross-channel) features
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        # Classifier head
        x = self.gap(x)
        x = self.dense1(x)
        x = self.drop4(x, training=training)
        x = self.out(x)

        return x


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_model(learning_rate=1e-3):
    """
    Build, compile and return the SeizureCNN model.

    Args:
        learning_rate : Adam learning rate (default 1e-3)

    Returns:
        compiled keras.Model
    """
    model = SeizureCNN()

    # Pass a dummy batch so Keras resolves all layer shapes before summary
    dummy = np.zeros((1,) + INPUT_SHAPE)
    model(dummy)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT -- print summary when run directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = build_model()
    model.summary()