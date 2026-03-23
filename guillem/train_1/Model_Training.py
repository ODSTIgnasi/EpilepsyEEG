import Extract_Wavelet_Coefficients as EX  # pyright: ignore[reportMissingImports]
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, LSTM,
                                     BatchNormalization, Conv1D, MaxPooling1D)
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, accuracy_score,
                             cohen_kappa_score, roc_curve, auc, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os


def plot_roc_per_fold(y_true, y_pred_proba, fold_num):
    os.makedirs("plots", exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate',  fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver Operating Characteristic',
              fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(f"plots/roc_binary_fold{fold_num}.png")
    plt.close()


def build_chbmit_model(input_len):
    model = Sequential()

    # ── Conv block 1 — Table 2 row 1: F=16, k=2, s=2 ─────────────────────
    model.add(Conv1D(filters=16, kernel_size=2,
                     input_shape=(input_len, 1),
                     strides=2, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # ── Conv block 2 — Table 2 row 4: F=32, k=2, s=2 ─────────────────────
    model.add(Conv1D(filters=32, kernel_size=2, strides=2, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # ── Conv block 3 — Table 2 row 7: F=64, k=2, s=2 ─────────────────────
    model.add(Conv1D(filters=64, kernel_size=2, strides=2, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # ── Pool 1 — Table 2 row 10: pool_size=3, strides=2 ──────────────────
    # 9 → floor((9-3)/2)+1 = 4
    model.add(MaxPooling1D(pool_size=3, strides=2, padding="valid"))

    # ── Conv block 4 — Table 2 row 11: F=128, k=1, s=1 ───────────────────
    # Sequence stays at 4
    model.add(Conv1D(filters=128, kernel_size=1, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # ── NO Pool 2 here — would reduce 4 → 1, killing LSTM temporal context ─
    # Paper states architecture is flexible for different input lengths.
    # With a 72-scalar input we skip this pooling to preserve sequence length.

    # ── Conv block 5 — Table 2 row 15: F=256, k=1, s=1 ───────────────────
    # Sequence stays at 4
    model.add(Conv1D(filters=256, kernel_size=1, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # ── Conv block 6 — Table 2 row 19: F=512, k=1, s=1 ───────────────────
    # Sequence stays at 4
    model.add(Conv1D(filters=512, kernel_size=1, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # ── LSTM — Table 2 row 23: Units=200 ─────────────────────────────────
    # Input: (batch, 4, 512) — LSTM processes 4 time-steps
    # Output: (batch, 200)
    model.add(LSTM(200))

    # ── Dense — Table 2 row 25: Dense(64, ReLU, L2=0.03) ─────────────────
    model.add(Dense(units=64, activation="relu",
                    kernel_regularizer=regularizers.L2(0.03)))

    # ── Dropout — Table 2 row 26: Rate=0.4 ───────────────────────────────
    model.add(Dropout(rate=0.4))

    # ── Output — Table 2 rows 27-28: Dense(1) + sigmoid ──────────────────
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def binary_model(data, label, dataset_choice):
    os.makedirs("plots", exist_ok=True)

    # ── Hyperparameters from the paper (Experimental Setup section) ────────
    # "batch size of 60, 300 epochs, validation split of 0.15"
    # "10-fold cross-validation"
    # "learning rate of 0.0001"
    batch_size        = 60
    nb_epoch          = 300
    num_k_fold_splits = 10
    validation_split  = 0.15
    # ───────────────────────────────────────────────────────────────────────

    data  = np.array(data)
    label = np.array(label).astype(int)

    # Paper: "data is randomly shuffled before training and at end of each epoch"
    kf = StratifiedKFold(n_splits=num_k_fold_splits, shuffle=True, random_state=2)

    specificity_scores, sensitivity_scores = [], []
    npv_scores, ppv_scores                 = [], []
    f1_scores, mcc_scores                  = [], []
    accuracy_scores, precision_scores      = [], []
    recall_scores, auc_scores              = [], []
    Kappa_scores, GDR_scores               = [], []

    fold_num = 0
    history  = None

    for train_index, test_index in kf.split(data, label):

        tf.keras.backend.clear_session()

        X_tr_va, X_test = data[train_index], data[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]

        if dataset_choice == 'chbmit':
            # (N, 18, 1024, 1) → (N, 18, 1024) — squeeze the trailing dim added by load
            X_train   = X_tr_va.squeeze(-1)
            X_test_sq = X_test.squeeze(-1)
            Y_train   = y_tr_va
            Y_test    = y_test

            # DWT energy features: 18 channels × 4 subband energies = 72 scalars
            # Paper Fig 4b: "72×1 for MIT-CHB Dataset"
            # Wavelet: db1, level 3 (paper Results section p.13: "db1 mother wavelet")
            features_Train = EX.extract_coeffs_BONN_CHB(X_train,   level=3)
            features_Test  = EX.extract_coeffs_BONN_CHB(X_test_sq, level=3)

            n_features = features_Train.shape[1]   # 72
            print(f"Fold {fold_num+1} — features shape: {features_Train.shape}")

            # Reshape to (N, 72, 1) for Conv1D input
            EEG_Train = features_Train.reshape(features_Train.shape[0], n_features, 1)
            EEG_Test  = features_Test.reshape( features_Test.shape[0],  n_features, 1)

        # ── Class-weighted loss (paper: addresses CHB-MIT class imbalance) ──
        classes           = np.unique(Y_train)
        cw                = compute_class_weight(class_weight='balanced',
                                                 classes=classes, y=Y_train)
        class_weight_dict = dict(zip(classes.astype(int), cw))
        print(f"  Class weights: {class_weight_dict}")

        # ── Build model ─────────────────────────────────────────────────────
        if dataset_choice == 'chbmit':
            model = build_chbmit_model(input_len=n_features)

        # Paper: "Adam optimization, learning rate of 0.0001"
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['acc'])

        if fold_num == 0:
            model.summary()

        # ── Callbacks ───────────────────────────────────────────────────────
        # Paper does not mention early stopping — model runs full 300 epochs.
        # ReduceLROnPlateau helps stable convergence consistent with the paper's
        # note that LR sensitivity is high for this model.
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=20,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = model.fit(
            EEG_Train, Y_train,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=nb_epoch,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            shuffle=True,   # paper: "data is randomly shuffled before training"
            verbose=1
        )

        y_pred_proba = model.predict(EEG_Test).flatten()
        y_pred       = (y_pred_proba > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        npv         = tn / (tn + fn)
        ppv         = tp / (tp + fp)
        f1          = f1_score(Y_test, y_pred)
        mcc         = matthews_corrcoef(Y_test, y_pred)
        accuracy    = accuracy_score(Y_test, y_pred)
        precision   = precision_score(Y_test, y_pred)
        recall      = recall_score(Y_test, y_pred, average='weighted')
        kappa       = cohen_kappa_score(Y_test, y_pred)
        gdr         = np.sqrt(sensitivity * specificity)
        auc_val     = roc_auc_score(Y_test, y_pred_proba)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        specificity_scores.append(specificity)
        sensitivity_scores.append(sensitivity)
        npv_scores.append(npv)
        ppv_scores.append(ppv)
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        recall_scores.append(recall)
        Kappa_scores.append(kappa)
        GDR_scores.append(gdr)
        auc_scores.append(auc_val)

        plot_roc_per_fold(Y_test, y_pred_proba, fold_num)

        print(f"\n--- Fold {fold_num+1} / {num_k_fold_splits} ---")
        print(f"  Accuracy:    {accuracy*100:.2f}%")
        print(f"  Precision:   {precision*100:.2f}%")
        print(f"  Specificity: {specificity*100:.2f}%")
        print(f"  Sensitivity: {sensitivity*100:.2f}%")
        print(f"  NPV:         {npv*100:.2f}%")
        print(f"  PPV:         {ppv*100:.2f}%")
        print(f"  F1:          {f1*100:.2f}%")
        print(f"  MCC:         {mcc*100:.2f}%")
        print(f"  Kappa:       {kappa*100:.2f}%")
        print(f"  GDR:         {gdr*100:.2f}%")
        print(f"  AUC:         {auc_val*100:.2f}%")

        fold_num += 1

    print("\n========== Final Results — CHB-MIT (Mean ± SD, 10 folds) ==========")
    print(f"Accuracy:    {np.mean(accuracy_scores)*100:.2f}% ± {np.std(accuracy_scores)*100:.2f}%  (paper: 96.94±1.22%)")
    print(f"Precision:   {np.mean(precision_scores)*100:.2f}% ± {np.std(precision_scores)*100:.2f}%  (paper: 95.43±1.23%)")
    print(f"Specificity: {np.mean(specificity_scores)*100:.2f}% ± {np.std(specificity_scores)*100:.2f}%  (paper: 98.12±1.28%)")
    print(f"Sensitivity: {np.mean(sensitivity_scores)*100:.2f}% ± {np.std(sensitivity_scores)*100:.2f}%  (paper: 92.21±1.17%)")
    print(f"NPV:         {np.mean(npv_scores)*100:.2f}% ± {np.std(npv_scores)*100:.2f}%  (paper: 96.83±1.20%)")
    print(f"PPV:         {np.mean(ppv_scores)*100:.2f}% ± {np.std(ppv_scores)*100:.2f}%  (paper: 95.30±1.29%)")
    print(f"F1:          {np.mean(f1_scores)*100:.2f}% ± {np.std(f1_scores)*100:.2f}%  (paper: 93.31±1.05%)")
    print(f"MCC:         {np.mean(mcc_scores)*100:.2f}% ± {np.std(mcc_scores)*100:.2f}%  (paper: 91.15±1.12%)")
    print(f"Recall:      {np.mean(recall_scores)*100:.2f}% ± {np.std(recall_scores)*100:.2f}%")
    print(f"Kappa:       {np.mean(Kappa_scores)*100:.2f}% ± {np.std(Kappa_scores)*100:.2f}%  (paper: 94.33%)")
    print(f"GDR:         {np.mean(GDR_scores)*100:.2f}% ± {np.std(GDR_scores)*100:.2f}%  (paper: 96.36%)")
    print(f"AUC:         {np.mean(auc_scores)*100:.2f}% ± {np.std(auc_scores)*100:.2f}%")

    return history
