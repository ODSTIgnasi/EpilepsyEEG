import Extract_Wavelet_Coefficients as EX  # pyright: ignore[reportMissingImports]
import numpy as np

# ── Single import block: tensorflow.keras only ───────────────────────────────
# FIX: removed the duplicate plain-keras imports that were silently overriding
# each other (keras.models, keras.layers imported first, then tensorflow.keras
# imported again below — only the second set was actually used).
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, LSTM,
                                     BatchNormalization, Conv1D, MaxPooling1D)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, accuracy_score,
                             cohen_kappa_score, roc_curve, auc, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def plot_roc_per_fold(y_true, y_pred_proba, fold_num):
    import os
    os.makedirs("plots", exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',  fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate',   fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver Operating Characteristic',
              fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(f"plots/roc_binary_fold{fold_num}.png")
    plt.close()


def binary_model(data, label, dataset_choice):
    import os
    os.makedirs("plots", exist_ok=True)

    # ── Hyperparameters (paper: Experimental Setup section + Table 2) ────────
    batch_size        = 60
    nb_epoch          = 50
    num_k_fold_splits = 3
    validation_split  = 0.15   # paper: "15% is further split for validation"
    # ─────────────────────────────────────────────────────────────────────────

    data  = np.array(data)
    label = np.array(label)

    kf = StratifiedKFold(n_splits=num_k_fold_splits, shuffle=True, random_state=2)

    specificity_scores = []
    sensitivity_scores = []
    npv_scores         = []
    ppv_scores         = []
    f1_scores          = []
    mcc_scores         = []
    accuracy_scores    = []
    precision_scores   = []
    recall_scores      = []
    auc_scores         = []
    Kappa_scores       = []
    GDR_scores         = []

    fold_num = 0
    history  = None

    for train_index, test_index in kf.split(data, label):

        tf.keras.backend.clear_session()

        X_tr_va, X_test = data[train_index], data[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]

        if dataset_choice == 'chbmit':
            # (N, 18, 512, 1) → (N, 18, 512)
            X_train   = X_tr_va.squeeze(-1)
            X_test_sq = X_test.squeeze(-1)
            Y_train   = y_tr_va
            Y_test    = y_test

            # DWT: 18 channels × 4 subband energies = 72 features per sample
            # EX.extract_coeffs_BONN_CHB now uses 'db3' and returns (N, 72)
            features_Train = EX.extract_coeffs_BONN_CHB(X_train,   level=3)
            features_Test  = EX.extract_coeffs_BONN_CHB(X_test_sq, level=3)

            n_features = features_Train.shape[1]   # should be 72
            print(f"Fold {fold_num+1} — features shape: {features_Train.shape}")

            # Reshape to (N, 72, 1) for Conv1D input
            EEG_Train = features_Train.reshape(features_Train.shape[0], n_features, 1)
            EEG_Test  = features_Test.reshape( features_Test.shape[0],  n_features, 1)

        # ── Class-weighted loss ───────────────────────────────────────────────
        classes           = np.unique(Y_train)
        cw                = compute_class_weight(class_weight='balanced',
                                                 classes=classes, y=Y_train)
        class_weight_dict = dict(zip(classes.astype(int), cw))
        print(f"  Class weights: {class_weight_dict}")

        # ── Architecture (paper Table 2, adapted for 72-feature CHB-MIT input)
        #
        # Dimension trace for 72-feature input:
        #   72 → Conv1(k=2,s=2) → 36 → Conv2(k=2,s=2) → 18
        #      → Conv3(k=2,s=2) →  9 → Pool1(p=3,s=2) →  4
        #      → Conv4(k=1,s=1) →  4 → Pool2(p=3,s=2) →  1
        #
        # Pool2 reduces to 1 time-step → LSTM(200) returns shape (batch, 200)
        # Paper Table 2 (BONN input 4100×1) has 4 pooling layers; with the
        # 72-feature CHB-MIT input only 2 pooling layers are needed to reach
        # a manageable spatial dimension before the LSTM.  The paper notes the
        # architecture is flexible and adapts to different input lengths.
        # ─────────────────────────────────────────────────────────────────────

        model = Sequential()

        # Conv block 1  — 72 → 36
        model.add(Conv1D(filters=16, kernel_size=2,
                         input_shape=(EEG_Train.shape[1], 1),
                         strides=2, padding="valid"))
        model.add(BatchNormalization())   # FIX: removed axis=2 (default -1 is correct)
        model.add(Activation('relu'))

        # Conv block 2  — 36 → 18
        model.add(Conv1D(filters=32, kernel_size=2, strides=2, padding="valid"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Conv block 3  — 18 → 9
        model.add(Conv1D(filters=64, kernel_size=2, strides=2, padding="valid"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Pool 1  — 9 → 4  (paper: pool_size=3, strides=2)
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="valid"))

        # Conv block 4  — 4 → 4
        model.add(Conv1D(filters=128, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Pool 2  — 4 → 1  (pool_size=3, strides=2: floor((4-3)/2)+1 = 1)
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="valid"))

        # Conv block 5  — 1 → 1
        model.add(Conv1D(filters=256, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Conv block 6  — 1 → 1
        model.add(Conv1D(filters=512, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # LSTM — input (batch, 1, 512), output (batch, 200)
        # return_sequences=False (default) → already returns (batch, 200)
        model.add(LSTM(200))

        # FIX: removed model.add(Flatten()) — LSTM already outputs a 1-D vector
        # (batch, 200).  Flatten on a rank-2 tensor is a no-op in Keras but is
        # misleading and can mask shape errors.

        # Fully connected (paper Table 2, row 25)
        model.add(Dense(units=64, activation="relu",
                        kernel_regularizer=regularizers.L2(0.03)))
        model.add(Dropout(rate=0.4))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['acc'])

        if fold_num == 0:
            model.summary()

        history = model.fit(
            EEG_Train, Y_train,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=nb_epoch,
            class_weight=class_weight_dict,
            verbose=1
        )

        y_pred_proba = model.predict(EEG_Test)
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