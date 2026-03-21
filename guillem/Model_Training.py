import Extract_Wavelet_Coefficients as EX  # pyright: ignore[reportMissingImports]
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras import regularizers
import keras
from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, Conv1D, MaxPooling1D, Flatten
import pywt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
from itertools import cycle
import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, accuracy_score,
                             cohen_kappa_score, roc_curve, auc, roc_auc_score)
from sklearn.model_selection import KFold, train_test_split
import tensorflow


def plot_roc_per_fold(y_true, y_pred_proba, fold_num):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver operating characteristic', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.legend(loc="lower right", prop={'size': 20, 'weight': 'bold'})
    plt.savefig(f"plots/roc_binary_fold{fold_num}.png")
    plt.close()

# Code for Binary Classification of proposed model
def binary_model(data, label, dataset_choice):
    import os
    os.makedirs("plots", exist_ok=True)

    # Parameters
    batch_size = 60
    nb_epoch = 300          # 300 for full training
    num_k_fold_splits = 10  # 10 for full training

    # Ensure your 'data' and 'label' are loaded before this
    data = np.array(data)
    label = np.array(label)

    kf = KFold(n_splits=num_k_fold_splits, shuffle=True, random_state=2)

    # Metrics storage
    specificity_scores = []
    sensitivity_scores = []
    npv_scores = []
    ppv_scores = []
    f1_scores = []
    mcc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []
    Kappa_scores = []
    GDR_scores = []

    fold_num = 0
    history = None

    for train_index, test_index in kf.split(data):

        # clear Keras session between folds to avoid graph state accumulation
        tf.keras.backend.clear_session()

        X_tr_va, X_test = data[train_index], data[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]

        if dataset_choice == 'chbmit':
            # load_chbmit_data returns shape (N, 18, 1024, 1) but
            # extract_coeffs_BONN_CHB expects (N, 18, 1024)
            X_train = X_tr_va.squeeze(-1)
            X_test  = X_test.squeeze(-1)
            Y_train = y_tr_va
            Y_test  = y_test

            # Extract Wavelet coefficients from the EEG data (Test and Train)
            features_Train = EX.extract_coeffs_BONN_CHB(X_train, level=3)
            features_Test  = EX.extract_coeffs_BONN_CHB(X_test,  level=3)

            print(f"features_Train shape: {features_Train.shape}")

            # FIX: was hardcoded to 72 which only worked for the broken single-channel
            # extraction. Now uses the actual feature size from the corrected
            # multi-channel extractor (18 channels x n_coeffs_per_channel).
            n_features = features_Train.shape[1]

            # Expanding the dimensions to be prepared as an input to CNN-LSTM model
            EEG_Train = features_Train.reshape(features_Train.shape[0], n_features, 1)
            EEG_Test  = features_Test.reshape(features_Test.shape[0],   n_features, 1)

        # Define CNN-LSTM model
        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=2, input_shape=(EEG_Train.shape[1], 1), strides=2, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=32, kernel_size=2, strides=2, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(Conv1D(filters=64, kernel_size=2, strides=2, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(Conv1D(filters=128, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(Conv1D(filters=256, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(Conv1D(filters=512, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(LSTM(200))
        model.add(Flatten())
        model.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.L2(0.0003)))
        model.add(Dropout(rate=0.4))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

        # Train
        history = model.fit(
            EEG_Train, Y_train,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1
        )

        # predict raw probabilities first (needed for AUC), then threshold
        y_pred_proba = model.predict(EEG_Test)
        y_pred = (y_pred_proba > 0.5)

        # Confusion matrix and metrics
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        npv = tn / (tn + fn)
        ppv = tp / (tp + fp)
        f1  = f1_score(Y_test, y_pred)
        mcc = matthews_corrcoef(Y_test, y_pred)
        accuracy  = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall    = recall_score(Y_test, y_pred, average='weighted')
        kappa     = cohen_kappa_score(Y_test, y_pred)
        gdr       = np.sqrt(sensitivity * specificity)
        auc_val   = roc_auc_score(Y_test, y_pred_proba)

        # Store metrics
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

        # Plot ROC per fold using raw probabilities (not thresholded predictions)
        plot_roc_per_fold(Y_test, y_pred_proba, fold_num)

        print(f"\n--- Fold {fold_num + 1} Results ---")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  NPV:         {npv:.4f}")
        print(f"  PPV:         {ppv:.4f}")
        print(f"  F1:          {f1:.4f}")
        print(f"  MCC:         {mcc:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  Kappa:       {kappa:.4f}")
        print(f"  GDR:         {gdr:.4f}")
        print(f"  AUC:         {auc_val:.4f}")

        fold_num += 1

    # Print average +- std scores across all folds
    print("\n========== Final Results (Mean +- Std across folds) ==========")
    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} +- {np.std(accuracy_scores):.4f}")
    print(f"Precision:   {np.mean(precision_scores):.4f} +- {np.std(precision_scores):.4f}")
    print(f"Specificity: {np.mean(specificity_scores):.4f} +- {np.std(specificity_scores):.4f}")
    print(f"Sensitivity: {np.mean(sensitivity_scores):.4f} +- {np.std(sensitivity_scores):.4f}")
    print(f"NPV:         {np.mean(npv_scores):.4f} +- {np.std(npv_scores):.4f}")
    print(f"PPV:         {np.mean(ppv_scores):.4f} +- {np.std(ppv_scores):.4f}")
    print(f"F1:          {np.mean(f1_scores):.4f} +- {np.std(f1_scores):.4f}")
    print(f"MCC:         {np.mean(mcc_scores):.4f} +- {np.std(mcc_scores):.4f}")
    print(f"Recall:      {np.mean(recall_scores):.4f} +- {np.std(recall_scores):.4f}")
    print(f"Kappa:       {np.mean(Kappa_scores):.4f} +- {np.std(Kappa_scores):.4f}")
    print(f"GDR:         {np.mean(GDR_scores):.4f} +- {np.std(GDR_scores):.4f}")
    print(f"AUC:         {np.mean(auc_scores):.4f} +- {np.std(auc_scores):.4f}")

    return history