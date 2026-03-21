import Extract_Wavelet_Coefficients as EX # pyright: ignore[reportMissingImports]
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras import regularizers
import keras
import shap
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
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, cohen_kappa_score, roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
import tensorflow


def plot_roc_per_fold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2 # this is already defined here
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) #, linewidth=20) remove this duplicate argument
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') #, linewidth=20) remove this duplicate argument
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver operating characteristic', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    # Instead of fontdict, use prop keyword argument
    plt.legend(loc="lower right", prop={'size': 20, 'weight': 'bold'})
    plt.savefig("roc_binary.png")
    plt.close()

def plot_multiclass_roc(y_true, y_prob):
    """
    Plots the ROC curves for multi-class classification.

    Args:
        y_true: True labels (one-hot encoded).
        y_prob: Predicted probabilities for each class.
    """
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (area = {roc_auc["micro"]:.2f})', linestyle="--", linewidth=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') #, linewidth=20) remove this duplicate argument
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver operating characteristic', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    # Instead of fontdict, use prop keyword argument
    plt.legend(loc="lower right", prop={'size': 10, 'weight': 'bold'})
    plt.savefig("roc_multiclass.png")
    plt.close()

#Code for Binary Classification of proposed model
def binary_model(data,label,dataset_choice):
    # Parameters
    batch_size = 60
    nb_epoch = 300

    # Ensure your 'data' and 'label' are loaded before this
    data = np.array(data)
    label = np.array(label)

    kf = KFold(n_splits=10, shuffle=True, random_state=2)

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

    fold_num = 0  # Moved outside loop

    for train_index, test_index in kf.split(data):
        X_tr_va, X_test = data[train_index], data[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]

        if dataset_choice == 'chbmit':
            X_train = X_tr_va
            Y_train = y_tr_va
            Y_test = y_test

            # Extract Wavelet coefficients from the EEG data (Test and Train)
            features_Train = EX.extract_coeffs_BONN_CHB(X_train, level=3)
            features_Test = EX.extract_coeffs_BONN_CHB(X_test, level=3)

            #Expanding the dimensions to be prepared as an input to CNN-LSTM model
            EEG_Train = features_Train.reshape(features_Train.shape[0],72, 1)
            EEG_Test = features_Test.reshape(features_Test.shape[0], 72, 1)
            

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
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))  # same instead of valid
        model.add(Conv1D(filters=128, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))  # same instead of valid
        model.add(Conv1D(filters=256, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))  # same instead of valid
        model.add(Conv1D(filters=512, kernel_size=1, strides=1, padding="valid"))
        model.add(BatchNormalization(axis=2, momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))  # same instead of valid
        model.add(LSTM(200))
        model.add(Flatten())
        model.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.L2(0.0003)))
        model.add(Dropout(rate=0.4))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        Adam = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['acc'])


        # Train and predict
        history = model.fit(EEG_Train, Y_train, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        y_pred = model.predict(EEG_Test)
        y_pred = (y_pred > 0.5)

        # Confusion matrix and metrics
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        npv = tn / (tn + fn)
        ppv = tp / (tp + fp)
        f1 = f1_score(Y_test, y_pred)
        mcc = matthews_corrcoef(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)
        gdr = np.sqrt(sensitivity * specificity)

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

        # Plot ROC (optional)
        plot_roc_per_fold(Y_test, y_pred)

        fold_num += 1  # Increment fold number

    # Print average scores
    print("Accuracy scores for each fold: ", np.mean(accuracy_scores))
    print("Precision scores for each fold: ", np.mean(precision_scores))
    print("Specificity scores for each fold: ", np.mean(specificity_scores))
    print("Sensitivity scores for each fold: ", np.mean(sensitivity_scores))
    print("Negative Predictive Value scores for each fold: ", np.mean(npv_scores))
    print("Positive Predictive Value scores for each fold: ", np.mean(ppv_scores))
    print("F1 scores for each fold: ", np.mean(f1_scores))
    print("Matthews Correlation Coefficient scores for each fold: ", np.mean(mcc_scores))
    print("Recall scores for each fold: ", np.mean(recall_scores))
    print("Kappa scores for each fold: ", np.mean(Kappa_scores))
    print("GDR scores for each fold: ", np.mean(GDR_scores))

    return history
