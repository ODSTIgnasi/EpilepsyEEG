#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CHB-MIT EEG Seizure Detection Demo
Converted from Jupyter notebook to standalone Python script.
"""

import os
import glob
import random
import gc
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import mne
import tqdm
from scipy.signal import find_peaks
from sklearn import model_selection, metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import visualkeras

# Create plots directory
os.makedirs("plots", exist_ok=True)

# Set MNE log level
mne.set_log_level(verbose='ERROR')

# Logging
logger = logging.getLogger(__name__)
fh = logging.FileHandler('read_files.log')
logger.addHandler(fh)

# Bipolar channel labels
ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3','P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ']

# Dataset path
path2pt = '/home/gcadevall/CVC/DL/dataset'

# Get patient folders
folders = sorted(glob.glob(path2pt+'/*/'))
n_patient = [m[-2:] for m in [l.rsplit('/', 2)[-2] for l in folders]]
print("Patient IDs:", n_patient)

# Split patients into train/test
random.seed(2023)
ratio_train = 0.8
train_patient_str = sorted(random.sample(n_patient, round(ratio_train*len(n_patient))))
test_patient_str = sorted([l for l in n_patient if l not in train_patient_str])
print('Train Patients:', train_patient_str)
print('Test Patients:', test_patient_str)

# File lists
files_train = []
for l in train_patient_str:
    files_train += glob.glob(path2pt+'/chb{}/*.edf'.format(l))

files_test = []
for l in test_patient_str:
    files_test += glob.glob(path2pt+'/chb{}/*.edf'.format(l))

print("Training files:", len(files_train))
print("Test files:", len(files_test))

# Signal extraction parameters
time_window = 4
time_step = 1

# Load preprocessed data if available
if os.path.exists('signal_samples.npy') and os.path.exists('is_sz.npy'):
    array_signals = np.load('signal_samples.npy')
    array_is_sz = np.load('is_sz.npy')
    print("Loaded preprocessed signals:", array_signals.shape)
else:
    p = 0.01
    counter = 0
    for temp_f in files_train:
        temp_edf = mne.io.read_raw_edf(temp_f)
        if sum([any([0 if re.match(c, l)==None else 1 for l in temp_edf.ch_names]) for c in ch_labels])==len(ch_labels):
            fs = int(1/(temp_edf.times[1]-temp_edf.times[0]))
            step_window = time_window*fs
            step = time_step*fs

            temp_is_sz = np.zeros((temp_edf.n_times,))
            if os.path.exists(temp_f+'.seizures'):
                temp_annotation = wfdb.rdann(temp_f, 'seizures')
                for i in range(int(temp_annotation.sample.size/2)):
                    temp_is_sz[temp_annotation.sample[i*2]:temp_annotation.sample[i*2+1]]=1

            temp_len = temp_edf.n_times
            temp_is_sz_ind = np.array(
                [temp_is_sz[i*step:i*step+step_window].sum()/step_window for i in range((temp_len-step_window)//step)]
            )

            temp_0_sample_size = round(p*np.where(temp_is_sz_ind==0)[0].size)
            temp_1_sample_size = np.where(temp_is_sz_ind>0)[0].size

            counter += temp_0_sample_size + temp_1_sample_size

        temp_edf.close()

    array_signals = np.zeros((counter, len(ch_labels), step_window), dtype=np.float32)
    array_is_sz = np.zeros(counter, dtype=bool)

    counter = 0
    for n, temp_f in enumerate(files_train):
        temp_edf = mne.io.read_raw_edf(temp_f)
        if sum([any([0 if re.match(c, l)==None else 1 for l in temp_edf.ch_names]) for c in ch_labels])==len(ch_labels):
            ch_mapping = {sorted([l for l in temp_edf.ch_names if re.match(c, l)!=None])[0]:c for c in ch_labels}
            temp_edf.rename_channels(ch_mapping)

            temp_is_sz = np.zeros((temp_edf.n_times,))
            temp_signals = temp_edf.get_data(picks=ch_labels)*1e6

            if os.path.exists(temp_f+'.seizures'):
                temp_annotation = wfdb.rdann(temp_f, 'seizures')
                for i in range(int(temp_annotation.sample.size/2)):
                    temp_is_sz[temp_annotation.sample[i*2]:temp_annotation.sample[i*2+1]]=1

            fs = int(1/(temp_edf.times[1]-temp_edf.times[0]))
            step_window = time_window*fs
            step = time_step*fs
            temp_is_sz_ind = np.array(
                [temp_is_sz[i*step:i*step+step_window].sum()/step_window for i in range((temp_edf.n_times-step_window)//step)]
            )

            temp_0_sample_size = round(p*np.where(temp_is_sz_ind==0)[0].size)
            temp_1_sample_size = np.where(temp_is_sz_ind>0)[0].size

            # sz samples
            for i in np.where(temp_is_sz_ind>0)[0]:
                array_signals[counter, :, :] = temp_signals[:, i*step:i*step+step_window]
                array_is_sz[counter] = True
                counter += 1

            # no sz samples
            for i in random.sample(list(np.where(temp_is_sz_ind==0)[0]), temp_0_sample_size):
                array_signals[counter, :, :] = temp_signals[:, i*step:i*step+step_window]
                array_is_sz[counter] = False
                counter += 1

        temp_edf.close()
        gc.collect()

    np.save('signal_samples.npy', array_signals)
    np.save('is_sz.npy', array_is_sz)
    print("Extracted signals shape:", array_signals.shape)

# Preprocessing: downsample 256 Hz -> 128 Hz
array_signals = array_signals[:, :, ::2]
print("Resampled signals shape:", array_signals.shape)
# Normalize per window
def normalize(x):
    return (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-6)

array_signals = normalize(array_signals)

# Plot a sample signal
vertical_width = 250
signals = array_signals[-1, :, :]
fs = 128
fig, ax = plt.subplots()
for i in range(signals.shape[0]):
    ax.plot(np.arange(signals.shape[-1])/fs, signals[i, :]+i*vertical_width, linewidth=0.5, color='tab:blue')
    ax.annotate(ch_labels[i], xy=(0, i*vertical_width))
ax.invert_yaxis()
plt.savefig("plots/sample_signal.png")
plt.close()

# Show seizure ratio
array_n = np.where(array_is_sz>0)[0]
print('Number of signals:', array_is_sz.size)
print('Number of seizure signals:', array_n.size)
print('Ratio of seizure signals: {:.3f}'.format(array_n.size/array_is_sz.size))

# Reshape for CNN
array_signals = array_signals[:, :, :, np.newaxis]
print("Signals reshaped for CNN:", array_signals.shape)

# Split training/validation
X_train, X_val, y_train, y_val = model_selection.train_test_split(
    array_signals, array_is_sz, test_size=0.3, stratify=(array_is_sz>0)
)
del array_signals, array_is_sz
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)

# Build CNN model
model = keras.models.Sequential()
model.add(layers.Conv2D(filters=64, kernel_size=(2, 4), padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(layers.Conv2D(filters=64, kernel_size=(2, 4), strides=(1, 2), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((1, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(2, 4), padding='same', activation='relu'))
model.add(layers.Conv2D(filters=128, kernel_size=(2, 4), strides=(1, 2), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(4, 4), padding='same', activation='relu'))
model.add(layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 2), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((1, 2)))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
visualkeras.layered_view(model, scale_xy=0.5, legend=True).save("plots/model_architecture.png")

# Compile model
LEARNING_RATE = 1e-4
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
callbacks = [es]

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: class_weights[0], 1: class_weights[1]}
print(class_weights)

# Train
hist = model.fit(
    x=X_train, y=y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=256,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save model
model.save('CHB_MIT_sz_detec_demo.h5')

# Plot training history
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist.history['loss'], label='loss')
ax[0].plot(hist.history['val_loss'], label='val_loss')
ax[0].axvline(x=es.best_epoch, color='tab:red', alpha=0.5, label='early stopping')
ax[0].set_title('Loss')
ax[0].legend()

ax[1].plot(hist.history['accuracy'], label='accuracy')
ax[1].plot(hist.history['val_accuracy'], label='val_accuracy')
ax[1].axvline(x=es.best_epoch, color='tab:red', alpha=0.5, label='early stopping')
ax[1].set_title('Accuracy')
ax[1].legend()
plt.savefig("plots/training_history.png")
plt.close()


# Patient IDs: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
# Train Patients: ['02', '03', '04', '05', '06', '09', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24']
# Test Patients: ['01', '07', '08', '10', '22']
# Training files: 549
# Test files: 137
