import os
import numpy as np
import random
import shutil
import zipfile
import mne
import wfdb
import pandas as pd
import pyedflib
import time
import logging
from glob import glob
import pickle
import glob
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import gc
from scipy.signal import find_peaks
import re
import tqdm

#  CHB-MIT Dataset
def load_chbmit_data(dataset_path):
    ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3','P3-O1',
                 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                 'FZ-CZ', 'CZ-PZ']

    path2pt = dataset_path
    folders = sorted(glob.glob(path2pt+'/*/'))
    n_patient = [m[-2:] for m in [l.rsplit('/', 2)[-2] for l in folders]]
    
    print(*n_patient)
    random.seed(2023)

    ratio_train = 0.99
    train_patient_str = sorted(random.sample(n_patient, round(ratio_train*len(n_patient))))
    test_patient_str = sorted([l for l in n_patient if l not in train_patient_str])
    print('Train PT: ', *train_patient_str)
    print('Test PT: ', *test_patient_str)

    files_train = []
    for l in train_patient_str:
        files_train = files_train + glob.glob(path2pt+'/chb{}/*.edf'.format(l))

    files_test = []
    for l in test_patient_str:
        files_test = files_test + glob.glob(path2pt+'/chb{}/*.edf'.format(l))
    
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('read_files.log')
    logger.addHandler(fh)

    time_window = 8
    time_step = 4

    cache_signals = './signal_samples.npy'
    cache_labels  = './is_sz.npy'

    if os.path.exists(cache_signals) and os.path.exists(cache_labels):
        array_signals = np.load(cache_signals)
        array_is_sz   = np.load(cache_labels)
    else:
        p = 0.01
        counter = 0
        for temp_f in files_train:
            temp_edf = mne.io.read_raw_edf(temp_f)
            temp_labels = temp_edf.ch_names
            if sum([any([0 if re.match(c, l) == None else 1 for l in temp_edf.ch_names]) for c in ch_labels]) == len(ch_labels):
                time_window = 8
                time_step = 4
                fs = int(1 / (temp_edf.times[1] - temp_edf.times[0]))
                step_window = time_window * fs
                step = time_step * fs

                temp_is_sz = np.zeros((temp_edf.n_times,))
                if os.path.exists(temp_f + '.seizures'):
                    temp_annotation = wfdb.rdann(temp_f, 'seizures')
                    for i in range(int(temp_annotation.sample.size / 2)):
                        temp_is_sz[temp_annotation.sample[i * 2]:temp_annotation.sample[i * 2 + 1]] = 1
                temp_len = temp_edf.n_times

                temp_is_sz_ind = np.array(
                    [temp_is_sz[i * step:i * step + step_window].sum() / step_window for i in range((temp_len - step_window) // step)]
                )

                temp_0_sample_size = round(p * np.where(temp_is_sz_ind == 0)[0].size)
                temp_1_sample_size = np.where(temp_is_sz_ind > 0)[0].size

                counter = counter + temp_0_sample_size + temp_1_sample_size
            temp_edf.close()

        array_signals = np.zeros((counter, len(ch_labels), step_window), dtype=np.float32)
        array_is_sz = np.zeros(counter, dtype=bool)

        counter = 0
        for n, temp_f in enumerate(tqdm.tqdm(files_train)):
            to_log = 'No. {}: Reading. '.format(n)
            temp_edf = mne.io.read_raw_edf(temp_f)
            temp_labels = temp_edf.ch_names
            n_label_match = sum([any([0 if re.match(c, l) == None else 1 for l in temp_edf.ch_names]) for c in ch_labels])
            if n_label_match == len(ch_labels):
                ch_mapping = {sorted([l for l in temp_edf.ch_names if re.match(c, l) != None])[0]: c for c in ch_labels}
                temp_edf.rename_channels(ch_mapping)

                temp_is_sz = np.zeros((temp_edf.n_times,))
                temp_signals = temp_edf.get_data(picks=ch_labels) * 1e6

                if os.path.exists(temp_f + '.seizures'):
                    to_log = to_log + 'sz exists.'
                    temp_annotation = wfdb.rdann(temp_f, 'seizures')
                    for i in range(int(temp_annotation.sample.size / 2)):
                        temp_is_sz[temp_annotation.sample[i * 2]:temp_annotation.sample[i * 2 + 1]] = 1
                else:
                    to_log = to_log + 'No sz.'

                temp_len = temp_edf.n_times

                time_window = 8
                time_step = 4
                fs = int(1 / (temp_edf.times[1] - temp_edf.times[0]))
                step_window = time_window * fs
                step = time_step * fs

                temp_is_sz_ind = np.array(
                    [temp_is_sz[i * step:i * step + step_window].sum() / step_window for i in range((temp_len - step_window) // step)]
                )
                del temp_is_sz

                temp_0_sample_size = round(p * np.where(temp_is_sz_ind == 0)[0].size)
                temp_1_sample_size = np.where(temp_is_sz_ind > 0)[0].size

                # sz data
                temp_ind = list(np.where(temp_is_sz_ind > 0)[0])
                for i in temp_ind:
                    array_signals[counter, :, :] = temp_signals[:, i * step:i * step + step_window]
                    array_is_sz[counter] = True
                    counter = counter + 1

                # no sz data
                temp_ind = random.sample(list(np.where(temp_is_sz_ind == 0)[0]), temp_0_sample_size)
                for i in temp_ind:
                    array_signals[counter, :, :] = temp_signals[:, i * step:i * step + step_window]
                    array_is_sz[counter] = False
                    counter = counter + 1

                to_log += '{} signals added: {} w/o sz, {} w/ sz.'.format(
                    temp_0_sample_size + temp_1_sample_size, temp_0_sample_size, temp_1_sample_size
                )

            else:
                to_log += 'Not appropriate channel labels. Reading skipped.'.format(n)

            logger.info(to_log)
            temp_edf.close()

            if n % 10 == 0:
                gc.collect()
        gc.collect()

        np.save('signal_samples', array_signals)
        np.save('is_sz', array_is_sz)

    array_signals = array_signals[:, :, ::2]

    array_n = np.where(array_is_sz > .0)[0]
    print('Number of all the extracted signals: {}'.format(array_is_sz.size))
    print('Number of signals with seizures: {}'.format(array_n.size))
    print('Ratio of signals with seizures: {:.3f}'.format(array_n.size / array_is_sz.size))
    array_signals = array_signals[:, :, :, np.newaxis]

    from sklearn import model_selection
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        array_signals, array_is_sz, test_size=0.0,
        stratify=(array_is_sz > 0))

    del array_signals, array_is_sz
    return X_train, y_train
