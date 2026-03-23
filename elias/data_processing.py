"""
Extract CHB-MIT EEG data from all pre-processed .npz and .parquet files.
Based on dataset documentation:
  - .npz contains EEG_win: shape (n_windows, 21, 128)
      - n_windows: number of 1-second windows per patient
      - 21 channels at 128 Hz
  - .parquet contains per-window metadata:
      - class:              0 = non-seizure, 1 = seizure
      - filename:           original .edf source file
      - filename_interval:  seizure ID within a file
      - global_interval:    seizure ID across all files for the patient

Outputs a random stratified train/val/test split ready for CNN training.
"""

import numpy as np
import pandas as pd
import mne
import os
import glob
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/hhome/ricse/Epilepsy"
OUT_DIR  = "/hhome/ricse01/DL/elias/Data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── CHB-MIT 21-channel montage (longitudinal bipolar, 10-20 system) ────────────
CHB_CHANNELS = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "T7-FT9", "FT9-FT10", "FT10-T8"
]

SFREQ = 128.0   # data has been downsampled to 128 Hz
SCALE = 1e-6    # uV to V for MNE

# ── Split ratios ───────────────────────────────────────────────────────────────
TEST_SIZE   = 0.1    # 10% test
VAL_SIZE    = 0.1    # 10% validation (of the remaining 90%)
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: load one patient's .npz + .parquet -> X (windows), y (labels), metadata
# ══════════════════════════════════════════════════════════════════════════════

def load_patient(eeg_path, meta_path, subject_id):

    # ── Load EEG windows ──────────────────────────────────────────────────────
    npz     = np.load(eeg_path, allow_pickle=True)
    eeg_win = npz["EEG_win"]          # shape: (n_windows, 21, 128)

    print(f"  EEG_win shape : {eeg_win.shape}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    metadata = pd.read_parquet(meta_path)

    # Sanity check: number of windows must match number of metadata rows
    assert len(metadata) == eeg_win.shape[0], (
        f"[{subject_id}] Mismatch: {eeg_win.shape[0]} windows vs "
        f"{len(metadata)} metadata rows"
    )

    y                = metadata["class"].values.astype(int)
    global_intervals = metadata["global_interval"].values

    print(f"  Total windows : {len(y)}")
    print(f"  Seizure       : {y.sum()}  ({100*y.mean():.1f}%)")
    print(f"  Non-seizure   : {(y==0).sum()}  ({100*(1-y.mean()):.1f}%)")
    print(f"  Unique seizure episodes (global): {len(np.unique(global_intervals[y==1]))}")

    return eeg_win, y, metadata


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: build MNE Epochs object from pre-segmented windows
# ══════════════════════════════════════════════════════════════════════════════

def build_epochs(eeg_win, y, subject_id):
    n_channels    = eeg_win.shape[1]
    channel_names = CHB_CHANNELS[:n_channels]

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=SFREQ,
        ch_types="eeg"
    )

    epochs          = mne.EpochsArray(eeg_win * SCALE, info, events=None, verbose=False)
    epochs.metadata = pd.DataFrame({"label": y})
    return epochs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP -- load all patients
# ══════════════════════════════════════════════════════════════════════════════

eeg_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_EEGwindow_*.npz")))

print(f"Found {len(eeg_files)} EEG files\n")
print("=" * 60)

all_X        = []
all_y        = []
all_metadata = []
all_epochs   = {}
failed       = []

for eeg_path in eeg_files:

    basename   = os.path.basename(eeg_path)
    subject_id = basename.split("_seizure_")[0]
    suffix     = basename.split("_EEGwindow_")[1].replace(".npz", "")
    meta_path  = os.path.join(DATA_DIR, f"{subject_id}_seizure_metadata_{suffix}.parquet")

    if not os.path.exists(meta_path):
        print(f"[{subject_id}] WARNING: metadata not found, skipping.")
        failed.append(subject_id)
        continue

    try:
        print(f"[{subject_id}] Loading...")
        eeg_win, y, metadata = load_patient(eeg_path, meta_path, subject_id)

        metadata["subject_id"] = subject_id
        epochs = build_epochs(eeg_win, y, subject_id)

        fif_path = os.path.join(OUT_DIR, f"{subject_id}_seizure_epo.fif")
        epochs.save(fif_path, overwrite=True, verbose=False)

        all_X.append(eeg_win)
        all_y.append(y)
        all_metadata.append(metadata)
        all_epochs[subject_id] = epochs

        print(f"  saved -> {fif_path}\n")

    except Exception as e:
        print(f"[{subject_id}] ERROR: {e}\n")
        failed.append(subject_id)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINE ALL PATIENTS
# ══════════════════════════════════════════════════════════════════════════════

X        = np.concatenate(all_X,        axis=0)   # (total_windows, 21, 128)
y        = np.concatenate(all_y,        axis=0)   # (total_windows,)
metadata = pd.concat(all_metadata, ignore_index=True)

print("=" * 60)
print("COMBINED DATASET")
print("=" * 60)
print(f"X shape         : {X.shape}   (windows x channels x time)")
print(f"Seizure windows : {y.sum()}  ({100*y.mean():.1f}%)")
print(f"Non-seizure     : {(y==0).sum()}  ({100*(1-y.mean()):.1f}%)")
print(f"Patients loaded : {len(all_epochs)}  |  Failed: {len(failed)}")
if failed:
    print(f"Failed: {failed}")


# ══════════════════════════════════════════════════════════════════════════════
# ADD CHANNEL DIMENSION FOR CNN
# (n_windows, 21, 128) -> (n_windows, 21, 128, 1)
# ══════════════════════════════════════════════════════════════════════════════

X = X[..., np.newaxis]
print(f"\nX shape after adding channel dim: {X.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM STRATIFIED TRAIN / VAL / TEST SPLIT
# stratify=y ensures seizure/non-seizure ratio is preserved in every split
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TRAIN / VAL / TEST SPLIT")
print("=" * 60)

# Step 1: split off the test set, keep indices to split metadata too
X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
    X, y, np.arange(len(y)),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# Step 2: split remaining data into train and val
# val_size_adjusted is relative to the train_val portion, not the full dataset
val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)

X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_train_val, y_train_val, idx_train_val,
    test_size=val_size_adjusted,
    random_state=RANDOM_SEED,
    stratify=y_train_val
)

# Split metadata to match each set
meta_train = metadata.iloc[idx_train].reset_index(drop=True)
meta_val   = metadata.iloc[idx_val].reset_index(drop=True)
meta_test  = metadata.iloc[idx_test].reset_index(drop=True)

print(f"Train : {X_train.shape[0]} windows  "
      f"| seizure: {y_train.sum()} ({100*y_train.mean():.1f}%)")
print(f"Val   : {X_val.shape[0]} windows  "
      f"| seizure: {y_val.sum()} ({100*y_val.mean():.1f}%)")
print(f"Test  : {X_test.shape[0]} windows  "
      f"| seizure: {y_test.sum()} ({100*y_test.mean():.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SPLITS
# ══════════════════════════════════════════════════════════════════════════════

np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), X=X_train, y=y_train)
np.savez_compressed(os.path.join(OUT_DIR, "val.npz"),   X=X_val,   y=y_val)
np.savez_compressed(os.path.join(OUT_DIR, "test.npz"),  X=X_test,  y=y_test)

meta_train.to_parquet(os.path.join(OUT_DIR, "meta_train.parquet"), index=False)
meta_val.to_parquet(  os.path.join(OUT_DIR, "meta_val.parquet"),   index=False)
meta_test.to_parquet( os.path.join(OUT_DIR, "meta_test.parquet"),  index=False)

print(f"\nSplits saved to: {OUT_DIR}")
print("  train.npz / val.npz / test.npz   <- X and y arrays, shape (n, 21, 128, 1)")
print("  meta_train/val/test.parquet      <- metadata per split")
print()
print("To load in your training script:")
print("  train = np.load('.../train.npz')")
print("  X_train, y_train = train['X'], train['y']")