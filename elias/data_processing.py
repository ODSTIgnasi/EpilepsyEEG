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

Outputs a context-aware stratified train/val/test split ready for CNN training.

AUGMENTATION-AWARE SPLITTING
─────────────────────────────
Because the dataset is already augmented, each window at index i carries
information from neighbouring windows. A naive random split would leak
augmented neighbours of a test window into the training set.

Strategy
  1. Pick val/test *seed* windows via stratified sampling (on class label).
  2. For every seed window, compute a contamination zone of ±CONTEXT_SIZE
     windows *within the same recording file* (filename column).
     Cross-file boundaries are respected — window 0 of file B is NOT
     a neighbour of the last window of file A.
  3. All indices inside any contamination zone are *excluded* from training;
     they are neither train, val, nor test.
  4. The seed windows themselves become the val/test sets.

This guarantees zero augmentation leakage across splits.
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

SFREQ        = 128.0   # data has been downsampled to 128 Hz
SCALE        = 1e-6    # uV → V for MNE
CONTEXT_SIZE = 5       # number of neighbouring windows to quarantine on each side

# ── Split ratios ───────────────────────────────────────────────────────────────
TEST_SIZE   = 0.2    # 20 % test  (of total windows)
VAL_SIZE    = 0.1    # 10 % val   (of total windows)
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: load one patient's .npz + .parquet → X (windows), y (labels), metadata
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
    print(f"  Unique seizure episodes (global): "
          f"{len(np.unique(global_intervals[y==1]))}")

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
# CORE: augmentation-aware split
#
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
# y              : integer label array, shape (N,)
# filenames      : string array, shape (N,) — the "filename" column from metadata.
#                  Windows from *different* files are never neighbours.
# test_size      : fraction of ALL windows to use as test seeds
# val_size       : fraction of ALL windows to use as val seeds
# context_size   : half-width of the quarantine zone (default 5)
# random_seed    : for reproducibility
#
# Returns
# ──────────────────────────────────────────────────────────────────────────────
# idx_train, idx_val, idx_test  — index arrays into the original N windows
# idx_quarantine                — windows excluded from every split
# ══════════════════════════════════════════════════════════════════════════════

def context_aware_split(
    y,
    filenames,
    test_size=0.2,
    val_size=0.1,
    context_size=5,
    random_seed=42,
):
    N      = len(y)
    all_idx = np.arange(N)

    # ── Step 1: stratified sample of test seeds ──────────────────────────────
    n_test = int(round(N * test_size))
    _, idx_test = train_test_split(
        all_idx,
        test_size=n_test,
        random_state=random_seed,
        stratify=y,
    )
    idx_test = set(idx_test.tolist())

    # ── Step 2: build per-file index ranges ──────────────────────────────────
    # Maps filename → sorted list of global indices belonging to that file.
    # Used to clip the context window at file boundaries.
    file_to_indices: dict[str, list[int]] = {}
    for i, fname in enumerate(filenames):
        file_to_indices.setdefault(fname, []).append(i)

    # Sort each file's index list (they should already be sorted, but be safe)
    for fname in file_to_indices:
        file_to_indices[fname].sort()

    # Reverse map: global index → position within its file
    idx_to_file_pos: dict[int, tuple[str, int]] = {}
    for fname, indices in file_to_indices.items():
        for pos, gidx in enumerate(indices):
            idx_to_file_pos[gidx] = (fname, pos)

    def get_context_zone(seed_idx: int) -> set[int]:
        """Return all global indices in the ±context_size zone of seed_idx,
        respecting file boundaries."""
        fname, pos = idx_to_file_pos[seed_idx]
        file_indices = file_to_indices[fname]
        lo = max(0,           pos - context_size)
        hi = min(len(file_indices) - 1, pos + context_size)
        return set(file_indices[lo : hi + 1])

    # ── Step 3: compute contamination zone for test seeds ────────────────────
    test_zone: set[int] = set()
    for seed in idx_test:
        test_zone |= get_context_zone(seed)

    # ── Step 4: stratified sample of val seeds from *non-test, non-zone* pool ─
    eligible_for_val = np.array([
        i for i in all_idx
        if i not in idx_test and i not in test_zone
    ])
    y_eligible = y[eligible_for_val]

    n_val = int(round(N * val_size))
    _, idx_val_local = train_test_split(
        np.arange(len(eligible_for_val)),
        test_size=n_val,
        random_state=random_seed,
        stratify=y_eligible,
    )
    idx_val = set(eligible_for_val[idx_val_local].tolist())

    # ── Step 5: compute contamination zone for val seeds ─────────────────────
    val_zone: set[int] = set()
    for seed in idx_val:
        val_zone |= get_context_zone(seed)

    # ── Step 6: quarantine = union of both zones minus the seeds themselves ──
    quarantine = (test_zone | val_zone) - idx_test - idx_val

    # ── Step 7: training set = everything not in test, val, or quarantine ────
    idx_train = set(all_idx.tolist()) - idx_test - idx_val - quarantine

    # Convert to sorted arrays
    idx_train     = np.array(sorted(idx_train))
    idx_val       = np.array(sorted(idx_val))
    idx_test      = np.array(sorted(idx_test))
    idx_quarantine = np.array(sorted(quarantine))

    return idx_train, idx_val, idx_test, idx_quarantine


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — load all patients
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
    meta_path  = os.path.join(
        DATA_DIR, f"{subject_id}_seizure_metadata_{suffix}.parquet"
    )

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

        print(f"  saved → {fif_path}\n")

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
print(f"X shape         : {X.shape}   (windows × channels × time)")
print(f"Seizure windows : {y.sum()}  ({100*y.mean():.1f}%)")
print(f"Non-seizure     : {(y==0).sum()}  ({100*(1-y.mean()):.1f}%)")
print(f"Patients loaded : {len(all_epochs)}  |  Failed: {len(failed)}")
if failed:
    print(f"Failed: {failed}")


# ══════════════════════════════════════════════════════════════════════════════
# ADD CHANNEL DIMENSION FOR CNN
# (n_windows, 21, 128) → (n_windows, 21, 128, 1)
# ══════════════════════════════════════════════════════════════════════════════

X = X[..., np.newaxis]
print(f"\nX shape after adding channel dim: {X.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION-AWARE TRAIN / VAL / TEST SPLIT
#
# The "filename" column is used as the file boundary marker so that the
# ±CONTEXT_SIZE quarantine zone never crosses recording boundaries.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("AUGMENTATION-AWARE TRAIN / VAL / TEST SPLIT")
print(f"Context zone    : ±{CONTEXT_SIZE} windows per seed (file-bounded)")
print("=" * 60)

idx_train, idx_val, idx_test, idx_quarantine = context_aware_split(
    y=y,
    filenames=metadata["filename"].values,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    context_size=CONTEXT_SIZE,
    random_seed=RANDOM_SEED,
)

X_train, y_train = X[idx_train], y[idx_train]
X_val,   y_val   = X[idx_val],   y[idx_val]
X_test,  y_test  = X[idx_test],  y[idx_test]

meta_train = metadata.iloc[idx_train].reset_index(drop=True)
meta_val   = metadata.iloc[idx_val].reset_index(drop=True)
meta_test  = metadata.iloc[idx_test].reset_index(drop=True)

total      = len(y)
quarantine_pct = 100 * len(idx_quarantine) / total

print(f"Train      : {len(idx_train):>7} windows  "
      f"| seizure: {y_train.sum()} ({100*y_train.mean():.1f}%)")
print(f"Val        : {len(idx_val):>7} windows  "
      f"| seizure: {y_val.sum()} ({100*y_val.mean():.1f}%)")
print(f"Test       : {len(idx_test):>7} windows  "
      f"| seizure: {y_test.sum()} ({100*y_test.mean():.1f}%)")
print(f"Quarantine : {len(idx_quarantine):>7} windows  "
      f"({quarantine_pct:.1f}% of total — excluded from all splits)")
print(f"Total used : {len(idx_train)+len(idx_val)+len(idx_test)+len(idx_quarantine)} "
      f"/ {total}")


# ── Sanity checks ─────────────────────────────────────────────────────────────
sets = [set(idx_train), set(idx_val), set(idx_test), set(idx_quarantine)]
assert sets[0].isdisjoint(sets[1]), "Train/Val overlap detected!"
assert sets[0].isdisjoint(sets[2]), "Train/Test overlap detected!"
assert sets[0].isdisjoint(sets[3]), "Train/Quarantine overlap detected!"
assert sets[1].isdisjoint(sets[2]), "Val/Test overlap detected!"
print("\n✓ All split disjointness checks passed.")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SPLITS
# ══════════════════════════════════════════════════════════════════════════════

np.savez_compressed(os.path.join(OUT_DIR, "train.npz"), X=X_train, y=y_train)
np.savez_compressed(os.path.join(OUT_DIR, "val.npz"),   X=X_val,   y=y_val)
np.savez_compressed(os.path.join(OUT_DIR, "test.npz"),  X=X_test,  y=y_test)

meta_train.to_parquet(os.path.join(OUT_DIR, "meta_train.parquet"), index=False)
meta_val.to_parquet(  os.path.join(OUT_DIR, "meta_val.parquet"),   index=False)
meta_test.to_parquet( os.path.join(OUT_DIR, "meta_test.parquet"),  index=False)

# Save quarantine indices for auditing
np.save(os.path.join(OUT_DIR, "idx_quarantine.npy"), idx_quarantine)

print(f"\nSplits saved to: {OUT_DIR}")
print("  train.npz / val.npz / test.npz      ← X and y arrays, shape (n, 21, 128, 1)")
print("  meta_train/val/test.parquet         ← metadata per split")
print("  idx_quarantine.npy                  ← global indices of quarantined windows")
print()
print("To load in your training script:")
print("  train = np.load('.../train.npz')")
print("  X_train, y_train = train['X'], train['y']")