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

Outputs an augmentation-aware train/val/test split ready for CNN training.

AUGMENTATION-AWARE SPLITTING
─────────────────────────────
Because the dataset is pre-augmented, windows near a seizure window carry
leaked information from their neighbours. A naive random split would let
augmented neighbours of a test window bleed into training.

Strategy (faithful to the provided pseudocode):

  all_windows
  train = all_windows
  test  = []

  for every seizure window (class == 1) in a test episode:
      left_samples  = min(5, position_within_episode)
      right_samples = min(5, episode_length - 1 - position_within_episode)

      pull left_samples  neighbours → test, remove from train
      pull the window itself        → test, remove from train
      pull right_samples neighbours → test, remove from train

  Non-seizure windows never explicitly iterated → stay in train unless
  pulled as a context neighbour of a seizure window.

Neighbour boundary: global_interval
  The ±5 walk is bounded by global_interval so it never crosses episode
  boundaries even if adjacent rows in the array belong to different episodes.

Episode selection for test:
  A random 20% of unique global_interval IDs are designated as test episodes.
  All remaining episodes (and their non-neighbour non-seizure windows) form
  the train pool, from which 10% is carved as a stratified val set.
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
CONTEXT_SIZE = 5       # neighbours pulled left and right of each seizure window

# ── Split ratios ───────────────────────────────────────────────────────────────
TEST_EPISODE_FRAC = 0.20   # fraction of unique global_interval IDs → test
VAL_SIZE          = 0.10   # fraction of remaining train windows → val
RANDOM_SEED       = 42


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: load one patient's .npz + .parquet → X (windows), y (labels), metadata
# ══════════════════════════════════════════════════════════════════════════════

def load_patient(eeg_path, meta_path, subject_id):

    npz     = np.load(eeg_path, allow_pickle=True)
    eeg_win = npz["EEG_win"]          # (n_windows, 21, 128)
    print(f"  EEG_win shape : {eeg_win.shape}")

    metadata = pd.read_parquet(meta_path)

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
# Faithfully implements the pseudocode:
#
#   all_windows
#   train = all_windows
#   test  = []
#   for window in all_windows:           ← only seizure windows in test episodes
#       left_samples  = 5
#       right_samples = 5
#       if window_index < 5:
#           left_samples = window_index  ← position within episode, not global
#       if window_index > (len - 5):
#           right_samples = len - window_index
#       for i in left_samples:
#           test.append / train.remove
#       for j in right_samples:
#           test.append / train.remove
#
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
# y                  : int array (N,)  — class labels (0 / 1)
# global_intervals   : array   (N,)  — global_interval per window
# test_episode_frac  : fraction of unique seizure episode IDs → test
# val_size           : fraction of remaining train windows → val
# context_size       : half-width of the neighbour context  (default 5)
# random_seed        : for reproducibility
#
# Returns
# ──────────────────────────────────────────────────────────────────────────────
# idx_train, idx_val, idx_test  — sorted index arrays into the N-window dataset
# ══════════════════════════════════════════════════════════════════════════════

def episode_aware_split(
    y,
    global_intervals,
    test_episode_frac=0.20,
    val_size=0.10,
    context_size=5,
    random_seed=42,
):
    N = len(y)

    # ── Step 1: randomly choose which seizure episodes go to test ─────────────
    seizure_mask     = y == 1
    seizure_episodes = np.unique(global_intervals[seizure_mask])
    n_test_episodes  = max(1, int(round(len(seizure_episodes) * test_episode_frac)))

    rng = np.random.default_rng(random_seed)
    test_episodes = set(
        rng.choice(seizure_episodes, size=n_test_episodes, replace=False).tolist()
    )

    print(f"\n  Total seizure episodes : {len(seizure_episodes)}")
    print(f"  Episodes → test        : {n_test_episodes}  {sorted(test_episodes)}")

    # ── Step 2: build lookup  global_interval → ordered list of global indices
    # Built over ALL windows in a test episode (both seizure and non-seizure),
    # because non-seizure windows adjacent to seizure windows can be neighbours.
    episode_to_indices: dict = {}
    for gidx in range(N):
        ep = global_intervals[gidx]
        if ep in test_episodes:
            episode_to_indices.setdefault(ep, []).append(gidx)
    for ep in episode_to_indices:
        episode_to_indices[ep].sort()

    # ── Step 3: pseudocode loop ───────────────────────────────────────────────
    # train starts as the full set; items are moved to test as we go.
    train_set = set(range(N))
    test_set  = set()

    for ep, ep_indices in episode_to_indices.items():
        ep_len = len(ep_indices)   # len(all_windows) in the pseudocode

        for pos, window_index in enumerate(ep_indices):
            # Only trigger context pull on seizure windows (class == 1)
            if y[window_index] != 1:
                continue

            # ── boundary checks from pseudocode ──────────────────────────────
            # "window_index" in the pseudocode refers to position within the
            # episode list (pos), not the global array index.
            left_samples  = context_size
            right_samples = context_size

            if pos < context_size:                    # window_index < 5
                left_samples = pos

            if pos > (ep_len - 1 - context_size):    # window_index > (len - 5)
                right_samples = ep_len - 1 - pos

            # left neighbours  (pseudocode: for i in left_samples)
            for i in range(1, left_samples + 1):
                nb = ep_indices[pos - i]
                test_set.add(nb)
                train_set.discard(nb)

            # the seizure window itself
            test_set.add(window_index)
            train_set.discard(window_index)

            # right neighbours (pseudocode: for j in right_samples)
            for j in range(1, right_samples + 1):
                nb = ep_indices[pos + j]
                test_set.add(nb)
                train_set.discard(nb)

    # ── Step 4: carve val from remaining train via stratified sampling ─────────
    train_arr = np.array(sorted(train_set))
    y_train   = y[train_arr]

    _, idx_val_local = train_test_split(
        np.arange(len(train_arr)),
        test_size=val_size,
        random_state=random_seed,
        stratify=y_train,
    )

    val_global = set(train_arr[idx_val_local].tolist())
    train_set  = train_set - val_global

    idx_train = np.array(sorted(train_set))
    idx_val   = np.array(sorted(val_global))
    idx_test  = np.array(sorted(test_set))

    return idx_train, idx_val, idx_test


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
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("AUGMENTATION-AWARE TRAIN / VAL / TEST SPLIT")
print(f"Context zone   : ±{CONTEXT_SIZE} windows per seizure window (global_interval-bounded)")
print(f"Test episodes  : {int(TEST_EPISODE_FRAC*100)}% of unique global_interval IDs (random)")
print(f"Val fraction   : {int(VAL_SIZE*100)}% of remaining train windows (stratified)")
print("=" * 60)

idx_train, idx_val, idx_test = episode_aware_split(
    y=y,
    global_intervals=metadata["global_interval"].values,
    test_episode_frac=TEST_EPISODE_FRAC,
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

total     = len(y)
accounted = len(idx_train) + len(idx_val) + len(idx_test)

print(f"\nTrain : {len(idx_train):>7} windows  "
      f"| seizure: {y_train.sum()} ({100*y_train.mean():.2f}%)")
print(f"Val   : {len(idx_val):>7} windows  "
      f"| seizure: {y_val.sum()} ({100*y_val.mean():.2f}%)")
print(f"Test  : {len(idx_test):>7} windows  "
      f"| seizure: {y_test.sum()} ({100*y_test.mean():.2f}%)")
print(f"Total : {accounted} / {total}  "
      f"({'OK — all windows accounted for' if accounted == total else 'WARNING: mismatch!'})")

# ── Disjointness + coverage sanity checks ─────────────────────────────────────
s_train, s_val, s_test = set(idx_train), set(idx_val), set(idx_test)
assert s_train.isdisjoint(s_val),  "FAIL: Train/Val overlap!"
assert s_train.isdisjoint(s_test), "FAIL: Train/Test overlap!"
assert s_val.isdisjoint(s_test),   "FAIL: Val/Test overlap!"
assert accounted == total,         "FAIL: Windows missing — splits don't cover the dataset!"
print("\n✓ All disjointness and coverage checks passed.")

# ── Episode leakage check ──────────────────────────────────────────────────────
test_eps  = set(meta_test.loc [meta_test ["class"] == 1, "global_interval"].unique())
train_eps = set(meta_train.loc[meta_train["class"] == 1, "global_interval"].unique())
val_eps   = set(meta_val.loc  [meta_val  ["class"] == 1, "global_interval"].unique())

print(f"\nSeizure episodes → test  : {sorted(test_eps)}")
print(f"Seizure episodes → train : {sorted(train_eps)}")
print(f"Seizure episodes → val   : {sorted(val_eps)}")
assert test_eps.isdisjoint(train_eps), \
    "FAIL: Same seizure episode appears in both train and test!"
print("✓ No seizure episode leakage between train and test.")


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
print("  train.npz / val.npz / test.npz      ← X and y arrays, shape (n, 21, 128, 1)")
print("  meta_train/val/test.parquet         ← metadata per split")
print()
print("To load in your training script:")
print("  train = np.load('.../train.npz')")
print("  X_train, y_train = train['X'], train['y']")