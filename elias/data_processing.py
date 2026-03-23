"""
Extract and inspect CHB-MIT EEG data from preprocessed .npz and .parquet files.
Dataset reference: https://physionet.org/content/chbmit/1.0.0/

Files expected in ./TestData/:
  - chb01_seizure_EEGwindow_1.npz      : EEG window array
  - chb01_seizure_metadata_1.parquet   : seizure metadata
"""

import numpy as np
import pandas as pd
import mne
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/elilw/university/UAB/Deep_Learning/Project2/TestData"
EEG_FILE   = os.path.join(DATA_DIR, "chb01_seizure_EEGwindow_1.npz")
META_FILE  = os.path.join(DATA_DIR, "chb01_seizure_metadata_1.parquet")

# ── CHB-MIT standard 23-channel montage (10-20 system) ────────────────────────
CHB_CHANNELS = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10","FT10-T8",
    "T8-P8-0"
]

SFREQ = 256.0  # CHB-MIT sampling rate (Hz)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load metadata
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("METADATA")
print("=" * 60)

metadata = pd.read_parquet(META_FILE)
print(metadata.to_string())
print()

# Pull seizure timing from metadata if available
seizure_onset  = metadata.get("seizure_onset_sec",  metadata.get("onset_sec",  None))
seizure_offset = metadata.get("seizure_offset_sec", metadata.get("offset_sec", None))
if seizure_onset is not None:
    seizure_onset  = float(seizure_onset.iloc[0])
    seizure_offset = float(seizure_offset.iloc[0])
    print(f"Seizure onset : {seizure_onset:.2f} s")
    print(f"Seizure offset: {seizure_offset:.2f} s")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load EEG array
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EEG NPZ FILE")
print("=" * 60)

npz = np.load(EEG_FILE, allow_pickle=True)
print(f"Keys in .npz : {list(npz.keys())}")

# Identify the EEG data array — try common key names
eeg_data = None
for key in ["eeg", "data", "X", "signals", "eeg_data", npz.files[0]]:
    if key in npz:
        eeg_data = npz[key]
        print(f"Using key    : '{key}'")
        break

if eeg_data is None:
    raise KeyError(f"Could not find EEG array. Available keys: {list(npz.keys())}")

# Ensure shape is (n_channels, n_times)
if eeg_data.ndim == 3:
    print(f"3-D array detected {eeg_data.shape}, using first epoch")
    eeg_data = eeg_data[0]

if eeg_data.shape[0] > eeg_data.shape[1]:
    eeg_data = eeg_data.T

n_channels, n_times = eeg_data.shape
duration_sec = n_times / SFREQ

print(f"Shape        : {n_channels} channels × {n_times} samples")
print(f"Duration     : {duration_sec:.2f} s  ({n_times} samples @ {SFREQ} Hz)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build MNE RawArray
# ══════════════════════════════════════════════════════════════════════════════

# Match channel list to actual number of channels in the data
channel_names = CHB_CHANNELS[:n_channels]
if "ch_names" in npz:
    channel_names = list(npz["ch_names"])[:n_channels]
elif "channel_names" in npz:
    channel_names = list(npz["channel_names"])[:n_channels]

while len(channel_names) < n_channels:
    channel_names.append(f"EEG{len(channel_names)+1:03d}")

info = mne.create_info(
    ch_names=channel_names,
    sfreq=SFREQ,
    ch_types="eeg"
)
info["description"] = "CHB-MIT Scalp EEG – reconstructed from .npz window"

# Scale assuming data is in µV → convert to Volts for MNE
SCALE = 1e-6

raw = mne.io.RawArray(eeg_data * SCALE, info, verbose=False)

print("=" * 60)
print("MNE RAW OBJECT")
print("=" * 60)
print(raw)
print()
print(f"Channel names : {raw.ch_names}")
print(f"Sampling freq : {raw.info['sfreq']} Hz")
print(f"Duration      : {raw.times[-1]:.2f} s")
print(f"n_channels    : {len(raw.ch_names)}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Add seizure annotation (if timing available)
# ══════════════════════════════════════════════════════════════════════════════
if seizure_onset is not None:
    duration = seizure_offset - seizure_onset
    annotations = mne.Annotations(
        onset=[seizure_onset],
        duration=[duration],
        description=["seizure"]
    )
    raw.set_annotations(annotations)
    print(f"Annotation added: seizure at {seizure_onset:.1f}s – {seizure_offset:.1f}s "
          f"(duration {duration:.1f}s)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Basic signal stats
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SIGNAL STATISTICS  (in µV)")
print("=" * 60)

data_uv, times = raw.get_data(return_times=True)
data_uv = data_uv / SCALE  # back to µV for readable stats

stats = pd.DataFrame({
    "channel": raw.ch_names,
    "mean_uV":   np.mean(data_uv, axis=1).round(3),
    "std_uV":    np.std(data_uv,  axis=1).round(3),
    "min_uV":    np.min(data_uv,  axis=1).round(3),
    "max_uV":    np.max(data_uv,  axis=1).round(3),
})
print(stats.to_string(index=False))
print()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Export summary
# ══════════════════════════════════════════════════════════════════════════════
out_csv = os.path.join(DATA_DIR, "chb01_seizure_stats_1.csv")
stats.to_csv(out_csv, index=False)
print(f"Stats saved to : {out_csv}")
print()
print("Done. `raw` object is ready for further MNE processing.")
print("Example next steps:")
print("  raw.plot()                         # interactive plot")
print("  raw.filter(1., 40.)                # bandpass filter")
print("  raw.compute_psd().plot()           # power spectral density")
raw.save("/home/elilw/university/UAB/Deep_Learning/Project2/TestData/chb01_seizure_raw_1.fif", overwrite=True)