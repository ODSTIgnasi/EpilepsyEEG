import pywt
import numpy as np


def extract_coeffs_BONN_CHB(eeg_data, level=3):
    """
    Extract DWT features from EEG data.

    BONN (single-channel, n_channels == 1):
        - Concatenates ALL raw coefficients from the single channel
        - Produces ~4097 features per sample  (paper Fig. 4b: "4100×1 for Bonn")

    CHB-MIT (multi-channel, n_channels == 18):
        - Takes the ENERGY (sum of squares) of each subband per channel
        - 18 channels × (level+1=4) subbands = 72 scalar features per sample
        - Matches paper Fig. 4b: "72×1 for MIT-CHB Dataset"

    KEY FIX vs. original code
    ─────────────────────────
    1. Wavelet changed from 'db1' (Haar) to 'db3'.
       Paper (p.4): "Daubechies-3 (DB-3) wavelet function is considered
       the mother wavelet."  In pywt the name is 'db3', not 'db1'.

    2. For CHB-MIT the code now keeps per-channel ENERGY scalars (the
       original intent), but collects them correctly so that the final
       array is shape (n_samples, 72) rather than (n_samples, 18, N).
       The previous bug was that np.array(channel_features) was called
       on a list of 4-element lists, yielding (18, 4) instead of (72,).

    Args
    ────
    eeg_data : np.ndarray
        For CHB-MIT (after squeeze): shape (n_samples, 18, 512)
        For BONN:                    shape (n_samples,  1, 4097)
    level    : int
        DWT decomposition level. Paper uses level=3.

    Returns
    ───────
    np.ndarray
        CHB-MIT : (n_samples, 72)    — 18 ch × 4 subband energies
        BONN    : (n_samples, ~4100) — raw concatenated coefficients
    """
    n_samples  = eeg_data.shape[0]
    n_channels = eeg_data.shape[1]
    all_samples = []

    for i in range(n_samples):
        # ── per-sample feature accumulator ──────────────────────────
        # For CHB-MIT: flat list of scalar energies (will become (72,))
        # For BONN:    list of coefficient arrays (concatenated at end)
        sample_features = []

        for ch in range(n_channels):
            signal = eeg_data[i, ch, :]          # (n_timepoints,)

            # FIX 1: 'db3' is the pywt name for Daubechies-3
            # Paper: "Daubechies-3 (DB-3) wavelet function"
            coeffs = pywt.wavedec(signal, 'db3', level=level)
            # coeffs = [A3, D3, D2, D1]  (level=3 → 4 arrays)

            if n_channels == 1:
                # ── BONN: concatenate raw coefficients ───────────────
                # All coefficient arrays flattened into one 1-D vector
                sample_features.append(np.concatenate(coeffs))
            else:
                # ── CHB-MIT: subband energy per channel ───────────────
                # FIX 2: extend with SCALARS, not a sub-list.
                # Original bug: channel_features.extend(subband_energies)
                # put 4 scalars per channel → after np.array() the shape
                # was (18, 4) instead of (72,).
                # Correct: extend the flat accumulator with the 4 scalars
                # so that after all 18 channels we have a (72,) vector.
                subband_energies = [np.sum(c ** 2) for c in coeffs]
                sample_features.extend(subband_energies)   # 4 scalars added

        if n_channels == 1:
            # BONN: concatenate the single channel's coefficient arrays
            all_samples.append(np.concatenate(sample_features))
        else:
            # CHB-MIT: sample_features is already a flat list of 72 scalars
            all_samples.append(np.array(sample_features))  # (72,)

    result = np.array(all_samples)
    # CHB-MIT → (n_samples, 72)
    # BONN    → (n_samples, ~4100)
    return result