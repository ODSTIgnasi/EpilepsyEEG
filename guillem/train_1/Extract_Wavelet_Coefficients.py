import pywt
import numpy as np


def extract_coeffs_BONN_CHB(eeg_data, level=3):
    n_samples  = eeg_data.shape[0]
    n_channels = eeg_data.shape[1]
    all_samples = []

    wavelet = 'db1'

    for i in range(n_samples):
        sample_features = []

        for ch in range(n_channels):
            signal = eeg_data[i, ch, :]   # (n_timepoints,)

            coeffs = pywt.wavedec(signal, wavelet, level=level)
            # coeffs = [A3, D3, D2, D1]

            if n_channels == 1:
                # ── BONN: concatenate raw coefficients ───────────────
                sample_features.append(np.concatenate(coeffs))
            else:
                # ── CHB-MIT: subband energy per channel ───────────────
                # 4 energies × 18 channels = 72 total scalars
                subband_energies = [np.sum(c ** 2) for c in coeffs]
                sample_features.extend(subband_energies)

        if n_channels == 1:
            all_samples.append(np.concatenate(sample_features))
        else:
            all_samples.append(np.array(sample_features, dtype=np.float32))

    return np.array(all_samples)