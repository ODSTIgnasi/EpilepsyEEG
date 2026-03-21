import pywt
import numpy as np


def extract_coeffs_BONN_CHB(eeg_data, level=3):
    """
    Extract per-subband energy from all channels.
    18 channels x (level+1) subbands = 18 x 4 = 72 features per sample.
    """
    coeffs_list = []

    for i in range(eeg_data.shape[0]):
        channel_coeffs = []

        for ch in range(eeg_data.shape[1]):  # 18 channels
            signal = eeg_data[i, ch, :]

            # DWT decomposition into level+1 subbands
            coeffs = pywt.wavedec(signal, 'db1', level=level)

            # Take the ENERGY of each subband (1 value per subband)
            # rather than all raw coefficients (hundreds of values)
            subband_energies = [np.sum(c**2) for c in coeffs]  # 4 values per channel
            channel_coeffs.extend(subband_energies)

        coeffs_list.append(channel_coeffs)

    return np.array(coeffs_list)  # shape: (n_samples, 72)