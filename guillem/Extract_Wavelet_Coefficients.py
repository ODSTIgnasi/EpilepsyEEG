import pywt  # type: ignore
import numpy as np
import pywt


#Extracting approximation and Detailed Coefficients from signal

def extract_coeffs_BONN_CHB(eeg_data, level=3):
    coeffs_list = []

    for i in range(eeg_data.shape[0]):
        # Select the EEG signal from the ith trial
        signal = eeg_data[i, :, 0]

        # Apply DWT on the signal
        coeffs = pywt.wavedec(signal, 'db1', level=level)  # 'db1' refers to Daubechies wavelet

        # Flatten the coefficients and append to the list
        coeffs_flattened = np.concatenate(coeffs)
        coeffs_list.append(coeffs_flattened)

    return np.array(coeffs_list)
