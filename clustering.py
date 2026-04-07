import os
import numpy as np
import mne
from scipy.signal import welch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler


# ===============================
# 1. Load EDF files for a patient
# ===============================
def load_patient_data(patient_folder):
    edf_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.edf')])
    raws = []

    for file in edf_files:
        path = os.path.join(patient_folder, file)
        print(f"Loading {path}")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raws.append((file, raw))

    return raws


# ===============================
# 2. Extract features per channel
# ===============================
def bandpower(freqs, psd, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])


def extract_features(raw):
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    features = []

    for ch in data:
        # Basic stats
        mean = np.mean(ch)
        var = np.var(ch)

        # PSD
        freqs, psd = welch(ch, sfreq, nperseg=1024)

        # Frequency band powers
        delta = bandpower(freqs, psd, 0.5, 4)
        theta = bandpower(freqs, psd, 4, 8)
        alpha = bandpower(freqs, psd, 8, 13)
        beta = bandpower(freqs, psd, 13, 30)

        features.append([mean, var, delta, theta, alpha, beta])

    return np.array(features)


# ===============================
# 3. Cluster channels
# ===============================
def cluster_channels(features, n_clusters=3):
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Hierarchical clustering
    Z = linkage(features_scaled, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return labels


# ===============================
# 4. Process one patient
# ===============================
def process_patient(patient_folder, n_clusters=3):
    raws = load_patient_data(patient_folder)

    all_results = {}

    for file_name, raw in raws:
        print(f"Processing {file_name}")

        features = extract_features(raw)
        labels = cluster_channels(features, n_clusters)

        channel_clusters = dict(zip(raw.ch_names, labels))
        all_results[file_name] = channel_clusters

    return all_results


# ===============================
# 5. Main execution
# ===============================
if __name__ == "__main__":
    dataset_path = "../Raw_dataset"  # root folder of dataset
    patient_id = "chb01"     # e.g., chb01, chb02...

    patient_folder = os.path.join(dataset_path, patient_id)

    if not os.path.exists(patient_folder):
        raise ValueError(f"Patient folder not found: {patient_folder}")

    results = process_patient(patient_folder, n_clusters=3)

    # ===============================
    # 6. Print results
    # ===============================
    for file_name, clusters in results.items():
        print(f"\n=== {file_name} ===")
        for ch, cluster in clusters.items():
            print(f"{ch}: Cluster {cluster}")