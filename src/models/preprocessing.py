"""
Signal preprocessing and data splitting utilities.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
try:
    from .config import (
        TARGET_LENGTH, SAMPLING_RATE,
        PPG_LOW_CUT, PPG_HIGH_CUT, PPG_FILTER_ORDER,
        ECG_LOW_CUT, ECG_HIGH_CUT, ECG_FILTER_ORDER,
        TEST_SIZE, VAL_SIZE
    )
except ImportError:
    from config import (
        TARGET_LENGTH, SAMPLING_RATE,
        PPG_LOW_CUT, PPG_HIGH_CUT, PPG_FILTER_ORDER,
        ECG_LOW_CUT, ECG_HIGH_CUT, ECG_FILTER_ORDER,
        TEST_SIZE, VAL_SIZE
    )


def preprocess_signals(signals, target_length=TARGET_LENGTH, sampling_rate=SAMPLING_RATE):
    """Preprocess physiological signals for model input."""
    print("🔄 Preprocessing signals...")

    processed_signals_all = []
    for signal_window in signals:
        # Assuming channel 0 is PPG, channel 1 is ECG
        ppg_signal = signal_window[0, :]
        ecg_signal = signal_window[1, :]

        # PPG bandpass filtering (0.5-8 Hz)
        nyquist_ppg = sampling_rate / 2
        low_cut_ppg = PPG_LOW_CUT / nyquist_ppg
        high_cut_ppg = PPG_HIGH_CUT / nyquist_ppg
        b_ppg, a_ppg = butter(PPG_FILTER_ORDER, [low_cut_ppg, high_cut_ppg], btype='band')
        filtered_ppg = filtfilt(b_ppg, a_ppg, ppg_signal)

        # ECG bandpass filtering (0.5-40 Hz)
        nyquist_ecg = sampling_rate / 2
        low_cut_ecg = ECG_LOW_CUT / nyquist_ecg
        high_cut_ecg = ECG_HIGH_CUT / nyquist_ecg
        b_ecg, a_ecg = butter(ECG_FILTER_ORDER, [low_cut_ecg, high_cut_ecg], btype='band')
        filtered_ecg = filtfilt(b_ecg, a_ecg, ecg_signal)

        # Standardize length by truncating or padding
        def standardize_length(signal, length):
            if len(signal) > length:
                start_idx = (len(signal) - length) // 2
                return signal[start_idx:start_idx + length]
            padding = length - len(signal)
            pad_left = padding // 2
            pad_right = padding - pad_left
            return np.pad(signal, (pad_left, pad_right), mode='constant')

        processed_ppg = standardize_length(filtered_ppg, target_length)
        processed_ecg = standardize_length(filtered_ecg, target_length)

        processed_signals_all.append(np.stack([processed_ppg, processed_ecg], axis=0))

    processed_signals = np.array(processed_signals_all)

    # Normalization (Z-score per channel)
    mean = np.mean(processed_signals, axis=2, keepdims=True)
    std = np.std(processed_signals, axis=2, keepdims=True)
    processed_signals = (processed_signals - mean) / (std + 1e-8)

    # Transpose to (samples, timesteps, channels) for Keras
    processed_signals = np.transpose(processed_signals, (0, 2, 1))

    print(f"   - Signal preprocessing complete: {processed_signals.shape}")
    print(f"   - Target length: {target_length} samples ({target_length/sampling_rate:.1f} seconds)")
    return processed_signals


def create_subject_wise_splits(patient_ids, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """Creates train/validation/test splits that are subject-wise and returns boolean masks."""
    print("📊 Creating subject-wise data splits...")

    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Split patient IDs into train, val, test
    train_val_pids, test_pids = train_test_split(unique_patients, test_size=test_size, random_state=42)

    # Adjust val_size relative to the remaining pool of patients
    relative_val_size = val_size / (1 - test_size)
    train_pids, val_pids = train_test_split(train_val_pids, test_size=relative_val_size, random_state=42)

    print(f"   - Total unique patients: {n_patients}")
    print(f"   - Train patients: {len(train_pids)}")
    print(f"   - Validation patients: {len(val_pids)}")
    print(f"   - Test patients: {len(test_pids)}")

    train_mask = np.isin(patient_ids, train_pids)
    val_mask = np.isin(patient_ids, val_pids)
    test_mask = np.isin(patient_ids, test_pids)

    print(f"\n   - Train samples: {np.sum(train_mask)} ({np.sum(train_mask)/len(patient_ids)*100:.1f}%)")
    print(f"   - Validation samples: {np.sum(val_mask)} ({np.sum(val_mask)/len(patient_ids)*100:.1f}%)")
    print(f"   - Test samples: {np.sum(test_mask)} ({np.sum(test_mask)/len(patient_ids)*100:.1f}%)")
    return train_mask, val_mask, test_mask
