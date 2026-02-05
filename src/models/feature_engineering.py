"""
Feature engineering utilities for extracting physiological features.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
try:
    from .config import (
        SAMPLING_RATE, TARGET_LENGTH,
        HR_MIN, HR_MAX, PAT_MIN, PAT_MAX,
        R_PEAK_HEIGHT_MULTIPLIER, MIN_R_PEAK_DISTANCE_MULTIPLIER
    )
except ImportError:
    from config import (
        SAMPLING_RATE, TARGET_LENGTH,
        HR_MIN, HR_MAX, PAT_MIN, PAT_MAX,
        R_PEAK_HEIGHT_MULTIPLIER, MIN_R_PEAK_DISTANCE_MULTIPLIER
    )


def extract_physiological_features(ecg_signals, ppg_signals, sampling_rate=SAMPLING_RATE):
    """Extract Pulse Arrival Time (PAT) and Heart Rate (HR) from ECG and PPG signals."""
    print("💓 Extracting physiological features (PAT, HR)...")

    all_pat_sequences = []
    all_hr_sequences = []
    all_peak_indices = {'r_peaks': [], 'ppg_feet': []}

    for i in range(ecg_signals.shape[0]):
        ecg = ecg_signals[i]
        ppg = ppg_signals[i]

        # --- R-peak detection for HR ---
        r_peaks, _ = find_peaks(
            ecg, 
            height=np.mean(ecg) + R_PEAK_HEIGHT_MULTIPLIER * np.std(ecg), 
            distance=sampling_rate * MIN_R_PEAK_DISTANCE_MULTIPLIER
        )
        all_peak_indices['r_peaks'].append(r_peaks)

        hr_sequence = np.full_like(ecg, fill_value=np.nan)
        if len(r_peaks) > 1:
            instant_hr = sampling_rate * 60.0 / np.diff(r_peaks)
            interp_func = interp1d(r_peaks[1:], instant_hr, kind='linear', bounds_error=False, fill_value='extrapolate')
            hr_sequence = interp_func(np.arange(len(ecg)))
            hr_sequence = np.clip(hr_sequence, HR_MIN, HR_MAX)
        all_hr_sequences.append(hr_sequence)

        # --- PPG foot detection for PAT ---
        ppg_feet, _ = find_peaks(-ppg, distance=sampling_rate * MIN_R_PEAK_DISTANCE_MULTIPLIER)
        all_peak_indices['ppg_feet'].append(ppg_feet)

        # --- PAT calculation ---
        pat_sequence = np.full_like(ppg, fill_value=np.nan)
        if len(r_peaks) > 0 and len(ppg_feet) > 0:
            pat_values = []
            corresponding_r_peaks = []
            for r_peak in r_peaks:
                following_feet = ppg_feet[ppg_feet > r_peak]
                if len(following_feet) > 0:
                    pat = (following_feet[0] - r_peak) / sampling_rate
                    if PAT_MIN <= pat <= PAT_MAX:
                        pat_values.append(pat)
                        corresponding_r_peaks.append(r_peak)
            if len(pat_values) > 1:
                interp_func = interp1d(
                    corresponding_r_peaks, pat_values, 
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                pat_sequence = interp_func(np.arange(len(ppg)))
                pat_sequence = np.clip(pat_sequence, PAT_MIN, PAT_MAX)
        all_pat_sequences.append(pat_sequence)

    pat_sequences = np.array(all_pat_sequences)
    hr_sequences = np.array(all_hr_sequences)

    # Fill remaining NaNs by forward/backward fill
    pat_sequences = pd.DataFrame(pat_sequences).ffill(axis=1).bfill(axis=1).to_numpy()
    hr_sequences = pd.DataFrame(hr_sequences).ffill(axis=1).bfill(axis=1).to_numpy()
    
    # Final safety check: If any NaN values remain (e.g., entire sequence was NaN),
    # replace with reasonable defaults
    if np.any(np.isnan(pat_sequences)):
        # For PAT, use median of valid values or middle of range
        valid_pat = pat_sequences[~np.isnan(pat_sequences)]
        if len(valid_pat) > 0:
            default_pat = np.median(valid_pat)
        else:
            default_pat = (PAT_MIN + PAT_MAX) / 2  # Middle of valid range
        pat_sequences = np.nan_to_num(pat_sequences, nan=default_pat)
        print(f"   - Warning: Replaced remaining PAT NaNs with {default_pat:.3f}s")
    
    if np.any(np.isnan(hr_sequences)):
        # For HR, use median of valid values or middle of range
        valid_hr = hr_sequences[~np.isnan(hr_sequences)]
        if len(valid_hr) > 0:
            default_hr = np.median(valid_hr)
        else:
            default_hr = (HR_MIN + HR_MAX) / 2  # Middle of valid range
        hr_sequences = np.nan_to_num(hr_sequences, nan=default_hr)
        print(f"   - Warning: Replaced remaining HR NaNs with {default_hr:.1f} bpm")

    print(f"   - PAT sequences created with shape: {pat_sequences.shape}")
    print(f"   - HR sequences created with shape: {hr_sequences.shape}")
    return pat_sequences, hr_sequences, all_peak_indices


def standardize_feature_length(feature_sequences, target_length=TARGET_LENGTH):
    """
    Standardizes the length of feature sequences by truncating or padding.
    """
    standardized_features = []
    for seq in feature_sequences:
        if len(seq) > target_length:
            start_idx = (len(seq) - target_length) // 2
            standardized_features.append(seq[start_idx:start_idx + target_length])
        else:
            padding = target_length - len(seq)
            pad_left = padding // 2
            pad_right = padding - pad_left
            standardized_features.append(np.pad(seq, (pad_left, pad_right), mode='edge'))
    return np.array(standardized_features)


def create_baseline_features(pat_sequences, hr_sequences, labels):
    """
    Creates a simple feature set for the baseline model.
    Features: mean PAT, std PAT, mean HR, std HR.
    """
    print("🛠️  Creating features for baseline model...")
    
    # Convert to numpy arrays if not already
    pat_sequences = np.asarray(pat_sequences)
    hr_sequences = np.asarray(hr_sequences)
    
    # Check for NaN/Inf values in inputs
    if np.any(np.isnan(pat_sequences)) or np.any(np.isinf(pat_sequences)):
        print("   - Warning: NaN or Inf found in PAT sequences. Filling with 0.")
        pat_sequences = np.nan_to_num(pat_sequences)
        
    if np.any(np.isnan(hr_sequences)) or np.any(np.isinf(hr_sequences)):
        print("   - Warning: NaN or Inf found in HR sequences. Filling with 0.")
        hr_sequences = np.nan_to_num(hr_sequences)

    mean_pat = np.mean(pat_sequences, axis=1)
    std_pat = np.std(pat_sequences, axis=1)
    mean_hr = np.mean(hr_sequences, axis=1)
    std_hr = np.std(hr_sequences, axis=1)
    
    X_baseline = np.column_stack([mean_pat, std_pat, mean_hr, std_hr])
    y_baseline = labels
    
    print(f"   - Baseline features created with shape: {X_baseline.shape}")
    return X_baseline, y_baseline


def create_4_channel_input(raw_signals, pat_sequences, hr_sequences):
    """
    Combines raw signals and physiological features into a 4-channel input.
    """
    print("🔄 Creating 4-channel input for the physiology-informed model...")

    pat_expanded = np.expand_dims(pat_sequences, axis=-1)
    hr_expanded = np.expand_dims(hr_sequences, axis=-1)

    X_phys_informed = np.concatenate([raw_signals, pat_expanded, hr_expanded], axis=-1)
    print(f"   - 4-channel input created with shape: {X_phys_informed.shape}")

    return X_phys_informed


def normalize_pat_subject_wise(pat_sequences, patient_ids, train_mask):
    """
    Normalize PAT per patient using mean and std computed from TRAIN data only.
    Apply the same normalization parameters for validation and test data.
    
    This prevents data leakage by ensuring normalization statistics are only computed
    from training data for each patient.
    
    Args:
        pat_sequences: Array of PAT sequences (n_samples, timesteps)
        patient_ids: Array of patient IDs for each sample
        train_mask: Boolean mask indicating training samples
        
    Returns:
        normalized_pat: Subject-wise normalized PAT sequences
        pat_stats: Dictionary {patient_id: {'mean': float, 'std': float}}
    """
    print("🔄 Normalizing PAT per patient using training data only...")
    
    normalized_pat = np.zeros_like(pat_sequences)
    pat_stats = {}
    
    # First pass: compute statistics from training data only
    train_patient_ids = patient_ids[train_mask]
    train_pat = pat_sequences[train_mask]
    unique_train_patients = np.unique(train_patient_ids)
    
    for patient_id in unique_train_patients:
        patient_train_mask = train_patient_ids == patient_id
        patient_train_pat = train_pat[patient_train_mask]
        
        # Compute mean and std across all time points and samples for this patient
        mean = np.mean(patient_train_pat)
        std = np.std(patient_train_pat)
        
        # Avoid division by zero
        if std < 1e-8:
            std = 1.0
        
        pat_stats[patient_id] = {'mean': mean, 'std': std}
    
    # Compute global statistics as fallback for patients not in training set
    global_mean = np.mean(train_pat)
    global_std = np.std(train_pat)
    if global_std < 1e-8:
        global_std = 1.0
    
    print(f"   - Computed PAT statistics for {len(pat_stats)} training patients")
    print(f"   - Global PAT stats: mean={global_mean:.4f}, std={global_std:.4f}")
    
    # Second pass: apply normalization to all samples
    for i, patient_id in enumerate(patient_ids):
        if patient_id in pat_stats:
            # Use patient-specific statistics
            mean = pat_stats[patient_id]['mean']
            std = pat_stats[patient_id]['std']
        else:
            # For validation/test patients not in training set, use global statistics
            mean = global_mean
            std = global_std
        
        normalized_pat[i] = (pat_sequences[i] - mean) / std
    
    print(f"   - PAT normalization complete")
    return normalized_pat, pat_stats
