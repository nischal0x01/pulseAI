"""
Feature engineering utilities for extracting physiological features with improved 
stability for Blood Pressure estimation.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
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


def bandpass_filter(data, lowcut=0.5, highcut=8.0, fs=SAMPLING_RATE, order=4):
    """Apply a butterworth bandpass filter to clean PPG/ECG signals."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Validate frequency bands
    if low <= 0 or high >= 1:
        print(f"⚠️  Invalid filter bands: low={low}, high={high}. Skipping filter.")
        return data
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, padlen=min(len(data)-1, 3*max(len(b), len(a))))


def extract_physiological_features(ecg_signals, ppg_signals, sampling_rate=SAMPLING_RATE):
    """
    Extract PAT and HR using improved PPG foot detection with APG (2nd derivative).
    Includes signal quality check to handle noisy windows.
    
    Returns:
        pat_sequences: (n_samples, timesteps) array of PAT values
        hr_sequences: (n_samples, timesteps) array of HR values
        quality_mask: (n_samples,) boolean array indicating valid windows
    """
    print("💓 Extracting robust physiological features (PAT, HR)...")

    all_pat_sequences = []
    all_hr_sequences = []
    quality_mask = []  # Track which windows have good signal quality

    for i in range(ecg_signals.shape[0]):
        ecg = ecg_signals[i]
        
        # Apply bandpass filter to PPG for cleaner foot detection
        try:
            ppg = bandpass_filter(ppg_signals[i], lowcut=0.5, highcut=8.0, fs=sampling_rate)
        except Exception as e:
            print(f"   ⚠️  Sample {i}: Filter failed ({e}), using raw PPG")
            ppg = ppg_signals[i]

        # --- 1. R-peak detection (ECG) ---
        r_peaks, _ = find_peaks(
            ecg, 
            height=np.mean(ecg) + R_PEAK_HEIGHT_MULTIPLIER * np.std(ecg), 
            distance=int(sampling_rate * MIN_R_PEAK_DISTANCE_MULTIPLIER)
        )

        # --- 2. PPG Foot Detection using 2nd derivative (APG) ---
        # The 2nd derivative helps identify the foot of the PPG more reliably
        ppg_diff2 = np.diff(ppg, n=2)
        # Pad to maintain original length
        ppg_diff2 = np.pad(ppg_diff2, (1, 1), mode='edge')
        
        # Find peaks in 2nd derivative (corresponds to rapid upstroke)
        ppg_feet, _ = find_peaks(
            ppg_diff2, 
            distance=int(sampling_rate * MIN_R_PEAK_DISTANCE_MULTIPLIER)
        )
        
        # Signal Quality Check: Need at least 2 cardiac cycles
        if len(r_peaks) < 2 or len(ppg_feet) < 2:
            quality_mask.append(False)
            # Fill with reasonable defaults
            all_hr_sequences.append(np.full(len(ecg), (HR_MIN + HR_MAX) / 2))
            all_pat_sequences.append(np.full(len(ecg), (PAT_MIN + PAT_MAX) / 2))
            continue
        
        quality_mask.append(True)

        # --- 3. HR Calculation with safe interpolation ---
        instant_hr = sampling_rate * 60.0 / np.diff(r_peaks)
        instant_hr = np.clip(instant_hr, HR_MIN, HR_MAX)  # Clip before interpolation
        
        if len(r_peaks) > 1:
            # Use constant fill values instead of extrapolation
            interp_hr = interp1d(
                r_peaks[1:], instant_hr, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(instant_hr[0], instant_hr[-1])
            )
            hr_sequence = interp_hr(np.arange(len(ecg)))
            hr_sequence = np.clip(hr_sequence, HR_MIN, HR_MAX)
        else:
            hr_sequence = np.full(len(ecg), (HR_MIN + HR_MAX) / 2)
        
        all_hr_sequences.append(hr_sequence)

        # --- 4. PAT Calculation (R-peak to PPG foot) ---
        pat_values = []
        corresponding_times = []
        
        for r_peak in r_peaks:
            # Find the first PPG foot occurring after the R-peak
            following_feet = ppg_feet[ppg_feet > r_peak]
            if len(following_feet) > 0:
                pat = (following_feet[0] - r_peak) / sampling_rate
                if PAT_MIN <= pat <= PAT_MAX:
                    pat_values.append(pat)
                    corresponding_times.append(r_peak)

        if len(pat_values) > 1:
            # Safe interpolation with constant boundary values
            interp_pat = interp1d(
                corresponding_times, pat_values, 
                kind='linear', 
                bounds_error=False, 
                fill_value=(pat_values[0], pat_values[-1])
            )
            pat_sequence = interp_pat(np.arange(len(ppg)))
            pat_sequence = np.clip(pat_sequence, PAT_MIN, PAT_MAX)
        else:
            # Not enough valid PAT measurements
            pat_sequence = np.full(len(ppg), (PAT_MIN + PAT_MAX) / 2)
        
        all_pat_sequences.append(pat_sequence)

    pat_sequences = np.array(all_pat_sequences)
    hr_sequences = np.array(all_hr_sequences)
    quality_mask = np.array(quality_mask)

    print(f"   - PAT sequences: {pat_sequences.shape}")
    print(f"   - HR sequences: {hr_sequences.shape}")
    print(f"   - Valid windows: {quality_mask.sum()}/{len(quality_mask)} ({100*quality_mask.mean():.1f}%)")
    
    if quality_mask.sum() < len(quality_mask) * 0.5:
        print("   ⚠️  WARNING: <50% of windows have good signal quality!")
    
    return pat_sequences, hr_sequences, quality_mask


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
    
    # Handle any remaining NaN/Inf
    pat_sequences = np.nan_to_num(pat_sequences, nan=0.0, posinf=PAT_MAX, neginf=PAT_MIN)
    hr_sequences = np.nan_to_num(hr_sequences, nan=0.0, posinf=HR_MAX, neginf=HR_MIN)

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
