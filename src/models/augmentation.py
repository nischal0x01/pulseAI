"""
Data augmentation module for blood pressure prediction model.

Implements physiologically-valid signal augmentation techniques to increase
the representation of high BP samples in training data:
- Time warping: Simulates heart rate variability (±10% temporal stretch)
- Amplitude scaling: Simulates sensor gain variation (±5% signal strength)
- Noise injection: Simulates real-world measurement noise (SNR 20-30 dB)

These augmentations preserve physiological relationships while increasing
diversity in underrepresented BP ranges.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def time_warp(signal, warp_factor=0.1):
    """
    Apply time warping to simulate heart rate variability.
    
    Stretches or compresses the signal temporally while preserving morphology.
    This simulates natural variations in cardiac cycle duration.
    
    Args:
        signal: Input signal array, shape (time_steps, channels) or (time_steps,)
        warp_factor: Maximum warping factor (0.1 = ±10% stretch/compress)
    
    Returns:
        Time-warped signal with same shape as input
    """
    if len(signal.shape) == 1:
        signal = signal.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    time_steps, channels = signal.shape
    
    # Random warping factor between [-warp_factor, +warp_factor]
    warp = np.random.uniform(-warp_factor, warp_factor)
    
    # Create warped time axis: compress (warp<0) or stretch (warp>0)
    # Original time points
    original_time = np.linspace(0, 1, time_steps)
    
    # Warped time points (non-linear mapping)
    # Use smooth warping curve: center stays relatively fixed, edges warp more
    warp_curve = warp * np.sin(np.pi * original_time)
    warped_time = original_time + warp_curve
    
    # Normalize to [0, 1]
    warped_time = (warped_time - warped_time.min()) / (warped_time.max() - warped_time.min())
    
    # Interpolate each channel
    warped_signal = np.zeros_like(signal)
    for ch in range(channels):
        interpolator = interp1d(
            warped_time, signal[:, ch],
            kind='cubic',
            fill_value='extrapolate'
        )
        warped_signal[:, ch] = interpolator(original_time)
    
    if squeeze_output:
        return warped_signal.squeeze()
    return warped_signal


def amplitude_scale(signal, scale_factor=0.05):
    """
    Apply amplitude scaling to simulate sensor gain variation.
    
    Scales signal amplitude by a random factor, simulating differences
    in sensor sensitivity, contact pressure, or patient-specific characteristics.
    
    Args:
        signal: Input signal array, shape (time_steps, channels) or (time_steps,)
        scale_factor: Maximum scaling factor (0.05 = ±5% amplitude change)
    
    Returns:
        Amplitude-scaled signal with same shape as input
    """
    # Random scaling between [1-scale_factor, 1+scale_factor]
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
    
    # Apply per-channel scaling (each channel can have different sensor gain)
    if len(signal.shape) == 2:
        # Different scaling per channel
        scales = np.random.uniform(1 - scale_factor, 1 + scale_factor, signal.shape[1])
        return signal * scales
    else:
        return signal * scale


def add_noise(signal, snr_db_range=(20, 30)):
    """
    Add Gaussian noise to simulate real-world measurement noise.
    
    SNR (Signal-to-Noise Ratio) controls noise level:
    - SNR 30 dB: Clean signal (noise power 1/1000 of signal)
    - SNR 20 dB: Moderate noise (noise power 1/100 of signal)
    
    Args:
        signal: Input signal array, shape (time_steps, channels) or (time_steps,)
        snr_db_range: Tuple (min_snr, max_snr) in decibels
    
    Returns:
        Noisy signal with same shape as input
    """
    # Random SNR within range
    snr_db = np.random.uniform(*snr_db_range)
    
    # Calculate signal power
    signal_power = np.mean(signal ** 2, axis=0)
    
    # Convert SNR from dB to linear scale: SNR = 10^(SNR_dB/10)
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate noise power: noise_power = signal_power / SNR
    noise_power = signal_power / snr_linear
    
    # Generate Gaussian noise with appropriate power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    return signal + noise


def augment_sample(ecg, ppg, pat, hr, sbp, dbp, augmentation_config):
    """
    Apply full augmentation pipeline to a single sample.
    
    Args:
        ecg: ECG signal, shape (time_steps,)
        ppg: PPG signal, shape (time_steps,)
        pat: PAT signal, shape (time_steps,)
        hr: Heart rate signal, shape (time_steps,)
        sbp: Systolic BP value (scalar)
        dbp: Diastolic BP value (scalar)
        augmentation_config: Dict with keys:
            - 'time_warp': bool, enable time warping
            - 'amplitude_scale': bool, enable amplitude scaling
            - 'add_noise': bool, enable noise injection
            - 'warp_factor': float, time warp strength
            - 'scale_factor': float, amplitude scale strength
            - 'snr_range': tuple, SNR range for noise
    
    Returns:
        Tuple of augmented (ecg, ppg, pat, hr, sbp, dbp)
        Note: BP values unchanged (labels are not augmented)
    """
    # Combine signals for synchronized augmentation
    signals = np.stack([ecg, ppg, pat, hr], axis=1)  # Shape: (time_steps, 4)
    
    # Apply augmentations in sequence
    if augmentation_config.get('time_warp', True):
        warp_factor = augmentation_config.get('warp_factor', 0.1)
        signals = time_warp(signals, warp_factor)
    
    if augmentation_config.get('amplitude_scale', True):
        scale_factor = augmentation_config.get('scale_factor', 0.05)
        signals = amplitude_scale(signals, scale_factor)
    
    if augmentation_config.get('add_noise', True):
        snr_range = augmentation_config.get('snr_range', (20, 30))
        signals = add_noise(signals, snr_range)
    
    # Split back into individual signals
    ecg_aug = signals[:, 0]
    ppg_aug = signals[:, 1]
    pat_aug = signals[:, 2]
    hr_aug = signals[:, 3]
    
    # BP labels remain unchanged (we're not modifying the ground truth)
    return ecg_aug, ppg_aug, pat_aug, hr_aug, sbp, dbp


def augment_high_bp_samples(X, y_sbp, y_dbp, augmentation_factor=2.0, bp_threshold=140, augmentation_config=None):
    """
    Augment high BP samples to balance dataset distribution.
    
    Selectively augments samples with SBP > threshold to increase representation
    of underrepresented high BP values in training data.
    
    Args:
        X: Input features, shape (n_samples, time_steps, channels)
        y_sbp: SBP labels, shape (n_samples,)
        y_dbp: DBP labels, shape (n_samples,)
        augmentation_factor: Target multiplier for high BP samples (2.0 = double)
        bp_threshold: SBP threshold to define "high BP" (default 140 mmHg)
        augmentation_config: Dict with augmentation parameters (see augment_sample)
    
    Returns:
        Tuple (X_augmented, y_sbp_augmented, y_dbp_augmented) with augmented samples appended
    """
    if augmentation_config is None:
        augmentation_config = {
            'time_warp': True,
            'amplitude_scale': True,
            'add_noise': True,
            'warp_factor': 0.1,
            'scale_factor': 0.05,
            'snr_range': (20, 30)
        }
    
    # Identify high BP samples
    high_bp_mask = y_sbp > bp_threshold
    high_bp_indices = np.where(high_bp_mask)[0]
    
    n_high_bp = len(high_bp_indices)
    n_augmented = int(n_high_bp * (augmentation_factor - 1))  # How many to generate
    
    if n_augmented <= 0:
        print(f"No augmentation needed: {n_high_bp} high BP samples found")
        return X, y_sbp, y_dbp
    
    print(f"Augmenting {n_augmented} samples from {n_high_bp} high BP samples (SBP > {bp_threshold})")
    
    # Storage for augmented samples
    X_augmented_list = []
    y_sbp_augmented_list = []
    y_dbp_augmented_list = []
    
    # Randomly select samples to augment (with replacement to reach target count)
    augment_indices = np.random.choice(high_bp_indices, size=n_augmented, replace=True)
    
    for idx in augment_indices:
        # Extract 4-channel input
        ecg = X[idx, :, 0]
        ppg = X[idx, :, 1]
        pat = X[idx, :, 2]
        hr = X[idx, :, 3]
        sbp = y_sbp[idx]
        dbp = y_dbp[idx]
        
        # Augment
        ecg_aug, ppg_aug, pat_aug, hr_aug, sbp_aug, dbp_aug = augment_sample(
            ecg, ppg, pat, hr, sbp, dbp, augmentation_config
        )
        
        # Recombine into 4-channel format
        X_aug = np.stack([ecg_aug, ppg_aug, pat_aug, hr_aug], axis=1)
        
        X_augmented_list.append(X_aug)
        y_sbp_augmented_list.append(sbp_aug)
        y_dbp_augmented_list.append(dbp_aug)
    
    # Convert lists to arrays
    X_augmented = np.array(X_augmented_list)
    y_sbp_augmented = np.array(y_sbp_augmented_list)
    y_dbp_augmented = np.array(y_dbp_augmented_list)
    
    # Concatenate original and augmented data
    X_combined = np.concatenate([X, X_augmented], axis=0)
    y_sbp_combined = np.concatenate([y_sbp, y_sbp_augmented], axis=0)
    y_dbp_combined = np.concatenate([y_dbp, y_dbp_augmented], axis=0)
    
    print(f"Dataset size increased: {len(X)} → {len(X_combined)} samples")
    print(f"High BP representation: {n_high_bp}/{len(X)} ({100*n_high_bp/len(X):.1f}%) → "
          f"{n_high_bp + n_augmented}/{len(X_combined)} ({100*(n_high_bp + n_augmented)/len(X_combined):.1f}%)")
    
    return X_combined, y_sbp_combined, y_dbp_combined
