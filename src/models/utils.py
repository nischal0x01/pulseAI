"""
Utility functions for data validation and normalization.
"""

import numpy as np


def _ensure_finite(name, arr):
    """
    Ensures all values in the array are finite (not NaN or Inf).
    Replaces non-finite values with 0.
    """
    arr = np.asarray(arr)
    finite_mask = np.isfinite(arr)
    if finite_mask.all():
        return arr
    n_bad = arr.size - int(finite_mask.sum())
    print(f"⚠️ {name}: found {n_bad} non-finite values (NaN/Inf). Replacing with 0.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_finite_1d(name, y_arr):
    """
    Return (filtered_y, good_mask) where good_mask indexes the original y_arr.
    Filters out non-finite values from a 1D array.
    """
    y_arr = np.asarray(y_arr).reshape(-1)
    finite_mask = np.isfinite(y_arr)
    if finite_mask.all():
        return y_arr, finite_mask
    n_bad = y_arr.size - int(finite_mask.sum())
    print(f"⚠️ {name}: found {n_bad} non-finite labels. Dropping those samples.")
    return y_arr[finite_mask], finite_mask


def normalize_data(X_train, X_val, X_test):
    """
    Normalize data using training set statistics.
    Applies Z-score normalization and handles any remaining non-finite values.
    """
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0) + 1e-8  # avoid division by zero
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    # If anything is still non-finite (e.g., all-NaN timesteps), zero it out
    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_norm = np.nan_to_num(X_test_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train_norm, X_val_norm, X_test_norm
