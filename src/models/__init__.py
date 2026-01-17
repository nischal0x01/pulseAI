"""
Blood Pressure Prediction Models Package

This package contains modules for training blood pressure prediction models
from physiological signals (ECG and PPG).

Modules:
    - config: Configuration and hyperparameters
    - data_loader: Data loading utilities
    - preprocessing: Signal preprocessing and data splitting
    - feature_engineering: Feature extraction from physiological signals
    - model_builder: Model architecture definitions
    - utils: Utility functions for data validation
    - train: Main training pipeline
"""

__version__ = "1.0.0"

from .config import *
from .data_loader import load_aggregate_data
from .preprocessing import preprocess_signals, create_subject_wise_splits
from .feature_engineering import (
    extract_physiological_features,
    standardize_feature_length,
    create_baseline_features,
    create_4_channel_input
)
from .model_builder import create_phys_informed_model
from .utils import _ensure_finite, _ensure_finite_1d, normalize_data
