"""
Blood Pressure Prediction Models Package

This package implements a physiology-informed CNN-LSTM model with PAT-based attention
mechanism for cuffless blood pressure estimation.

Key Features:
    - 4-channel input: [ECG, PPG, PAT, HR]
    - PAT-based attention mechanism for temporal weighting
    - CNN-LSTM architecture for morphology and dynamics
    - Comprehensive evaluation metrics and visualization
    - Patient-wise data splitting (no leakage)
    - Rigorous data validation (no NaN/Inf)

Main Modules:
    - config: Configuration and hyperparameters
    - data_loader: Data loading from MAT files
    - preprocessing: Signal preprocessing and data splitting
    - feature_engineering: Feature extraction (PAT, HR)
    - model_builder_attention: CNN-LSTM with attention architecture
    - evaluation: Comprehensive metrics and visualization
    - utils: Utility functions for data validation
    - train_attention: Main training pipeline

Usage:
    from src.models.train_attention import main
    model, history, results = main()
"""

__version__ = "2.0.0"

from .config import *
from .data_loader import load_aggregate_data
from .preprocessing import preprocess_signals, create_subject_wise_splits
from .feature_engineering import (
    extract_physiological_features,
    standardize_feature_length,
    create_baseline_features,
    create_4_channel_input
)
from .model_builder_attention import (
    create_phys_informed_model,
    create_attention_visualization_model,
    create_pat_attention_layer
)
from .evaluation import (
    calculate_comprehensive_metrics,
    print_metrics,
    plot_predictions_vs_actual,
    plot_error_distribution,
    visualize_attention_weights,
    comprehensive_evaluation
)
from .utils import _ensure_finite, _ensure_finite_1d, normalize_data
