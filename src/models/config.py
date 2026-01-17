"""
Configuration file for blood pressure prediction model.
Contains hyperparameters, paths, and training configuration.
"""

import numpy as np
import tensorflow as tf

# Random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
VERBOSE = 1
LEARNING_RATE = 1e-4

# Data split configuration
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Signal preprocessing configuration
TARGET_LENGTH = 875  # samples
SAMPLING_RATE = 125  # Hz

# PPG filter configuration
PPG_LOW_CUT = 0.5  # Hz
PPG_HIGH_CUT = 8.0  # Hz
PPG_FILTER_ORDER = 4

# ECG filter configuration
ECG_LOW_CUT = 0.5  # Hz
ECG_HIGH_CUT = 40.0  # Hz
ECG_FILTER_ORDER = 4

# Heart rate bounds
HR_MIN = 40  # bpm
HR_MAX = 180  # bpm

# PAT (Pulse Arrival Time) bounds
PAT_MIN = 0.05  # seconds
PAT_MAX = 0.5  # seconds

# R-peak detection configuration
R_PEAK_HEIGHT_MULTIPLIER = 1.5
MIN_R_PEAK_DISTANCE_MULTIPLIER = 60 / 80  # ~80 bpm minimum

# Model configuration
CONV1D_FILTERS_1 = 64
CONV1D_FILTERS_2 = 128
CONV1D_KERNEL_SIZE = 5
GRU_UNITS_1 = 128
GRU_UNITS_2 = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3

# Paths
PROCESSED_DATA_DIR = '../../data/processed'
CHECKPOINT_DIR = '../../checkpoints'
