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
BATCH_SIZE = 16
VERBOSE = 1
LEARNING_RATE = 1e-3  # Lower learning rate for LSTM stability
GRADIENT_CLIP_NORM = 1.0  # Gradient clipping for LSTM stability

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

# Model configuration - CNN-LSTM with Attention (Reduced for laptop training)
CONV1D_FILTERS_1 = 64 
CONV1D_FILTERS_2 = 128 
CONV1D_KERNEL_SIZE = 5
LSTM_UNITS_1 = 128 
LSTM_UNITS_2 = 64 
ATTENTION_UNITS = 64  
DENSE_UNITS = 64 
DROPOUT_RATE = 0.3

# Weighted loss configuration
SBP_LOSS_WEIGHT = 2.0  # SBP weighted 3x more than DBP (increased for better focus)
EXTREME_BP_WEIGHT = 3.0  # High/Low BP weighted 5x more than normal BP (increased)

# Checkpoint resumption configuration  
RESUME_LR_REDUCTION_FACTOR = 0.5  # Reduce LR by 50% when resuming

# Paths configuration
import os

# Check if running on Kaggle
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🤖 Kaggle environment detected. Configuring paths...")
    KAGGLE_INPUT_DIR = '/kaggle/input'
    
    # Locate dataset path under /kaggle/input
    pulsedb_path = None
    if os.path.exists(KAGGLE_INPUT_DIR):
        try:
            for name in os.listdir(KAGGLE_INPUT_DIR):
                full_path = os.path.join(KAGGLE_INPUT_DIR, name)
                if os.path.isdir(full_path) and ('pulsedb' in name.lower() or 'mimic' in name.lower()):
                    pulsedb_path = full_path
                    break
        except Exception as e:
            print(f"⚠️ Error scanning /kaggle/input: {e}")
            
    if pulsedb_path is None:
        # Fallback if no matching name found: use first directory under /kaggle/input
        try:
            subdirs = [os.path.join(KAGGLE_INPUT_DIR, d) for d in os.listdir(KAGGLE_INPUT_DIR) if os.path.isdir(os.path.join(KAGGLE_INPUT_DIR, d))]
            if subdirs:
                pulsedb_path = subdirs[0]
        except Exception:
            pass
            
    if pulsedb_path:
        print(f"📂 Found PulseDB dataset path: {pulsedb_path}")
        RAW_DATA_DIR = pulsedb_path
        if os.path.exists(os.path.join(pulsedb_path, 'processed')):
            PROCESSED_DATA_DIR = os.path.join(pulsedb_path, 'processed')
        elif os.path.exists(os.path.join(pulsedb_path, 'raw')):
            PROCESSED_DATA_DIR = os.path.join(pulsedb_path, 'raw')
        else:
            PROCESSED_DATA_DIR = pulsedb_path
    else:
        print("⚠️ PulseDB dataset folder not found under /kaggle/input. Using default fallback.")
        RAW_DATA_DIR = '/kaggle/input/mimiciii-pulsedb'
        PROCESSED_DATA_DIR = '/kaggle/input/mimiciii-pulsedb'
        
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    print(f"   - PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"   - CHECKPOINT_DIR: {CHECKPOINT_DIR}")
else:
    BASE_DATA_DIR = os.environ.get('SCRATCH', os.path.join(os.path.dirname(__file__), '../..'))
    RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, 'data/raw')
    PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'data/processed')
    CHECKPOINT_DIR = os.path.join(BASE_DATA_DIR, 'checkpoints')

