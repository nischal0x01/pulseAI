# Blood Pressure Prediction Models - Refactored Structure

This document describes the refactored code structure for the blood pressure prediction model.

## 📁 Project Structure

```
src/models/
├── __init__.py              # Package initialization
├── config.py                # Configuration and hyperparameters
├── data_loader.py           # Data loading utilities
├── preprocessing.py         # Signal preprocessing and data splitting
├── feature_engineering.py   # Physiological feature extraction
├── model_builder.py         # Neural network architecture
├── utils.py                 # Utility functions
├── train.py                 # Main training pipeline
├── model.py                 # Entry point (delegates to train.py)
└── model_old.py             # Original monolithic code (backup)
```

## 🔧 Module Descriptions

### `config.py`
Contains all configuration parameters and hyperparameters:
- Random seeds
- Training configuration (epochs, batch size, learning rate)
- Data split ratios
- Signal processing parameters (sampling rate, filter settings)
- Model architecture parameters
- File paths

### `data_loader.py`
Handles loading physiological data from MAT files:
- `_read_subj_wins()`: Read MAT v7.3 HDF5 files
- `load_aggregate_data()`: Load and aggregate data from multiple patient files

### `preprocessing.py`
Signal preprocessing and data splitting:
- `preprocess_signals()`: Filter and normalize ECG and PPG signals
- `create_subject_wise_splits()`: Create train/val/test splits by patient

### `feature_engineering.py`
Extract physiological features from raw signals:
- `extract_physiological_features()`: Extract PAT and HR from ECG/PPG
- `standardize_feature_length()`: Standardize sequence lengths
- `create_baseline_features()`: Create simple statistical features
- `create_4_channel_input()`: Combine raw signals with extracted features

### `model_builder.py`
Neural network architecture definitions:
- `create_phys_informed_model()`: CNN + GRU model for 4-channel input

### `utils.py`
Utility functions for data validation:
- `_ensure_finite()`: Replace NaN/Inf values
- `_ensure_finite_1d()`: Filter out non-finite labels
- `normalize_data()`: Z-score normalization

### `train.py`
Main training pipeline that orchestrates all modules:
- `train_baseline_model()`: Train linear regression baseline
- `train_physiology_informed_model()`: Train deep learning model
- `main()`: Main execution pipeline

### `model.py`
Simple entry point that delegates to `train.py`

## 🚀 Usage

### Running the Model

You can run the model using either of these methods:

```bash
# Method 1: Using the main entry point
python src/models/model.py

# Method 2: Using train.py directly
python src/models/train.py
```

### Importing Modules

You can import specific functions from the package:

```python
from src.models import load_aggregate_data, preprocess_signals
from src.models.config import EPOCHS, BATCH_SIZE

# Load data
signals, labels, demographics, patient_ids = load_aggregate_data()

# Preprocess
processed_signals = preprocess_signals(signals)
```

## 🔄 Changes from Original Code

### Before (model.py - 703 lines)
- Single monolithic file
- All functions mixed together
- Hard-coded constants scattered throughout
- Difficult to maintain and test

### After (Modular Structure)
- **config.py** (65 lines): All configuration in one place
- **data_loader.py** (234 lines): Data loading logic
- **preprocessing.py** (105 lines): Signal preprocessing
- **feature_engineering.py** (155 lines): Feature extraction
- **model_builder.py** (48 lines): Model architecture
- **utils.py** (48 lines): Utility functions
- **train.py** (287 lines): Main training orchestration
- **model.py** (28 lines): Simple entry point

### Benefits
- ✅ **Modularity**: Each module has a single, clear responsibility
- ✅ **Maintainability**: Easy to find and update specific functionality
- ✅ **Testability**: Each module can be tested independently
- ✅ **Reusability**: Functions can be imported and reused elsewhere
- ✅ **Configuration**: All hyperparameters centralized in config.py
- ✅ **Readability**: Clear structure and documentation

## 🐛 Bug Fixes

The refactoring also fixed the following bug from the original code:

### TypeError in `create_baseline_features()`
**Issue**: `pat_sequences` and `hr_sequences` were lists of arrays, causing:
```
TypeError: ufunc 'isnan' not supported for the input types
```

**Fix**: Added `np.asarray()` conversion before checking for NaN values:
```python
pat_sequences = np.asarray(pat_sequences)
hr_sequences = np.asarray(hr_sequences)
```

### Missing Constants
**Issue**: `EPOCHS`, `BATCH_SIZE`, and `VERBOSE` were referenced but not defined

**Fix**: Added these constants to `config.py`

## 📝 Configuration

To modify training parameters, edit `config.py`:

```python
# Training configuration
EPOCHS = 100          # Number of training epochs
BATCH_SIZE = 32       # Batch size for training
LEARNING_RATE = 1e-4  # Learning rate for Adam optimizer

# Data split
TEST_SIZE = 0.2       # 20% for testing
VAL_SIZE = 0.2        # 20% for validation

# Signal parameters
TARGET_LENGTH = 875   # Target signal length (samples)
SAMPLING_RATE = 125   # Sampling rate (Hz)
```

## 🧪 Testing

Each module can be tested independently:

```python
# Test data loading
from src.models.data_loader import load_aggregate_data
signals, labels, demographics, ids = load_aggregate_data()
print(f"Loaded {len(signals)} samples")

# Test preprocessing
from src.models.preprocessing import preprocess_signals
processed = preprocess_signals(signals[:10])  # Test with 10 samples
print(f"Processed shape: {processed.shape}")

# Test feature extraction
from src.models.feature_engineering import extract_physiological_features
pat, hr, peaks = extract_physiological_features(ecg, ppg)
print(f"PAT shape: {pat.shape}, HR shape: {hr.shape}")
```

## 📦 Dependencies

All dependencies are listed in `requirements.txt`. The refactored code uses the same dependencies as the original.

## 🔮 Future Improvements

Potential enhancements to the refactored structure:

1. **Add unit tests** for each module
2. **Add logging** instead of print statements
3. **Add command-line arguments** for configuration
4. **Add model evaluation module** with visualization
5. **Add model checkpoint saving/loading**
6. **Add data augmentation module**
7. **Add experiment tracking** (e.g., MLflow, Weights & Biases)

## 📚 Additional Notes

- The original monolithic code is preserved as `model_old.py` for reference
- All functions maintain the same behavior as the original implementation
- The refactored code follows Python best practices and PEP 8 style guidelines
