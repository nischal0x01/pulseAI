# Cuffless Blood Pressure Estimation

A machine learning project for estimating blood pressure from PPG (Photoplethysmography) and ECG (Electrocardiography) signals using deep learning techniques. This project implements CNN-based feature extraction combined with various regression models for non-invasive blood pressure prediction.

## Project Structure(T)

### 📁 `data/`
Contains all datasets and data-related files for the project.

- **`raw/`**: Raw PPG/ECG signal files from PulseDB dataset or other biomedical signal databases
- **`processed/`**: Cleaned, filtered, and normalized signal data ready for training
- **`splits/`**: Train/validation/test dataset splits stored as CSV files or indices

### 📁 `notebooks/` 
Jupyter notebooks for exploratory data analysis and prototyping.

- **`01_data_exploration.ipynb`**: Initial data exploration, visualization of PPG/ECG signals, and statistical analysis
- **`02_preprocessing.ipynb`**: Signal preprocessing pipeline development and testing
- **`03_model_prototyping.ipynb`**: Model architecture experimentation and initial results

### 📁 `src/`
Source code modules organized by functionality.

#### 🔄 `data_loading/`
Data loading and signal segmentation utilities.
- **`loader.py`**: Functions to load raw PulseDB data and other biomedical signal formats
- **`segmentation.py`**: Segment continuous signals into fixed-length windows for training

#### 🧹 `preprocessing/`
Signal preprocessing and cleaning modules.
- **`filters.py`**: Digital signal processing filters (bandpass, smoothing, detrending)
- **`normalization.py`**: Signal normalization and scaling techniques
- **`utils.py`**: Helper functions for preprocessing tasks

#### 🔍 `features/`
Feature extraction components.
- **`cnn_feature_extractor.py`**: CNN-based feature extraction model for PPG/ECG signals

#### 🤖 `models/`
Deep learning model implementations.
- **`cnn_svr.py`**: CNN feature extractor + Support Vector Regression pipeline
- **`cnn_lstm.py`**: CNN + Long Short-Term Memory network for temporal modeling
- **`cnn_gru.py`**: CNN + Gated Recurrent Unit network architecture
- **`layers.py`**: Custom neural network layers and components

#### 🏋️ `training/`
Training pipeline and model evaluation.
- **`train.py`**: Main training loop with hyperparameter management
- **`evaluate.py`**: Model validation and testing procedures
- **`checkpoints.py`**: Model checkpointing and state management

#### 🛠️ `utils/`
Utility functions and configurations.
- **`metrics.py`**: Evaluation metrics (MAE, RMSE, R², correlation coefficients)
- **`plot.py`**: Visualization functions for PPG/ECG signals and training curves
- **`config.py`**: Global configuration settings and hyperparameters

#### 🔮 `inference/`
Model deployment and prediction.
- **`predict.py`**: Use trained models to predict blood pressure from new signal data

### 📁 `experiments/`
Experiment configurations and results tracking.
- **`cnn_svr_experiment.json`**: Configuration for CNN+SVR experiments
- **`cnn_lstm_experiment.json`**: Configuration for CNN+LSTM experiments
- **`logs/`**: Training logs, metrics, and experiment results

### 📁 `checkpoints/`
Saved model weights and training states.
- Contains model checkpoints saved during training (e.g., `model_epoch_xx.pt`)

### 📋 Root Files
- **`requirements.txt`**: Python package dependencies
- **`main.py`**: Entry point script to run training or inference
- **`README.md`**: This documentation file

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   - Place raw PPG/ECG data in `data/raw/`
   - Run preprocessing notebooks to generate processed data

3. **Train Models**
   ```bash
   python main.py --mode train --model cnn_svr
   ```

4. **Run Inference**
   ```bash
   python main.py --mode predict --model cnn_svr --input path/to/signal.csv
   ```

## Features

- 🔬 **Multi-modal Signal Processing**: Handles both PPG and ECG signals
- 🧠 **Deep Learning Models**: CNN, LSTM, GRU architectures for blood pressure estimation
- 📊 **Comprehensive Evaluation**: Multiple regression metrics and visualization tools
- ⚡ **Flexible Pipeline**: Modular design for easy experimentation
- 💾 **Experiment Tracking**: JSON-based configuration and logging system

## Models Implemented

1. **CNN + SVR**: Convolutional feature extraction with Support Vector Regression
2. **CNN + LSTM**: Deep learning approach with temporal modeling
3. **CNN + GRU**: Alternative recurrent architecture for sequence processing

## Dataset

This project is designed to work with biomedical signal datasets such as:
- PulseDB (PPG signals)
- PhysioNet databases
- Custom PPG/ECG recordings

*Note: Due to privacy and licensing, datasets are not included in this repository.*
