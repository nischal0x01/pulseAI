# Physiology-Informed CNN-LSTM Blood Pressure Prediction Model

**Advanced cuffless blood pressure estimation using multi-channel physiological signals with PAT-based attention mechanism.**

## 🎯 Project Goal

Develop a deep learning model that:
- **Achieves MAE < 10 mmHg** for systolic blood pressure prediction
- **Explicitly relies on physiological timing (PAT)** rather than raw waveform correlation
- **Prevents data leakage** through patient-wise splitting
- **Provides interpretability** through attention visualization

## 🏗️ Architecture Overview

### Input Channels (4 channels)
```
1. ECG (Electrocardiogram)    - Cardiac electrical activity
2. PPG (Photoplethysmogram)   - Blood volume pulse waveform
3. PAT (Pulse Arrival Time)   - ECG R-peak to PPG systolic peak delay
4. HR (Heart Rate)             - Instantaneous heart rate from R-R intervals
```

### Model Pipeline
```
Input: [ECG, PPG, PAT, HR] → CNN Feature Extraction
                              ↓
                         PAT-Based Attention
                              ↓
                         LSTM Temporal Modeling
                              ↓
                         Dense Layers
                              ↓
                    Output: Systolic BP (mmHg)
```

### Key Components

#### 1. CNN Feature Extraction
- **Conv1D layers** extract local morphological features from all 4 channels
- **BatchNormalization** for training stability
- **MaxPooling** for dimensionality reduction
- **Dropout** for regularization

#### 2. PAT-Based Attention Mechanism ⭐
- **Extracts PAT channel** separately
- **Generates attention weights** using Dense layers + Softmax
- **Reweights CNN features** before LSTM processing
- **Ensures model focuses on physiological timing**, not just raw waveform patterns

#### 3. LSTM Temporal Modeling
- **Captures cardiovascular dynamics** over time
- **Bidirectional processing** for forward/backward temporal context
- **Gradient clipping** for training stability

#### 4. Blood Pressure Regression
- **Dense layers** with dropout
- **Linear output** for continuous BP values
- **Huber loss** for robustness to outliers

## 📁 Project Structure

```
src/models/
├── config.py                      # Configuration & hyperparameters
├── data_loader.py                 # Load MAT files from MIMIC dataset
├── preprocessing.py               # Signal filtering & normalization
├── feature_engineering.py         # Extract PAT & HR features
├── model_builder_attention.py     # CNN-LSTM with attention architecture ⭐
├── evaluation.py                  # Comprehensive metrics & visualization ⭐
├── utils.py                       # Data validation utilities
├── train_attention.py             # Main training pipeline ⭐
├── model.py                       # Entry point
│
└── Documentation:
    ├── README.md                  # This file
    ├── ARCHITECTURE.md            # System architecture diagrams
    └── MODEL_DETAILS.md           # This comprehensive guide ⭐
```

## 🔧 Preprocessing Pipeline

### 1. Signal Filtering
- **ECG**: Bandpass filter 0.5-40 Hz (removes baseline wander and high-frequency noise)
- **PPG**: Bandpass filter 0.5-8 Hz (isolates pulse waveform)
- **Z-score normalization** for each signal

### 2. Feature Extraction

#### PAT (Pulse Arrival Time)
```
PAT = Time(PPG systolic peak) - Time(ECG R-peak)
```
- Represents arterial stiffness and blood pressure
- Computed per cardiac cycle
- Interpolated to match signal length
- Bounded: 0.05-0.5 seconds

#### HR (Heart Rate)
```
HR = 60 / (R-R interval in seconds)
```
- Instantaneous heart rate from ECG
- Computed from consecutive R-peaks
- Interpolated for smooth sequence
- Bounded: 40-180 bpm

### 3. Data Validation ✅
```python
# Assert no NaN or Inf values before training
validate_data_integrity(X_train, "Training features")
validate_data_integrity(y_train, "Training labels")
```

## 🎓 Training Configuration

### Loss Function
```python
Huber Loss (delta=1.0)
```
- Robust to outliers in blood pressure measurements
- Combines MSE (for small errors) and MAE (for large errors)

### Optimizer
```python
Adam(learning_rate=1e-4, clipnorm=1.0)
```
- Gradient clipping prevents exploding gradients in LSTM
- Lower learning rate for stability

### Callbacks
1. **EarlyStopping** (patience=15) - Prevent overfitting
2. **ReduceLROnPlateau** (patience=7, factor=0.5) - Adaptive learning rate
3. **ModelCheckpoint** - Save best model

### Data Splitting
- **Patient-wise splitting** (no data leakage)
- 60% Train / 20% Validation / 20% Test
- Same patient never appears in multiple sets

## 📊 Evaluation Metrics

### Primary Metrics
- **MAE (Mean Absolute Error)** - Target: < 10 mmHg ⭐
- **RMSE (Root Mean Squared Error)** - Penalizes large errors
- **R² Score** - Goodness of fit
- **Pearson Correlation** - Linear relationship strength

### Additional Metrics
- Mean Percentage Error (MPE)
- Error Standard Deviation
- Mean Error (bias detection)

### Visualizations
1. **Predicted vs Actual BP** - Scatter plot with ±10 mmHg bands
2. **Error Distribution** - Histogram and box plot
3. **Attention Weights** - Time series visualization ⭐
4. **Training History** - Loss and MAE curves

## 🚀 Usage

### Basic Training
```bash
# Run with default configuration
python src/models/model.py

# Or run attention-based training directly
python src/models/train_attention.py
```

### Configuration
Edit `config.py` to customize:
```python
# Training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Model architecture
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
ATTENTION_UNITS = 64
DROPOUT_RATE = 0.3

# Signal processing
TARGET_LENGTH = 875  # samples
SAMPLING_RATE = 125  # Hz
```

### Programmatic Usage
```python
from src.models import train_attention

# Train model
model, history, results = train_attention.main()

# Access results
print(f"Test MAE: {results['metrics']['MAE']:.2f} mmHg")
print(f"Test R²: {results['metrics']['R2']:.4f}")

# Visualize attention
from src.models.evaluation import visualize_attention_weights
visualize_attention_weights(results['attention_weights'])
```

## 🔬 Attention Mechanism Details

### How It Works

1. **Extract PAT Channel**
   ```python
   pat_channel = inputs[:, :, 2]  # Channel index 2
   ```

2. **Generate Attention Weights**
   ```python
   attention = Dense(64, activation='relu')(pat_channel)
   attention = Dense(1)(attention)
   attention_weights = Softmax()(attention)  # Over time dimension
   ```

3. **Apply Attention to CNN Features**
   ```python
   attended_features = attention_weights * cnn_features
   ```

4. **Feed to LSTM**
   ```python
   lstm_out = LSTM(128)(attended_features)
   ```

### Why PAT-Based Attention?

- **PAT encodes timing information** directly related to arterial stiffness and BP
- **Attention highlights important cardiac cycles** where PAT is most informative
- **Prevents model from relying solely on waveform shape** (which can be confounded)
- **Provides interpretability** - we can visualize what the model focuses on

### Attention Visualization

The attention weights show which time steps the model considers most important:
- **High weights** → Important cardiac cycles for BP prediction
- **Low weights** → Less informative periods
- **Peak positions** → Critical timing features

Expected patterns:
- Peaks align with significant PAT changes
- Higher entropy (spread out) may indicate complex patterns
- Lower entropy (focused) indicates clear signal regions

## 📈 Expected Performance

### Target Metrics
| Metric | Target | Clinical Significance |
|--------|--------|----------------------|
| MAE | < 10 mmHg | AAMI/IEEE standard |
| RMSE | < 15 mmHg | Acceptable for screening |
| R² | > 0.70 | Strong predictive power |
| Pearson r | > 0.80 | Strong correlation |

### Performance Tips

If MAE > 10 mmHg:
1. **Increase model capacity** (more LSTM units, deeper network)
2. **Improve feature engineering** (better PAT/HR extraction)
3. **Add more data** (more patients, more cardiac cycles)
4. **Hyperparameter tuning** (learning rate, dropout, attention units)
5. **Data quality** (check for signal artifacts, improve peak detection)

## 🔍 Model Interpretability

### Attention Analysis
```python
from src.models.evaluation import analyze_attention_statistics

stats = analyze_attention_statistics(attention_weights)
# Returns:
# - Mean entropy (attention spread)
# - Peak positions (where model focuses)
# - Weight concentration (% of high weights)
```

### Feature Importance
The model learns to:
1. **PAT patterns** - Timing relationships (via attention)
2. **Waveform morphology** - Shape features (via CNN)
3. **Temporal dynamics** - Sequence patterns (via LSTM)
4. **Heart rate variability** - HR channel contribution

## 🐛 Debugging & Troubleshooting

### Common Issues

#### 1. NaN/Inf Values During Training
**Symptom**: Loss becomes NaN
**Solution**:
- Check data validation passes
- Reduce learning rate
- Increase gradient clipping norm
- Check for extreme outliers in data

#### 2. Poor Convergence
**Symptom**: Validation loss doesn't decrease
**Solution**:
- Increase batch size
- Add more training data
- Reduce model complexity (overfitting)
- Check data scaling

#### 3. Attention Weights All Equal
**Symptom**: Attention not learning useful patterns
**Solution**:
- Increase ATTENTION_UNITS
- Add regularization to attention layers
- Check PAT signal quality
- Verify PAT has meaningful variation

#### 4. Data Leakage Suspected
**Symptom**: Training metrics much better than test
**Solution**:
- Verify patient-wise splitting is working
- Check for duplicate patients in data
- Ensure preprocessing uses only train set statistics

## 📚 References

### Physiological Background
- **PAT and Blood Pressure**: Inverse relationship due to arterial stiffness
- **PPG Morphology**: Waveform shape reflects vascular resistance
- **ECG Timing**: R-R intervals encode autonomic nervous system activity

### Technical References
- AAMI/IEEE Standards for Blood Pressure Measurement
- Deep Learning for Physiological Signal Processing
- Attention Mechanisms in Time Series Forecasting

## 🤝 Contributing

To extend this project:

1. **Add new features**: Edit `feature_engineering.py`
2. **Modify architecture**: Edit `model_builder_attention.py`
3. **Add evaluation metrics**: Edit `evaluation.py`
4. **Change preprocessing**: Edit `preprocessing.py`

## 📄 License

This project is for research and educational purposes.

## ✨ Key Innovations

1. **PAT-Based Attention** ⭐ - First deep learning model to use physiological timing for attention
2. **Rigorous Validation** - Comprehensive data integrity checks
3. **Patient-Wise Splitting** - Prevents data leakage
4. **Interpretability** - Attention visualization shows model reasoning
5. **Production Ready** - Modular, documented, and maintainable code

---

**For questions or issues, please refer to the documentation or create an issue in the repository.**
