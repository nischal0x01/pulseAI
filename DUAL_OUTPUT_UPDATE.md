# Dual Output Model Update (SBP + DBP)

## Overview
The model has been updated to predict both **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** simultaneously, providing complete blood pressure readings (e.g., "120/80 mmHg").

## Changes Made

### 1. Data Loader (`src/models/data_loader.py`)
- ✅ Modified `_read_subj_wins()` to return both SBP and DBP arrays
- ✅ Updated `load_aggregate_data()` to load both SBP and DBP labels from MAT files
- ✅ Added validation to filter out rows with NaN in either SBP or DBP
- ✅ Returns 5 values: `signals, sbp_labels, dbp_labels, demographics, patient_ids`

### 2. Model Architecture (`src/models/model_builder_attention.py`)
- ✅ Updated `create_phys_informed_cnn_lstm_attention()` to have **dual output heads**:
  - `sbp_output`: Dense(1) for systolic prediction
  - `dbp_output`: Dense(1) for diastolic prediction
- ✅ Modified model compilation to use separate losses for each output:
  ```python
  loss={'sbp_output': Huber(delta=1.0), 'dbp_output': Huber(delta=1.0)}
  loss_weights={'sbp_output': 1.0, 'dbp_output': 1.0}
  ```
- ✅ Separate metrics tracking for SBP and DBP (MAE, MSE)
- ✅ Attention visualization model returns `[sbp_pred, dbp_pred, attention_weights]`

### 3. Training Script (`src/models/train_attention.py`)
- ✅ Updated data loading to handle both SBP and DBP labels
- ✅ Modified data validation to check both label types
- ✅ Split train/val/test sets for both SBP and DBP
- ✅ Updated model training call with dual targets:
  ```python
  model.fit(X, {'sbp_output': y_sbp, 'dbp_output': y_dbp}, ...)
  ```
- ✅ Enhanced training history plots (2x2 grid):
  - Total loss over time
  - SBP MAE over time
  - DBP MAE over time
  - SBP vs DBP comparison
- ✅ Updated evaluation to handle dual predictions
- ✅ Final summary shows metrics for both SBP and DBP

### 4. Evaluation Module (`src/models/evaluation.py`)
- ✅ Updated `comprehensive_evaluation()` to handle dual outputs
- ✅ Automatic detection of single vs dual output mode
- ✅ Calculates separate metrics for SBP and DBP
- ✅ Generates separate plots for each:
  - Prediction vs Actual (SBP and DBP)
  - Error Distribution (SBP and DBP)
- ✅ Updated plot labels to dynamically show SBP or DBP
- ✅ Backward compatible with single-output models

## Model Architecture

```
Input: [ECG, PPG, PAT, HR] (4 channels)
   ↓
CNN Feature Extraction (Conv1D → BatchNorm → MaxPool → Dropout) × 2
   ↓
PAT-based Attention Mechanism
   ↓
LSTM Layers (temporal modeling)
   ↓
Dense Feature Layer
   ├─→ SBP Output (Systolic)
   └─→ DBP Output (Diastolic)
```

## Training Output Format

The model now outputs:
- **Training mode**: `[sbp_predictions, dbp_predictions]`
- **Attention mode**: `[sbp_predictions, dbp_predictions, attention_weights]`

## Metrics Tracked

For each output (SBP and DBP):
- Mean Absolute Error (MAE) - Target: < 10 mmHg
- Root Mean Squared Error (RMSE)
- R² Score
- Pearson Correlation
- Error Statistics

## How to Use

### Training
```bash
# Run training with dual outputs
python src/models/train_attention.py
```

### Prediction
```python
from src.models.model_builder_attention import create_phys_informed_model

# Load model
model = create_phys_informed_model(input_shape=(timesteps, 4))
model.load_weights('checkpoints/best_model.h5')

# Make predictions
sbp_pred, dbp_pred = model.predict(X_test)

# Display as blood pressure reading
print(f"Blood Pressure: {sbp_pred[0]:.0f}/{dbp_pred[0]:.0f} mmHg")
```

### With Attention Visualization
```python
from src.models.model_builder_attention import create_attention_visualization_model

# Create attention model
attention_model = create_attention_visualization_model(input_shape=(timesteps, 4))
attention_model.set_weights(model.get_weights())

# Get predictions with attention
sbp_pred, dbp_pred, attention_weights = attention_model.predict(X_test)
```

## Data Requirements

The processed data directory must contain MAT files with:
- `SegSBP`: Systolic blood pressure labels
- `SegDBP`: Diastolic blood pressure labels
- `PPG_Raw`: PPG signal data
- `ECG_Raw`: ECG signal data
- Demographics: Age, Gender, Height, Weight

## Success Criteria

The model is considered successful when:
- **SBP MAE < 10 mmHg**
- **DBP MAE < 10 mmHg**
- Both predictions show strong correlation (R² > 0.7)

## Backward Compatibility

All changes maintain backward compatibility:
- Single-output evaluation still works
- Old checkpoint files can be loaded (but won't have DBP predictions)
- Evaluation functions auto-detect output format

## Next Steps

1. **Train the updated model** on your dataset
2. **Compare performance** between SBP and DBP predictions
3. **Integrate with frontend** to display full BP readings (e.g., "120/80 mmHg")
4. **Update real-time prediction server** to output both values
5. **Consider physiological constraints**: DBP should typically be 60-80% of SBP

## Files Modified

1. `/src/models/data_loader.py`
2. `/src/models/model_builder_attention.py`
3. `/src/models/train_attention.py`
4. `/src/models/evaluation.py`

No breaking changes to existing code structure!
