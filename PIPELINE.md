# Real-time Blood Pressure Estimation Pipeline

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ESP32 + MAX30102                            │
│                                                                     │
│  1. Read Raw PPG (200 Hz)                                          │
│  2. DC Removal (α = 0.01)                                          │
│  3. Bandpass Filter (0.7-4 Hz, 3rd order Butterworth)              │
│  4. Output: timestamp,ir_raw,filtered_value                        │
└────────────────────┬────────────────────────────────────────────────┘
                     │ Serial (115200 baud)
                     │ USB Cable
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Python Script (ppg_realtime_bp.py)               │
│                                                                     │
│  STEP 1: Data Reception & Buffering                                │
│  ├─ Read filtered PPG values                                       │
│  ├─ Buffer ~1400 samples (7 seconds @ 200 Hz)                      │
│  └─ Downsample to 125 Hz → 875 samples                             │
│                                                                     │
│  STEP 2: Feature Extraction                                        │
│  ├─ Generate simulated ECG (or use real ECG)                       │
│  ├─ Detect R-peaks (ECG) and PPG feet                              │
│  ├─ Calculate PAT (Pulse Arrival Time)                             │
│  └─ Calculate HR (Heart Rate)                                      │
│                                                                     │
│  STEP 3: Create 4-Channel Input                                    │
│  ├─ Channel 1: ECG (normalized)                                    │
│  ├─ Channel 2: PPG (normalized)                                    │
│  ├─ Channel 3: PAT sequence                                        │
│  └─ Channel 4: HR sequence                                         │
│         Shape: (1, 875, 4)                                          │
│                                                                     │
│  STEP 4: Model Inference                                           │
│  ├─ Load trained CNN-LSTM model                                    │
│  ├─ Forward pass through network                                   │
│  └─ Output: [SBP, DBP] predictions                                 │
│                                                                     │
│  STEP 5: Visualization & Logging                                   │
│  ├─ Real-time plots (PPG, BP history)                              │
│  ├─ Console output                                                 │
│  └─ CSV logging                                                    │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
         Results: BP = 120/80 mmHg
```

## Model Architecture (CNN-LSTM with Attention)

```
Input: (875, 4)
    ↓
[Conv1D Layers]
    ├─ Extract morphological features
    ├─ 32 → 64 → 128 filters
    └─ MaxPooling for dimensionality reduction
    ↓
[PAT-based Attention]
    ├─ Weight temporal importance
    └─ Focus on cardiovascular events
    ↓
[LSTM Layers]
    ├─ Model temporal dynamics
    └─ Capture cardiovascular patterns
    ↓
[Dense Layers]
    ├─ Regression layers
    └─ Output: [SBP, DBP]
    ↓
Output: Blood Pressure (mmHg)
```

## Signal Processing Details

### On ESP32 (Embedded):
```
Raw PPG (12-bit ADC) → DC Removal → Bandpass Filter → Serial Output
                          ↓              ↓
                    EMA smoothing    IIR Filter
                    α = 0.01         6th order
                                     0.7-4 Hz
```

### On Host (Python):
```
Filtered PPG → Downsample → Normalization → Feature Extraction → Model
     ↓            ↓             ↓                  ↓
  200 Hz      125 Hz      Z-score         PAT, HR, etc.
```

## Timing Diagram

```
Time (seconds):  0    1    2    3    4    5    6    7    8    9    10
                 ├────┴────┴────┴────┴────┴────┴────┼────┴────┴────┤
                 │                                  │               │
                 │← Fill Buffer (7 sec) ────────────┤               │
                                                    ▼               │
                                              [Prediction #1]       │
                                                                    │
                 │                                  │← Continue ───┤
                                                    ▼
                                              [Prediction #2]
                                              (every 2 sec)
```

## File Structure

```
cuffless-blood-pressure/
│
├── esp32/
│   ├── ppg_reader_filtered/
│   │   └── ppg_reader_filtered.ino    # ESP32 sketch with filtering
│   └── README.md                       # ESP32 setup guide
│
├── scripts/
│   ├── ppg_reader.py                  # Basic PPG visualization
│   ├── ppg_realtime_bp.py             # Real-time BP estimation
│   └── README.md                       # Usage documentation
│
├── src/models/
│   ├── model_builder_attention.py     # CNN-LSTM architecture
│   ├── feature_engineering.py         # PAT, HR extraction
│   ├── preprocessing.py               # Signal preprocessing
│   └── train_attention.py             # Model training
│
├── checkpoints/
│   ├── best_model.h5                  # Trained model weights
│   └── training_metadata.json         # Training metrics
│
└── data/
    ├── processed/                     # Training data
    └── raw/                           # Original dataset
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| **ESP32 Processing** | <5ms per sample |
| **Serial Transfer** | ~0.1ms per sample |
| **Python Buffering** | Real-time (200 Hz) |
| **Feature Extraction** | ~50ms per window |
| **Model Inference** | ~100ms per prediction |
| **Total Latency** | ~150ms |
| **Update Rate** | Every 2 seconds |

## Dependencies

### ESP32:
- Arduino IDE
- ESP32 board support
- SparkFun MAX3010x library

### Python:
- tensorflow (model inference)
- pyserial (serial communication)
- scipy (signal processing)
- numpy (numerical operations)
- matplotlib (visualization)
- pandas (data handling)

## Usage Examples

### 1. Data Collection Only
```bash
python scripts/ppg_reader.py --port /dev/ttyUSB0
```

### 2. Real-time BP Estimation with Visualization
```bash
python scripts/ppg_realtime_bp.py --port /dev/ttyUSB0
```

### 3. Headless Mode (no plots, faster)
```bash
python scripts/ppg_realtime_bp.py --no-plot --output my_bp_data.csv
```

### 4. Custom Model Path
```bash
python scripts/ppg_realtime_bp.py --model /path/to/custom_model.h5
```

## Output Examples

### Console:
```
[  1] BP:  118.3/ 78.2 mmHg  (samples: 250)
[  2] BP:  119.1/ 78.8 mmHg  (samples: 500)
[  3] BP:  117.5/ 77.6 mmHg  (samples: 750)
```

### CSV (`bp_predictions.csv`):
```csv
prediction_id,timestamp,sbp,dbp,samples_used
1,1738512345.678,118.3,78.2,1400
2,1738512347.890,119.1,78.8,1400
3,1738512350.123,117.5,77.6,1400
```

## Important Notes

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. Not intended for medical diagnosis or treatment.

⚠️ **ECG Requirement**: Current implementation uses simulated ECG. For production use:
- Add real ECG sensor (e.g., AD8232)
- Modify ESP32 sketch to read both sensors
- Update Python script to use real ECG data

⚠️ **Calibration**: Model may require individual calibration for best accuracy.

⚠️ **Signal Quality**: Ensure:
- Good sensor contact
- Minimal motion
- Proper finger placement
- Clean sensors
