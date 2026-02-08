# Real-time Blood Pressure Estimation Pipeline

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ESP32 + MAX30102 + ECG Sensor                    │
│                                                                     │
│  1. Read Raw PPG from MAX30102 (250 Hz)                            │
│  2. Read Raw ECG from sensor (500 Hz)                              │
│  3. Read Heart Rate from sensor (if available)                     │
│  4. Output: timestamp,ppg_raw,ecg_raw,hr (optional)                │
└────────────────────┬────────────────────────────────────────────────┘
                     │ Serial (115200 baud)
                     │ USB Cable
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Python Script (bridge_server.py)                 │
│                                                                     │
│  STEP 1: Data Reception & Filtering                                │
│  ├─ Read RAW PPG and ECG values from serial                        │
│  ├─ Apply bandpass filters:                                        │
│  │  • PPG: 0.5-4.0 Hz (IIR, 3rd order Butterworth)                 │
│  │  • ECG: 0.5-40.0 Hz (IIR, 3rd order Butterworth)                │
│  ├─ Buffer filtered data (~1400 PPG, ~2800 ECG samples)            │
│  └─ Downsample to 125 Hz → 875 samples                             │
│                                                                     │
│  STEP 2: Signal Quality Validation                                 │
│  ├─ Check ECG quality (std > 0.01, no NaN/Inf)                     │
│  ├─ Check PPG quality (std > 0.01, no NaN/Inf)                     │
│  ├─ Validate filtered signals (not raw)                            │
│  └─ Calculate combined quality score (0.0 - 1.0)                   │
│                                                                     │
│  STEP 3: Feature Extraction                                        │
│  ├─ Generate simulated ECG (or use real ECG)                       │
│  ├─ Detect R-peaks (ECG) and PPG feet                              │
│  ├─ Calculate PAT (Pulse Arrival Time)                             │
│  ├─ Calculate HR (Heart Rate)                                      │
│  └─ Reject low quality signals (use defaults if quality fails)     │
│                                                                     │
│  STEP 4: Create 4-Channel Input                                    │
│  ├─ Channel 1: ECG (normalized with separate statistics)           │
│  ├─ Channel 2: PPG (normalized with separate statistics)           │
│  ├─ Channel 3: PAT sequence                                        │
│  └─ Channel 4: HR sequence                                         │
│         Shape: (1, 875, 4)                                          │
│                                                                     │
│  STEP 5: Quality Threshold Check                                   │
│  ├─ Require signal quality ≥ 60%                                   │
│  ├─ Block predictions if quality too low                           │
│  └─ Send error message to frontend                                 │
│                                                                     │
│  STEP 6: Model Inference                                           │
│  ├─ Load trained CNN-LSTM model                                    │
│  ├─ Forward pass through network                                   │
│  └─ Output: [SBP, DBP] predictions                                 │
│                                                                     │
│  STEP 7: Visualization & Logging                                   │
│  ├─ Real-time plots (PPG, BP history)                              │
│  ├─ Console output with quality indicators                         │
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
PPG Sensor (MAX30102) → ADC Read (12-bit) → Serial Output (RAW)
                           ↓
                        250 Hz

ECG Sensor (AD8232)   → ADC Read (12-bit) → Serial Output (RAW)
                           ↓
                        500 Hz

Format: timestamp,ppg_raw,ecg_raw,hr
```

### On Host (Python - bridge_server.py):
```
RAW Signals → IIR Bandpass Filter → Buffer → Downsample → Normalization → Quality → Features → Model
    ↓              ↓                   ↓         ↓             ↓              ↓          ↓
250/500 Hz   PPG: 0.5-4Hz          250/500   125 Hz    Separate stats   ≥60%    PAT, HR
             ECG: 0.5-40Hz          samples              (PPG & ECG)   required
             (3rd order IIR)
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
│   │   └── ppg_reader_filtered.ino    # ESP32 sketch (reads raw PPG + ECG)
│   └── README.md                       # ESP32 setup guide
│
├── bridge_server.py                   # WebSocket server (real-time filtering & BP)
├── scripts/
│   ├── ppg_reader.py                  # Basic PPG visualization
│   ├── ppg_realtime_bp.py             # Real-time BP estimation (alternative)
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

### 1. Real-time BP Estimation with WebSocket (Recommended)
```bash
python bridge_server.py
# Then open frontend at http://localhost:3000
```

### 2. Real-time BP Estimation with Visualization (Standalone)
```bash
python scripts/ppg_realtime_bp.py --port /dev/ttyUSB0
```

### 3. Basic Data Collection
```bash
python scripts/ppg_reader.py --port /dev/ttyUSB0
```

### 4. Custom Serial Port
```bash
python bridge_server.py --port /dev/ttyUSB1 --baudrate 115200
```

## Output Examples

### Console:
```
[  1] BP:  118.3/ 78.2 mmHg (quality: 85.2%, sent to 1 clients)
[  2] BP:  119.1/ 78.8 mmHg (quality: 87.3%, sent to 1 clients)
Signal quality too low (45.0%) - skipping prediction. Connect both PPG and ECG sensors.
[  3] BP:  117.5/ 77.6 mmHg (quality: 82.1%, sent to 1 clients)
```

### CSV (`bp_predictions.csv`):
```csv
prediction_id,timestamp,sbp,dbp,samples_used
1,1738512345.678,118.3,78.2,1400
2,1738512347.890,119.1,78.8,1400
3,1738512350.123,117.5,77.6,1400
```

## Signal Quality Validation

### Quality Checks Applied:

1. **ECG Signal Quality**:
   - Bandpass filter: 0.5-40 Hz (3rd order Butterworth)
   - Standard deviation > 0.01
   - No NaN or Inf values
   - Independent normalization statistics

2. **PPG Signal Quality**:
   - Bandpass filter: 0.5-8 Hz (3rd order Butterworth)
   - Standard deviation > 0.01
   - No NaN or Inf values
   - Independent normalization statistics

3. **Combined Quality Score**:
   - Both signals valid: 80-100% (average of both)
   - Only PPG valid: Max 50%
   - Only ECG valid: Max 50%
   - Both invalid: 0%

4. **Prediction Threshold**:
   - Requires ≥ 60% signal quality
   - Below threshold: Display "--/--" with warning
   - Prevents misleading predictions from poor signals

### Quality Indicators:

| Quality | Status | Action |
|---------|--------|--------|
| 80-100% | Excellent | Normal predictions |
| 60-79% | Good | Predictions allowed |
| 40-59% | Poor | Predictions blocked |
| 0-39% | Failed | Check connections |

## Important Notes

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. Not intended for medical diagnosis or treatment.

⚠️ **ECG Requirement**: For production use with real ECG sensor:
- Add ECG sensor (e.g., AD8232)
- Modify ESP32 sketch to read both sensors
- Update `bridge_server.py` for dual-sensor input
- Signal quality will automatically validate both sensors

⚠️ **Calibration**: Model may require individual calibration for best accuracy.

⚠️ **Signal Quality**: Ensure:
- Good sensor contact (both PPG and ECG)
- Minimal motion during measurement
- Proper electrode/sensor placement
- Clean sensors and skin contact areas
- ECG sensor connected (or quality will drop to ~50%)
