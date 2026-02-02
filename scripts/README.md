# PPG Data Acquisition Scripts

Python scripts for reading and visualizing PPG data from ESP32.

## Scripts Overview

1. **ppg_reader.py** - Basic PPG data reception and visualization
2. **ppg_realtime_bp.py** - Real-time blood pressure estimation using trained model

---

## ppg_reader.py

Receives pre-filtered PPG data from ESP32 with on-device signal processing.

### Installation

```bash
pip install pyserial numpy matplotlib
```

### Usage

Basic usage:
```bash
python ppg_reader.py
```

With custom port:
```bash
python ppg_reader.py --port /dev/ttyACM0
```

All options:
```bash
python ppg_reader.py --port /dev/ttyUSB0 --baudrate 115200 --output data.csv
```

Disable plotting (data logging only):
```bash
python ppg_reader.py --no-plot
```

### Command-line Arguments

- `--port`: Serial port (default: `/dev/ttyUSB0`)
  - Linux: `/dev/ttyUSB0`, `/dev/ttyACM0`
  - macOS: `/dev/cu.usbserial-*`
  - Windows: `COM3`, `COM4`, etc.

- `--baudrate`: Baud rate (default: `115200`)

- `--output`: Output CSV filename (default: `ppg_stream.csv`)

- `--no-plot`: Disable real-time plotting for better performance

### Output

Creates a CSV file with columns:
- `timestamp`: Milliseconds from ESP32
- `ir_raw`: Raw IR sensor reading
- `ir_filtered`: Filtered PPG signal from ESP32

### Linux Permissions

If you get a permission error, add your user to the `dialout` group:

```bash
sudo usermod -a -G dialout $USER
```

Then log out and log back in.

### Finding the Serial Port

Linux:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

macOS:
```bash
ls /dev/cu.usb*
```

Windows:
```
Device Manager → Ports (COM & LPT)
```

## Example Workflow

1. Upload ESP32 sketch (see `../esp32/README.md`)
2. Connect ESP32 to computer via USB
3. Run the Python script:
   ```bash
   python ppg_reader.py
   ```
4. Place finger on MAX30102 sensor
5. Observe real-time PPG waveform
6. Press Ctrl+C to stop
7. Data is saved to `ppg_stream.csv`

## Troubleshooting

### "No module named 'serial'"
```bash
pip install pyserial
```

### "Permission denied" on Linux
```bash
sudo chmod 666 /dev/ttyUSB0
# Or permanently:
sudo usermod -a -G dialout $USER
```

### "Port not found"
- Check ESP32 is connected
- Try different USB cable
- Check `dmesg | grep tty` on Linux
- Look in Device Manager on Windows

### No data or timeout
- Verify ESP32 sketch is uploaded and running
- Check baud rate matches (115200)
- Open Arduino Serial Monitor to verify ESP32 output
- Try resetting ESP32

### Plot is slow or laggy
- Use `--no-plot` for data collection only
- Reduce `WINDOW_SIZE` in script
- Close other applications

---

## ppg_realtime_bp.py

Real-time blood pressure estimation using the trained CNN-LSTM model.

### Features
- Receives pre-filtered PPG from ESP32
- Buffers data into 7-second windows (875 samples @ 125 Hz)
- Extracts physiological features (PAT, HR)
- Predicts SBP/DBP using trained model
- Real-time visualization of signals and predictions

### Installation

```bash
pip install pyserial numpy matplotlib tensorflow scipy pandas scikit-learn
```

Or use the project requirements:
```bash
pip install -r requirements/requirements.txt
```

### Prerequisites

1. **Trained Model**: You must have a trained model at `checkpoints/best_model.h5`
   ```bash
   python src/models/model.py  # Train the model first
   ```

2. **ESP32 Setup**: ESP32 must be running the filtered PPG sketch
   ```bash
   # See esp32/README.md for setup
   ```

### Usage

Basic usage:
```bash
python ppg_realtime_bp.py
```

With custom settings:
```bash
python ppg_realtime_bp.py --port /dev/ttyUSB0 --model checkpoints/best_model.h5
```

All options:
```bash
python ppg_realtime_bp.py \
  --port /dev/ttyUSB0 \
  --baudrate 115200 \
  --model checkpoints/best_model.h5 \
  --output bp_predictions.csv
```

Disable plotting (faster):
```bash
python ppg_realtime_bp.py --no-plot
```

### Command-line Arguments

- `--port`: Serial port (default: `/dev/ttyUSB0`)
- `--baudrate`: Baud rate (default: `115200`)
- `--model`: Path to trained model (default: `checkpoints/best_model.h5`)
- `--output`: Output CSV filename (default: `bp_predictions.csv`)
- `--no-plot`: Disable real-time plotting

### Output

**Console Output:**
```
[  1] BP:  118.3/ 78.2 mmHg  (samples: 250)
[  2] BP:  119.1/ 78.8 mmHg  (samples: 500)
[  3] BP:  117.5/ 77.6 mmHg  (samples: 750)
```

**CSV Output** (`bp_predictions.csv`):
```csv
prediction_id,timestamp,sbp,dbp,samples_used
1,1738512345.678,118.3,78.2,1400
2,1738512347.890,119.1,78.8,1400
```

**Real-time Plots:**
1. **Top**: Filtered PPG waveform
2. **Middle**: SBP/DBP prediction history
3. **Bottom Left**: Latest BP reading (large display)
4. **Bottom Right**: Status information

### How It Works

1. **Data Collection**
   - Reads filtered PPG from ESP32 (200 Hz)
   - Buffers ~1400 samples (7 seconds)

2. **Preprocessing**
   - Downsamples to 125 Hz (model's training rate)
   - Generates simulated ECG from PPG (for demo; use real ECG in production)
   - Normalizes signals

3. **Feature Extraction**
   - Extracts Pulse Arrival Time (PAT)
   - Calculates Heart Rate (HR)
   - Creates 4-channel input: [ECG, PPG, PAT, HR]

4. **Prediction**
   - Feeds window to CNN-LSTM model
   - Outputs SBP and DBP estimates
   - Updates every 250 samples (~2 seconds)

### Important Notes

⚠️ **ECG Simulation**: Currently uses a simulated ECG derived from PPG. For accurate predictions:
- Use a real ECG sensor (e.g., AD8232)
- Connect to ESP32 and modify the sketch to send both ECG and PPG
- Update the script to use real ECG instead of `simulate_ecg_from_ppg()`

⚠️ **Model Accuracy**: Predictions depend on:
- Model training quality
- Signal quality from sensors
- Proper sensor placement
- Individual calibration

⚠️ **Medical Use**: This is for research/educational purposes only. Not for medical diagnosis.

### Troubleshooting

**"Model not found"**
```bash
# Train the model first
python src/models/model.py
```

**"Error loading model"**
- Check TensorFlow version compatibility
- Ensure model was trained with same TensorFlow version
- Try: `pip install tensorflow==2.15.0`

**Poor predictions**
- Ensure good PPG signal quality
- Check sensor contact
- Verify model is properly trained
- Collect more training data for your specific setup

**Slow performance**
- Use `--no-plot` to disable visualization
- Close other applications
- Consider using a GPU for inference

**"Feature extraction failed"**
- Signal quality may be poor
- Check PPG waveform for artifacts
- Ensure proper sensor placement
- Reduce motion

### Performance Metrics

- **Latency**: ~100ms per prediction
- **Update Rate**: Every 2 seconds (configurable via `UPDATE_INTERVAL`)
- **Memory**: ~500MB (model + buffers)
- **CPU Usage**: ~20-30% on modern CPU

### Example Workflow

1. **Train Model** (one-time setup):
   ```bash
   python src/models/model.py
   ```

2. **Upload ESP32 Sketch** (one-time setup):
   - See `esp32/README.md`

3. **Start Real-time Prediction**:
   ```bash
   python scripts/ppg_realtime_bp.py --port /dev/ttyUSB0
   ```

4. **Place Finger on Sensor**:
   - Keep still for best results
   - Wait for buffer to fill (~7 seconds)
   - Observe predictions updating every ~2 seconds

5. **Stop and Review**:
   - Press Ctrl+C
   - Check `bp_predictions.csv` for logged data
