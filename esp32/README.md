# ESP32 PPG + ECG Reader with On-Device Filtering

This directory contains the ESP32 Arduino sketch for reading PPG and ECG data from sensors with on-device signal processing.

## Features

- **PPG Processing (MAX30102)**:
  - DC Component Removal (EMA filter)
  - Bandpass Filtering (0.7-4 Hz, 3rd order Butterworth)
  
- **ECG Processing (AD8232)**:
  - Bandpass Filtering (0.5-40 Hz, 3rd order Butterworth)
  - Leads-off detection
  
- **Real-time Processing**: All filtering happens on ESP32
- **High Sample Rate**: 200 Hz sampling for accurate signal capture

## Hardware Requirements

### Sensors
- **ESP32 development board**
- **MAX30102 pulse oximeter** (PPG sensor)
- **AD8232 ECG sensor module**

### Connections

#### MAX30102 (I2C):
- SDA → GPIO 21 (default I2C SDA on ESP32)
- SCL → GPIO 22 (default I2C SCL on ESP32)
- VIN → 3.3V
- GND → GND

#### AD8232 (Analog + Digital):
- OUTPUT → GPIO 34 (ADC1_CH6, analog input)
- LO+ → GPIO 32 (leads-off detection)
- LO- → GPIO 33 (leads-off detection)
- 3.3V → 3.3V
- GND → GND

**ECG Electrode Placement:**
- Red (RA): Right arm or right side of chest
- Yellow (LA): Left arm or left side of chest  
- Green (RL): Right leg or lower right abdomen (reference)

## Software Requirements

### Arduino IDE Setup

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Add ESP32 board support:
   - File → Preferences
   - Add to "Additional Board Manager URLs": 
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Tools → Board → Boards Manager → Search "ESP32" → Install

3. Install required library:
   - Tools → Manage Libraries
   - Search "MAX30105" → Install "SparkFun MAX3010x Pulse and Proximity Sensor Library"

## Installation

1. Open `ppg_reader_filtered.ino` in Arduino IDE
2. Select your ESP32 board: Tools → Board → ESP32 Arduino → (your board)
3. Select the correct port: Tools → Port → (your port)
4. Upload the sketch: Sketch → Upload

## Configuration

### Sensor Parameters (in `setup()`):

```cpp
byte ledBrightness = 60;   // LED brightness (0-255)
byte sampleAverage = 1;     // Number of samples to average (1, 2, 4, 8, 16, 32)
int sampleRate = 200;       // Sample rate in Hz (50, 100, 200, 400, 800, 1000, 1600, 3200)
int pulseWidth = 411;       // LED pulse width in µs (69, 118, 215, 411)
int adcRange = 4096;        // ADC range (2048, 4096, 8192, 16384)
```

### Filter Parameters:

**PPG Filter:**
```cpp
const float DC_ALPHA = 0.01;  // DC removal smoothing factor (0.001-0.1)
```

**ECG Pins:**
```cpp
const int ECG_PIN = 34;        // Change if using different ADC pin
const int LO_PLUS_PIN = 32;    // Leads-off detection LO+
const int LO_MINUS_PIN = 33;   // Leads-off detection LO-
```

The bandpass filter coefficients are pre-calculated:
- PPG: 0.7-4 Hz at 200 Hz sampling rate
- ECG: 0.5-40 Hz at 200 Hz sampling rate

To change these:

1. Use Python with scipy:
   ```python
   from scipy.signal import butter
   
   # For PPG (0.7-4 Hz)
   ppg_b, ppg_a = butter(N=3, Wn=[0.7, 4], btype='band', fs=200)
   print("PPG b_coeff:", ppg_b)
   print("PPG a_coeff:", ppg_a)
   
   # For ECG (0.5-40 Hz)
   ecg_b, ecg_a = butter(N=3, Wn=[0.5, 40], btype='band', fs=200)
   print("ECG b_coeff:", ecg_b)
   print("ECG a_coeff:", ecg_a)
   ```

2. Update the corresponding `ppg_b_coeff`, `ppg_a_coeff`, `ecg_b_coeff`, and `ecg_a_coeff` arrays in the sketch

## Output Format

The ESP32 outputs data via serial (115200 baud) in CSV format:

```
timestamp,ppg_raw,ppg_filtered,ecg_raw,ecg_filtered
```

- `timestamp`: Milliseconds since ESP32 boot
- `ppg_raw`: Raw IR sensor reading (ADC value)
- `ppg_filtered`: Filtered PPG signal value
- `ecg_raw`: Raw ECG ADC reading (0-4095 for 12-bit)
- `ecg_filtered`: Filtered ECG signal value

**Note**: When ECG leads are not properly attached (leads-off detection), `ecg_raw` and `ecg_filtered` will be 0.

## Usage with Python

Use the companion Python script to receive and visualize the data:

```bash
cd ../scripts
python ppg_reader.py --port /dev/ttyUSB0
```

## Troubleshooting

### MAX30102 Not Found
- Check I2C connections (SDA, SCL)
- Verify power supply (3.3V)
- Try different I2C pins if necessary

### AD8232 No Signal
- Verify electrode placement and contact
- Check leads-off detection (should see valid ECG values when attached)
- Ensure proper grounding
- Clean skin before applying electrodes

### ECG Always Zero
- Check LO+ and LO- pins are connected
- Verify electrodes are properly attached to body
- Try reversing electrode positions
- Check OUTPUT pin connection to GPIO 34

### No Data or Garbled Output
- Check baud rate matches (115200)
- Ensure correct serial port selected
- Try resetting the ESP32

### Poor Signal Quality
- **PPG**: 
  - Adjust `ledBrightness` (typically 30-100)
  - Ensure good sensor contact with finger
  - Reduce motion artifacts
  - Try different `sampleAverage` values
  
- **ECG**:
  - Use alcohol wipes to clean skin
  - Ensure electrodes have good adhesion
  - Minimize motion during recording
  - Keep away from electrical interference

### Filter Not Working
- Verify filter coefficients are correct
- Check that `SAMPLE_RATE` matches sensor configuration
- Ensure buffers are properly initialized

## Technical Details

### PPG DC Component Removal

Uses exponential moving average (EMA):
```
dc_estimate = dc_estimate + α × (ir - dc_estimate)
ac_component = ir - dc_estimate
```

### Bandpass Filters

Both signals use 6th-order IIR filters (cascaded biquads represented as single Direct Form II):

**PPG Filter:**
- Passband: 0.7-4 Hz (typical heart rate range)
- Purpose: Extract cardiac pulsation
- Filter type: Butterworth (flat passband response)

**ECG Filter:**
- Passband: 0.5-40 Hz (diagnostic ECG range)
- Purpose: Remove baseline wander and high-frequency noise
- Filter type: Butterworth

### Leads-Off Detection

The AD8232 provides leads-off detection pins:
- When electrodes are properly attached: LO+ = LOW, LO- = LOW
- When electrodes are detached: LO+ = HIGH or LO- = HIGH
- ESP32 checks both pins before reading ECG data

## Performance

- Memory usage: ~2KB RAM for filter buffers
- CPU usage: <10% at 200 Hz sampling rate
- Latency: <5ms per sample
- ADC resolution: 12-bit (0-4095) for ECG

## Wiring Diagram

```
ESP32           MAX30102        AD8232
-----           --------        ------
3.3V     -----> VIN             3.3V
GND      -----> GND             GND
GPIO 21  -----> SDA
GPIO 22  -----> SCL
GPIO 34  <--------------------- OUTPUT
GPIO 32  <--------------------- LO+
GPIO 33  <--------------------- LO-
```

## License

MIT License - See parent repository for details
