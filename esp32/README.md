# ESP32 PPG Reader with On-Device Filtering

This directory contains the ESP32 Arduino sketch for reading PPG data from a MAX30102 sensor with on-device signal processing.

## Features

- **DC Component Removal**: Removes the DC offset using an exponential moving average filter
- **Bandpass Filtering**: 3rd order Butterworth bandpass filter (0.7-4 Hz) for PPG signal extraction
- **Real-time Processing**: All filtering happens on the ESP32, reducing computational load on the host
- **High Sample Rate**: 200 Hz sampling for accurate PPG capture

## Hardware Requirements

- ESP32 development board
- MAX30102 pulse oximeter sensor
- Connection:
  - SDA → GPIO 21 (default I2C SDA on ESP32)
  - SCL → GPIO 22 (default I2C SCL on ESP32)
  - VIN → 3.3V
  - GND → GND

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

```cpp
const float DC_ALPHA = 0.01;  // DC removal smoothing factor (0.001-0.1)
```

The bandpass filter coefficients are pre-calculated for 0.7-4 Hz at 200 Hz sampling rate. To change these:

1. Use Python with scipy:
   ```python
   from scipy.signal import butter
   b, a = butter(N=3, Wn=[0.7, 4], btype='band', fs=200)
   print("b_coeff:", b)
   print("a_coeff:", a)
   ```

2. Update the `b_coeff` and `a_coeff` arrays in the sketch

## Output Format

The ESP32 outputs data via serial (115200 baud) in CSV format:

```
timestamp,ir_raw,filtered_value
```

- `timestamp`: Milliseconds since ESP32 boot
- `ir_raw`: Raw IR sensor reading (ADC value)
- `filtered_value`: Filtered PPG signal value

## Usage with Python

Use the companion Python script to receive and visualize the data:

```bash
cd ../scripts
python ppg_reader.py --port /dev/ttyUSB0
```

## Troubleshooting

### Sensor Not Found
- Check I2C connections (SDA, SCL)
- Verify power supply (3.3V)
- Try different I2C pins if necessary

### No Data or Garbled Output
- Check baud rate matches (115200)
- Ensure correct serial port selected
- Try resetting the ESP32

### Poor Signal Quality
- Adjust `ledBrightness` (typically 30-100)
- Ensure good sensor contact with finger
- Reduce motion artifacts
- Try different `sampleAverage` values

### Filter Not Working
- Verify filter coefficients are correct
- Check that `SAMPLE_RATE` matches sensor configuration
- Ensure DC component removal is working (check raw values)

## Technical Details

### DC Component Removal

Uses exponential moving average (EMA):
```
dc_estimate = dc_estimate + α × (ir - dc_estimate)
ac_component = ir - dc_estimate
```

### Bandpass Filter

Implements a 6th-order IIR filter (cascaded biquads represented as single Direct Form II):
- Passband: 0.7-4 Hz (typical heart rate range)
- Filter type: Butterworth (flat passband response)
- Order: 3 (results in 6th order due to bandpass transformation)

## Performance

- Memory usage: ~1KB RAM for filter buffers
- CPU usage: <5% at 200 Hz sampling rate
- Latency: <5ms per sample

## License

MIT License - See parent repository for details
