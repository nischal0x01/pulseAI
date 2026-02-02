# Real-time BP Estimation Setup Guide

## Quick Start

### 1. Start the Bridge Server

The bridge server connects ESP32 → Model → Frontend.

```bash
# Basic usage (auto-detects serial port and model)
python bridge_server.py

# With custom settings
python bridge_server.py \
  --port 8080 \
  --serial-port /dev/ttyUSB0 \
  --serial-baudrate 115200 \
  --model checkpoints/best_model.h5
```

Server will start on `ws://localhost:8080/signals`

### 2. Start the Frontend

```bash
cd frontend
pnpm dev
```

Frontend runs on `http://localhost:3000`

### 3. Connect and Monitor

1. Open `http://localhost:3000` in browser
2. Connection should auto-establish to `localhost:8080`
3. Place finger on MAX30102 sensor
4. Watch real-time BP predictions appear!

---

## Complete Architecture

```
┌──────────────┐
│   ESP32      │  Filters PPG (DC removal + bandpass)
│  MAX30102    │  Sends: timestamp,ir_raw,filtered_value
└──────┬───────┘
       │ Serial (USB)
       │ 115200 baud
       ▼
┌──────────────────────────────────────┐
│   bridge_server.py (Python)          │
│                                      │
│  • Reads filtered PPG from serial   │
│  • Buffers 7 seconds of data        │
│  • Extracts PAT, HR features        │
│  • Runs CNN-LSTM model              │
│  • Predicts SBP/DBP                 │
│  • Broadcasts via WebSocket         │
└──────┬───────────────────────────────┘
       │ WebSocket
       │ ws://localhost:8080/signals
       ▼
┌──────────────────────────────────────┐
│   Frontend (Next.js)                 │
│                                      │
│  • Receives PPG signal data         │
│  • Receives BP predictions          │
│  • Displays real-time charts        │
│  • Shows latest BP reading          │
└──────────────────────────────────────┘
```

---

## Message Format

### Signal Data (sent continuously)

```json
{
  "type": "signal_data",
  "payload": {
    "timestamp": 1738512345678,
    "ppg_value": 0.234,
    "ecg_value": 0.0,
    "sample_rate": 200,
    "quality": 0.85
  }
}
```

### BP Prediction (sent every ~2 seconds)

```json
{
  "type": "bp_prediction",
  "payload": {
    "sbp": 118.3,
    "dbp": 78.2,
    "timestamp": 1738512345678,
    "confidence": 0.85,
    "prediction_count": 42
  }
}
```

---

## Frontend Integration

### Display the BP Component

Add to your page (e.g., `frontend/src/app/page.tsx`):

```tsx
import { BloodPressureDisplay } from "@/components/blood-pressure-display"

export default function Home() {
  return (
    <div className="container mx-auto p-4">
      <BloodPressureDisplay />
    </div>
  )
}
```

### Access BP Data in Any Component

```tsx
"use client"

import { useSignal } from "@/lib/signal-context"

export function MyComponent() {
  const { latestBP, connectionStatus } = useSignal()
  
  if (latestBP) {
    console.log(`BP: ${latestBP.sbp}/${latestBP.dbp} mmHg`)
  }
  
  return <div>SBP: {latestBP?.sbp}</div>
}
```

---

## Troubleshooting

### Bridge Server Won't Start

**Error: "Model not found"**
```bash
# Train the model first
python src/models/model.py
```

**Error: "Serial port not found"**
```bash
# List available ports
ls /dev/ttyUSB* /dev/ttyACM*

# Use correct port
python bridge_server.py --serial-port /dev/ttyACM0
```

**Error: "Permission denied"**
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER
# Log out and log back in
```

### Frontend Not Receiving Data

**Check server is running:**
```bash
curl http://localhost:8080
# Should return: {"status":"running","model_loaded":true,...}
```

**Check WebSocket connection in browser console:**
- Should see: "WebSocket connected to ESP32"
- If error, verify server is on `localhost:8080`

**Check frontend WebSocket URL:**
In `frontend/src/lib/signal-context.tsx`, default should be:
```tsx
const [esp32Ip, setEsp32Ip] = useState("localhost:8080")
```

### No BP Predictions

**Check logs in bridge server:**
- Should see: "Model loaded successfully"
- Should see: "Samples collected: ..."
- Should see: "[1] BP: 120.0/80.0 mmHg" periodically

**Check buffer is filling:**
- Needs 7 seconds of data before first prediction
- Watch for "Collecting data..." message

**Check signal quality:**
- Ensure good finger contact on sensor
- Reduce motion artifacts
- Check PPG waveform is clean

### Performance Issues

**Slow predictions:**
- Check CPU usage (should be <50%)
- Consider using GPU for inference
- Reduce UPDATE_INTERVAL in bridge_server.py

**Laggy frontend:**
- Reduce chart update rate
- Limit buffer size
- Close other browser tabs

---

## Configuration Options

### Bridge Server (`bridge_server.py`)

Edit these constants near the top:

```python
WINDOW_SIZE = 875           # Samples for model (don't change)
PPG_SAMPLE_RATE = 200       # ESP32 sampling rate
MODEL_SAMPLE_RATE = 125     # Model training rate
UPDATE_INTERVAL = 250       # Predict every N samples (~2 sec)
```

### ESP32 Filtering

See `esp32/ppg_reader_filtered/ppg_reader_filtered.ino`:

```cpp
const float DC_ALPHA = 0.01;        // DC removal smoothing
int sampleRate = 200;                // Sample rate (Hz)
byte ledBrightness = 60;             // LED power (0-255)
```

---

## Development vs Production

### Development Setup (Mock Data)

```bash
# Use mock server instead of ESP32
python mock_server/mock_esp32_server.py

# Frontend connects to mock
cd frontend
pnpm dev
```

### Production Setup (Real ESP32)

```bash
# 1. Upload sketch to ESP32
# See esp32/README.md

# 2. Start bridge server
python bridge_server.py --serial-port /dev/ttyUSB0

# 3. Start frontend
cd frontend
pnpm dev
```

---

## Monitoring & Logging

### Check Server Status

```bash
curl http://localhost:8080
```

Returns:
```json
{
  "status": "running",
  "model_loaded": true,
  "connections": 1,
  "samples": 5000,
  "predictions": 20
}
```

### View Server Logs

Bridge server logs to console:
```
INFO - WebSocket Bridge Server Started
INFO - Serial Port: /dev/ttyUSB0
INFO - WebSocket: ws://localhost:8080/signals
INFO - Model: Loaded
INFO - Frontend connected. Total: 1
INFO - [1] BP: 118.3/78.2 mmHg
INFO - [2] BP: 119.1/78.8 mmHg
```

### Browser Console

Open DevTools (F12) → Console:
```
[v0] WebSocket connected to ESP32
[v0] BP Prediction received: {sbp: 118.3, dbp: 78.2, ...}
BP Prediction received in context: {sbp: 118.3, dbp: 78.2, ...}
```

---

## Files Created

- **`bridge_server.py`** - WebSocket bridge server
- **`frontend/src/components/blood-pressure-display.tsx`** - BP display component
- **`frontend/src/lib/websocket-manager.ts`** - Updated with BP prediction handling
- **`frontend/src/lib/signal-context.tsx`** - Updated with latestBP state
- **`REALTIME_SETUP.md`** - This file

---

## Next Steps

1. ✅ **Test the setup** - Verify predictions work
2. 🔧 **Add real ECG sensor** - AD8232 module (future)
3. 📊 **Add BP history chart** - Track predictions over time
4. 💾 **Add data export** - Save predictions to CSV
5. 🎯 **Calibration UI** - Personal calibration interface
6. 📱 **Mobile optimization** - Responsive design
7. 🔔 **Alerts** - Threshold warnings for abnormal BP

---

## Support

For issues:
1. Check this guide
2. Review server and browser logs
3. Verify hardware connections
4. Check model is trained
