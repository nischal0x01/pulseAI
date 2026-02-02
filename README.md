# Cuffless Blood Pressure Estimation

A machine learning project for estimating blood pressure from PPG (Photoplethysmography) and ECG (Electrocardiography) signals using deep learning techniques. This project implements CNN-LSTM architecture with attention mechanisms for non-invasive blood pressure prediction.

## 🎯 Quick Start - Real-time BP Monitoring

### Hardware Setup

**Required:**
- ESP32 development board
- MAX30102 PPG sensor (pulse oximeter)
- AD8232 ECG sensor module
- 3x disposable ECG electrodes

See **[HARDWARE_SETUP.md](HARDWARE_SETUP.md)** for complete wiring guide with diagrams.

### Software Setup

#### Option 1: Automated Script (Recommended)

```bash
# Start everything with one command
./bash/start-realtime-bp.sh

# Or with custom serial port
./bash/start-realtime-bp.sh /dev/ttyACM0
```

Then open `http://localhost:3000` in your browser.

### Option 2: Manual Setup

**Hardware Setup:**
1. **ESP32 + MAX30102** - See [esp32/README.md](esp32/README.md)
2. Upload the filtered PPG sketch to ESP32
3. Connect MAX30102 sensor and place finger on it

**Software:**
```bash
# 1. Start the bridge server (connects ESP32 → Model → Frontend)
python bridge_server.py --serial-port /dev/ttyUSB0

# 2. In another terminal, start frontend
cd frontend
pnpm dev

# 3. Open browser
open http://localhost:3000
```

See [REALTIME_SETUP.md](REALTIME_SETUP.md) for complete setup guide.

---

## 🚀 Development Setup

### Backend (WebSocket Server)

For development and testing, you can use the mock ESP32 server:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the mock ESP32 WebSocket server
python mock_esp32_server.py
```

The server will start on `http://localhost:8080` and provide:
- WebSocket endpoint at `ws://localhost:8080/signals`
- Web interface at `http://localhost:8080`
- Real-time PPG and ECG signal simulation

### Frontend (Next.js)

```bash
cd frontend
npm install
# or
pnpm install

# Start the development server
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`.

## 📡 WebSocket Communication

The system uses WebSocket connections to stream real-time physiological signals:

### Signal Data Format
```json
{
  "type": "signal_data",
  "payload": {
    "timestamp": 1640995200000,
    "ppg_value": 0.5,
    "ecg_value": 0.2,
    "sample_rate": 250,
    "quality": 0.8
  }
}
```

### Control Commands
```json
{
  "type": "control",
  "command": "start_acquisition" | "stop_acquisition" | "calibrate"
}
```

## 🔧 Troubleshooting

If you see WebSocket errors in the console:

1. **Check if the backend server is running**: Make sure `python mock_esp32_server.py` is running
2. **Verify the connection URL**: The frontend should connect to `ws://localhost:8080/signals`
3. **Check network connectivity**: Ensure no firewall is blocking the connection
4. **Review browser console**: Look for detailed error messages with troubleshooting tips

## 📁 Data Directories

The project uses the following data structure:

- `data/raw/` - Raw dataset files downloaded from PulseDB
- `data/processed/` - Preprocessed .mat files ready for training
- `checkpoints/` - Model checkpoints and saved weights
