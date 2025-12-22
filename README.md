# Cuffless Blood Pressure Estimation

A machine learning project for estimating blood pressure from PPG (Photoplethysmography) and ECG (Electrocardiography) signals using deep learning techniques. This project implements CNN-based feature extraction combined with various regression models for non-invasive blood pressure prediction.

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
