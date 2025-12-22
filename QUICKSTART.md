# Quick Start Guide

## 🚀 Running the Development Environment

### Option 1: Use the automated script (Recommended)

```bash
# Make script executable (if not already done)
chmod +x start-dev.sh

# Start both backend and frontend
./start-dev.sh

# Or start individual services
./start-dev.sh backend   # Only WebSocket server
./start-dev.sh frontend  # Only Next.js frontend
```

### Option 2: Manual setup

#### 1. Start the Backend WebSocket Server

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the mock ESP32 server
python mock_esp32_server.py
```

The server will start on `http://localhost:8080`.

#### 2. Start the Frontend

```bash
cd frontend

# Install dependencies (first time only)
npm install
# or if using pnpm
pnpm install

# Start development server
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`.

## 🔧 WebSocket Connection

The frontend will automatically connect to `ws://localhost:8080/signals` when you:
1. Click the "Connect" button in the UI
2. The mock server provides realistic PPG and ECG signals

## ✅ Verifying the Setup

1. **Backend**: Visit `http://localhost:8080` - you should see the mock ESP32 status page
2. **Frontend**: Visit `http://localhost:3000` - you should see the blood pressure monitoring interface
3. **Connection**: Click "Connect" in the frontend - you should see signal data streaming

## 🐛 Troubleshooting

### WebSocket Connection Issues

If you see `[WebSocket error:` in the browser console:

1. **Check backend server**: Make sure `python mock_esp32_server.py` is running
2. **Check the URL**: Verify the frontend is connecting to `localhost:8080`
3. **Check ports**: Ensure port 8080 is not blocked or used by another service
4. **Restart services**: Stop and restart both backend and frontend

### Dependencies Issues

```bash
# Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Node.js dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Port Conflicts

If port 8080 or 3000 are in use:

- **Backend**: Edit `mock_esp32_server.py` and change the port in the `uvicorn.run()` call
- **Frontend**: Create a `.env.local` file in the frontend directory with `PORT=3001`

## 📊 Testing Signal Streaming

1. Start both servers
2. Open the frontend at `http://localhost:3000`
3. Click "Connect" to establish WebSocket connection
4. Click "Start Acquisition" to begin signal streaming
5. You should see real-time PPG and ECG waveforms

The mock server generates realistic physiological signals with:
- Heart rate around 72 BPM (varies naturally)
- Realistic PPG and ECG waveforms
- Signal noise and occasional motion artifacts
- Configurable signal quality

## 🎯 Next Steps

- Customize signal parameters in `mock_esp32_server.py`
- Implement additional signal processing features
- Connect to real ESP32 hardware by updating the IP address in the frontend
- Deploy the system for production use
