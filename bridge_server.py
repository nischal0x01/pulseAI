#!/usr/bin/env python3
"""
WebSocket Bridge Server

This server bridges the ESP32 PPG data to the frontend with real-time BP predictions.
It receives filtered PPG from ESP32, runs predictions, and broadcasts to frontend clients.

Architecture:
    ESP32 → Serial → This Server → WebSocket → Frontend
                        ↓
                    BP Model
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional
import argparse
import sys
import os
from collections import deque
import threading
import queue

import serial
import numpy as np

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("❌ Missing dependencies. Please install them with:")
    print("   pip install fastapi uvicorn websockets")
    sys.exit(1)

try:
    import tensorflow as tf
    from model_builder_attention import create_phys_informed_model
    from feature_engineering import extract_physiological_features, standardize_feature_length, create_4_channel_input
except ImportError as e:
    print(f"⚠️  Model modules not loaded: {e}")
    print("   Predictions will be disabled")
    MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WINDOW_SIZE = 875
PPG_SAMPLE_RATE = 200
MODEL_SAMPLE_RATE = 125
DOWNSAMPLE_FACTOR = PPG_SAMPLE_RATE // MODEL_SAMPLE_RATE
BUFFER_SIZE = int(WINDOW_SIZE * (PPG_SAMPLE_RATE / MODEL_SAMPLE_RATE))
UPDATE_INTERVAL = 250

app = FastAPI(title="BP WebSocket Bridge", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BloodPressurePredictor:
    """Real-time blood pressure prediction."""
    
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.mean = 0.0
        self.std = 1.0
        
        if MODEL_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Load trained model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found: {self.model_path}")
            return False
        
        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={'huber_loss': tf.keras.losses.Huber(delta=1.0)}
            )
            logger.info("✓ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def downsample(self, signal):
        """Downsample signal."""
        return signal[::int(DOWNSAMPLE_FACTOR)]
    
    def simulate_ecg_from_ppg(self, ppg_signal):
        """Generate synthetic ECG from PPG."""
        ecg_sim = np.gradient(ppg_signal)
        ecg_sim = ecg_sim - np.mean(ecg_sim)
        if np.std(ecg_sim) > 0:
            ecg_sim = ecg_sim / np.std(ecg_sim) * 0.5
        return ecg_sim
    
    def preprocess_window(self, ppg_window):
        """Preprocess PPG window."""
        ppg_downsampled = self.downsample(ppg_window)
        
        if len(ppg_downsampled) < WINDOW_SIZE:
            padding = WINDOW_SIZE - len(ppg_downsampled)
            ppg_downsampled = np.pad(ppg_downsampled, (0, padding), mode='edge')
        elif len(ppg_downsampled) > WINDOW_SIZE:
            ppg_downsampled = ppg_downsampled[:WINDOW_SIZE]
        
        ecg_sim = self.simulate_ecg_from_ppg(ppg_downsampled)
        
        self.mean = 0.9 * self.mean + 0.1 * np.mean(ppg_downsampled)
        self.std = 0.9 * self.std + 0.1 * (np.std(ppg_downsampled) + 1e-6)
        
        ppg_norm = (ppg_downsampled - self.mean) / (self.std + 1e-6)
        ecg_norm = (ecg_sim - np.mean(ecg_sim)) / (np.std(ecg_sim) + 1e-6)
        
        signals = np.stack([ppg_norm, ecg_norm], axis=0)
        signals = signals[np.newaxis, :, :]
        
        try:
            ecg_for_feat = ecg_norm[np.newaxis, :]
            ppg_for_feat = ppg_norm[np.newaxis, :]
            
            pat_seqs, hr_seqs, _ = extract_physiological_features(ecg_for_feat, ppg_for_feat)
            pat_seqs = standardize_feature_length(pat_seqs, WINDOW_SIZE)
            hr_seqs = standardize_feature_length(hr_seqs, WINDOW_SIZE)
            
            input_4ch = create_4_channel_input(
                signals.transpose(0, 2, 1),
                pat_seqs,
                hr_seqs
            )
            return input_4ch
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            zeros = np.zeros((1, WINDOW_SIZE, 1))
            ppg_ch = ppg_norm.reshape(1, WINDOW_SIZE, 1)
            ecg_ch = ecg_norm.reshape(1, WINDOW_SIZE, 1)
            return np.concatenate([ecg_ch, ppg_ch, zeros, zeros], axis=2)
    
    def predict(self, ppg_window):
        """Predict BP."""
        if self.model is None:
            return None, None
        
        try:
            input_data = self.preprocess_window(ppg_window)
            predictions = self.model.predict(input_data, verbose=0)
            
            if predictions.shape[1] == 2:
                sbp, dbp = predictions[0]
            else:
                sbp = predictions[0, 0]
                dbp = sbp * 0.67
            
            return float(sbp), float(dbp)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, None


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Frontend connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Frontend disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        json_msg = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json_msg)
            except Exception as e:
                logger.warning(f"Failed to send: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)


class SerialReader:
    """Reads PPG data from ESP32 in background thread."""
    
    def __init__(self, port, baudrate, data_queue):
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self.running = False
        self.thread = None
    
    def start(self):
        """Start reading from serial port."""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("Serial reader started")
    
    def stop(self):
        """Stop reading."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _read_loop(self):
        """Main reading loop."""
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=1)
            logger.info(f"Connected to {self.port}")
            
            while self.running:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if not line or ',' not in line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 3:
                        continue
                    
                    timestamp, ir_raw, filtered = parts
                    data = {
                        'timestamp': timestamp,
                        'ir_raw': float(ir_raw),
                        'filtered': float(filtered)
                    }
                    
                    self.data_queue.put(data)
                    
                except (ValueError, UnicodeDecodeError):
                    continue
            
            ser.close()
            logger.info("Serial connection closed")
            
        except Exception as e:
            logger.error(f"Serial error: {e}")


# Global state
manager = ConnectionManager()
predictor = None
ppg_buffer = deque(maxlen=BUFFER_SIZE)
sample_count = 0
prediction_count = 0
data_queue = queue.Queue()
serial_reader = None


@app.websocket("/signals")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frontend."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_loop():
    """Process serial data and broadcast to clients."""
    global sample_count, prediction_count
    
    while True:
        try:
            # Get data from queue (non-blocking)
            try:
                data = data_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            
            filtered_value = data['filtered']
            ppg_buffer.append(filtered_value)
            sample_count += 1
            
            # Broadcast PPG signal
            await manager.broadcast({
                'type': 'signal_data',
                'payload': {
                    'timestamp': int(time.time() * 1000),
                    'ppg_value': filtered_value,
                    'ecg_value': 0.0,  # Placeholder
                    'sample_rate': PPG_SAMPLE_RATE,
                    'quality': 0.85
                }
            })
            
            # Make prediction
            if len(ppg_buffer) >= BUFFER_SIZE and sample_count % UPDATE_INTERVAL == 0:
                if predictor and predictor.model:
                    ppg_window = np.array(list(ppg_buffer))
                    sbp, dbp = predictor.predict(ppg_window)
                    
                    if sbp is not None and dbp is not None:
                        prediction_count += 1
                        
                        # Broadcast BP prediction
                        await manager.broadcast({
                            'type': 'bp_prediction',
                            'payload': {
                                'sbp': round(sbp, 1),
                                'dbp': round(dbp, 1),
                                'timestamp': int(time.time() * 1000),
                                'confidence': 0.85,
                                'prediction_count': prediction_count
                            }
                        })
                        
                        logger.info(f"[{prediction_count:3d}] BP: {sbp:6.1f}/{dbp:5.1f} mmHg")
        
        except Exception as e:
            logger.error(f"Broadcast loop error: {e}")
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global predictor, serial_reader
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.h5')
    predictor = BloodPressurePredictor(model_path)
    
    # Start serial reader
    port = os.getenv('SERIAL_PORT', '/dev/ttyUSB0')
    baudrate = int(os.getenv('SERIAL_BAUDRATE', '115200'))
    
    serial_reader = SerialReader(port, baudrate, data_queue)
    serial_reader.start()
    
    # Start broadcast loop
    asyncio.create_task(broadcast_loop())
    
    logger.info("=" * 60)
    logger.info("  WebSocket Bridge Server Started")
    logger.info("=" * 60)
    logger.info(f"Serial Port: {port}")
    logger.info(f"WebSocket: ws://localhost:8080/signals")
    logger.info(f"Model: {'Loaded' if predictor.model else 'Not loaded'}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if serial_reader:
        serial_reader.stop()
    logger.info("Server shutdown")


@app.get("/")
async def root():
    """Status page."""
    return {
        "status": "running",
        "model_loaded": predictor.model is not None if predictor else False,
        "connections": len(manager.active_connections),
        "samples": sample_count,
        "predictions": prediction_count
    }


def main():
    parser = argparse.ArgumentParser(description='WebSocket Bridge Server')
    parser.add_argument('--port', type=int, default=8080, help='WebSocket server port')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0', help='Serial port for ESP32')
    parser.add_argument('--serial-baudrate', type=int, default=115200, help='Serial baudrate')
    parser.add_argument('--model', default='checkpoints/best_model.h5', help='Model path')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['SERIAL_PORT'] = args.serial_port
    os.environ['SERIAL_BAUDRATE'] = str(args.serial_baudrate)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
