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
from scipy.signal import butter, lfilter, lfilter_zi

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
UPDATE_INTERVAL = 100  # Predict every 100 samples (0.5 seconds at 200Hz)

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
        # Separate normalization statistics for PPG and ECG
        self.ppg_mean = 0.0
        self.ppg_std = 1.0
        self.ecg_mean = 0.0
        self.ecg_std = 1.0
        
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
                custom_objects={'huber_loss': tf.keras.losses.Huber(delta=1.0)},
                safe_mode=False  # Allow loading Lambda layers
            )
            logger.info("✓ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def calculate_signal_quality(self, ppg_buffer, ecg_buffer):
        """
        Calculate signal quality based on both PPG and ECG signals.
        Returns a quality score between 0.0 and 1.0.
        """
        if len(ppg_buffer) < 100 or len(ecg_buffer) < 100:
            return 0.0
        
        ppg_array = np.array(list(ppg_buffer)[-100:])  # Last 100 samples
        ecg_array = np.array(list(ecg_buffer)[-100:])
        
        # Check PPG quality
        ppg_std = np.std(ppg_array)
        ppg_valid = ppg_std > 0.01 and not np.any(np.isnan(ppg_array)) and not np.any(np.isinf(ppg_array))
        
        # Check ECG quality
        ecg_std = np.std(ecg_array)
        ecg_valid = ecg_std > 0.01 and not np.any(np.isnan(ecg_array)) and not np.any(np.isinf(ecg_array))
        
        # Calculate quality score (0.0 to 1.0)
        quality_score = 0.0
        
        if ppg_valid and ecg_valid:
            # Both signals good
            ppg_quality = min(1.0, ppg_std / 0.5)  # Normalize std to 0-1 range
            ecg_quality = min(1.0, ecg_std / 0.5)
            quality_score = (ppg_quality + ecg_quality) / 2.0
        elif ppg_valid:
            # Only PPG good
            quality_score = min(0.5, ppg_std / 1.0)  # Max 50% if only PPG valid
        elif ecg_valid:
            # Only ECG good
            quality_score = min(0.5, ecg_std / 1.0)  # Max 50% if only ECG valid
        
        # Ensure reasonable range
        return max(0.0, min(1.0, quality_score))
    
    def downsample(self, signal):
        """Downsample signal."""
        return signal[::int(DOWNSAMPLE_FACTOR)]
    
    def preprocess_window(self, ppg_window, ecg_window, hr_data=None):
        """
        Preprocess PPG and ECG windows for model input.
        Returns 4-channel input: [ECG, PPG, PAT, HR]
        
        Args:
            ppg_window: PPG signal data
            ecg_window: ECG signal data
            hr_data: Optional heart rate data from sensor (array of HR values).
                    If provided, will use this instead of extracting HR from ECG.
        """
        # Downsample to model's sample rate
        ppg_downsampled = self.downsample(ppg_window)
        ecg_downsampled = self.downsample(ecg_window)
        
        # Ensure correct length
        if len(ppg_downsampled) < WINDOW_SIZE:
            padding = WINDOW_SIZE - len(ppg_downsampled)
            ppg_downsampled = np.pad(ppg_downsampled, (0, padding), mode='edge')
        elif len(ppg_downsampled) > WINDOW_SIZE:
            ppg_downsampled = ppg_downsampled[:WINDOW_SIZE]
        
        if len(ecg_downsampled) < WINDOW_SIZE:
            padding = WINDOW_SIZE - len(ecg_downsampled)
            ecg_downsampled = np.pad(ecg_downsampled, (0, padding), mode='edge')
        elif len(ecg_downsampled) > WINDOW_SIZE:
            ecg_downsampled = ecg_downsampled[:WINDOW_SIZE]
        
        # Update running stats for normalization (separate for PPG and ECG)
        self.ppg_mean = 0.9 * self.ppg_mean + 0.1 * np.mean(ppg_downsampled)
        self.ppg_std = 0.9 * self.ppg_std + 0.1 * (np.std(ppg_downsampled) + 1e-6)
        
        self.ecg_mean = 0.9 * self.ecg_mean + 0.1 * np.mean(ecg_downsampled)
        self.ecg_std = 0.9 * self.ecg_std + 0.1 * (np.std(ecg_downsampled) + 1e-6)
        
        # Normalize signals with their respective running statistics
        ppg_norm = (ppg_downsampled - self.ppg_mean) / (self.ppg_std + 1e-6)
        ecg_norm = (ecg_downsampled - self.ecg_mean) / (self.ecg_std + 1e-6)
        
        signals = np.stack([ppg_norm, ecg_norm], axis=0)
        signals = signals[np.newaxis, :, :]
        
        try:
            ecg_for_feat = ecg_norm[np.newaxis, :]
            ppg_for_feat = ppg_norm[np.newaxis, :]
            
            # Prepare HR from sensor if provided
            hr_from_sensor = None
            if hr_data is not None:
                # Downsample and normalize HR data
                hr_downsampled = self.downsample(hr_data)
                if len(hr_downsampled) < WINDOW_SIZE:
                    padding = WINDOW_SIZE - len(hr_downsampled)
                    hr_downsampled = np.pad(hr_downsampled, (0, padding), mode='edge')
                elif len(hr_downsampled) > WINDOW_SIZE:
                    hr_downsampled = hr_downsampled[:WINDOW_SIZE]
                hr_from_sensor = hr_downsampled[np.newaxis, :]  # Shape: (1, WINDOW_SIZE)
            
            # Extract features, using sensor HR if available
            pat_seqs, hr_seqs, _ = extract_physiological_features(
                ecg_for_feat, ppg_for_feat, hr_from_sensor=hr_from_sensor
            )
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
    
    def predict(self, ppg_window, ecg_window, hr_data=None):
        """
        Predict BP from PPG and ECG windows.
        
        Args:
            ppg_window: PPG signal data
            ecg_window: ECG signal data
            hr_data: Optional heart rate data from sensor
            
        Returns: (sbp, dbp) tuple
        """
        if self.model is None:
            return None, None
        
        try:
            input_data = self.preprocess_window(ppg_window, ecg_window, hr_data)
            predictions = self.model.predict(input_data, verbose=0)
            
            # Convert predictions to numpy array if it's a list
            if isinstance(predictions, list):
                predictions = np.array(predictions[0]) if len(predictions) == 1 else np.array(predictions)
            
            # Handle different prediction formats
            if len(predictions.shape) == 1:
                # Single output or flattened
                if predictions.shape[0] >= 2:
                    sbp, dbp = float(predictions[0]), float(predictions[1])
                else:
                    sbp = float(predictions[0])
                    dbp = sbp * 0.67
            elif predictions.shape[1] == 2:
                sbp, dbp = float(predictions[0, 0]), float(predictions[0, 1])
            else:
                sbp = float(predictions[0, 0])
                dbp = sbp * 0.67
            
            return sbp, dbp
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


class SignalFilter:
    """Applies bandpass filtering to PPG or ECG signals using scipy.signal."""
    
    def __init__(self, signal_type='ppg'):
        """
        Initialize filter.
        
        Args:
            signal_type: 'ppg' or 'ecg'
        """
        self.signal_type = signal_type
        self.dc_est = 0.0
        self.DC_ALPHA = 0.01
        self.initialized = False
        
        # Design filter coefficients
        if signal_type == 'ppg':
            # PPG: 0.7-4 Hz bandpass at 200 Hz sampling
            self.b, self.a = butter(N=3, Wn=[0.7, 4], btype='band', fs=200)
        else:  # ecg
            # ECG: 0.5-40 Hz bandpass at 200 Hz sampling
            self.b, self.a = butter(N=3, Wn=[0.5, 40], btype='band', fs=200)
        
        self.zi = None
    
    def filter(self, raw_value):
        """
        Apply filtering to raw signal value.
        
        Args:
            raw_value: Raw sensor reading
            
        Returns:
            Filtered value
        """
        if self.signal_type == 'ppg':
            # PPG: Apply DC removal first
            if not self.initialized:
                self.dc_est = raw_value
                self.initialized = True
            
            self.dc_est += self.DC_ALPHA * (raw_value - self.dc_est)
            ac_component = raw_value - self.dc_est
            input_val = ac_component
        else:
            # ECG: Normalize to -1 to 1 range (12-bit ADC: 0-4095)
            input_val = (raw_value - 2048.0) / 2048.0
        
        # Initialize filter state on first call
        if self.zi is None:
            self.zi = lfilter_zi(self.b, self.a) * input_val
        
        # Apply bandpass filter
        filtered_sample, self.zi = lfilter(self.b, self.a, [input_val], zi=self.zi)
        
        return filtered_sample[0]


class SerialReader:
    """Reads PPG and ECG data from ESP32 and applies filtering."""
    
    def __init__(self, port, baudrate, data_queue):
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self.running = False
        self.thread = None
        
        # Initialize filters
        self.ppg_filter = SignalFilter('ppg')
        self.ecg_filter = SignalFilter('ecg')
    
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
                    
                    # Log first few lines for debugging
                    if hasattr(self, '_line_count'):
                        self._line_count += 1
                    else:
                        self._line_count = 1
                    
                    if self._line_count <= 5:
                        logger.info(f"Serial data received: {line}")
                    
                    parts = line.split(',')
                    if len(parts) not in [3, 4]:  # timestamp,ppg_raw,ecg_raw or timestamp,ppg_raw,ecg_raw,hr
                        if self._line_count <= 5:
                            logger.warning(f"Expected 3-4 fields, got {len(parts)}: {line}")
                        continue
                    
                    timestamp = parts[0]
                    ppg_raw = float(parts[1])
                    ecg_raw = float(parts[2])
                    heart_rate = float(parts[3]) if len(parts) == 4 else 0.0
                    
                    # Apply filtering
                    ppg_filtered = self.ppg_filter.filter(ppg_raw)
                    ecg_filtered = self.ecg_filter.filter(ecg_raw) if ecg_raw > 0 else 0.0
                    
                    data = {
                        'timestamp': timestamp,
                        'ppg_raw': ppg_raw,
                        'ppg_filtered': ppg_filtered,
                        'ecg_raw': ecg_raw,
                        'ecg_filtered': ecg_filtered,
                        'heart_rate': heart_rate
                    }
                    
                    self.data_queue.put(data)
                    
                except (ValueError, UnicodeDecodeError):
                    continue
            
            ser.close()
            logger.info("Serial connection closed")
            
        except serial.SerialException as e:
            if "Permission denied" in str(e):
                logger.error(f"Permission denied: {self.port}")
                logger.error("=" * 60)
                logger.error("FIX: Add your user to the serial port group")
                logger.error("")
                logger.error("For Arch Linux / Manjaro:")
                logger.error(f"  sudo usermod -a -G uucp $USER")
                logger.error("")
                logger.error("For Debian / Ubuntu:")
                logger.error(f"  sudo usermod -a -G dialout $USER")
                logger.error("")
                logger.error("Alternative (temporary, current session only):")
                logger.error(f"  sudo chmod 666 {self.port}")
                logger.error("")
                logger.error("After adding to group, log out and log back in!")
                logger.error("=" * 60)
            else:
                logger.error(f"Serial error: {e}")
        except Exception as e:
            logger.error(f"Serial error: {e}")


# Global state
manager = ConnectionManager()
predictor = None
ppg_buffer = deque(maxlen=BUFFER_SIZE)
ecg_buffer = deque(maxlen=BUFFER_SIZE)
hr_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer for sensor-provided heart rate
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
    
    logger.info("Broadcast loop started, waiting for serial data...")
    broadcast_count = 0
    
    while True:
        try:
            # Get data from queue (non-blocking)
            try:
                data = data_queue.get_nowait()
                broadcast_count += 1
                
                if broadcast_count <= 5:
                    logger.info(f"Broadcasting data #{broadcast_count}: PPG={data['ppg_filtered']:.3f}, ECG={data['ecg_filtered']:.3f}")
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            
            ppg_filtered = data['ppg_filtered']
            ecg_filtered = data['ecg_filtered']
            heart_rate = data.get('heart_rate', 0)
            
            ppg_buffer.append(ppg_filtered)
            ecg_buffer.append(ecg_filtered)
            hr_buffer.append(heart_rate)
            sample_count += 1
            
            # Calculate signal quality based on both PPG and ECG
            signal_quality = predictor.calculate_signal_quality(ppg_buffer, ecg_buffer) if predictor else 0.0
            
            # Broadcast PPG and ECG signals
            await manager.broadcast({
                'type': 'signal_data',
                'payload': {
                    'timestamp': int(time.time() * 1000),
                    'ppg_value': ppg_filtered,
                    'ecg_value': ecg_filtered,
                    'sample_rate': PPG_SAMPLE_RATE,
                    'quality': signal_quality,
                    'heart_rate': heart_rate
                }
            })
            
            # Make prediction
            if len(ppg_buffer) >= BUFFER_SIZE and len(ecg_buffer) >= BUFFER_SIZE and sample_count % UPDATE_INTERVAL == 0:
                if predictor and predictor.model:
                    # Check signal quality threshold - require at least 60% quality
                    if signal_quality < 0.6:
                        logger.warning(f"Signal quality too low ({signal_quality:.1%}) - skipping prediction. Connect both PPG and ECG sensors.")
                        # Broadcast error message
                        error_message = {
                            'type': 'bp_prediction',
                            'payload': {
                                'sbp': None,
                                'dbp': None,
                                'timestamp': int(time.time() * 1000),
                                'confidence': signal_quality,
                                'error': 'Low signal quality - check sensor connections',
                                'prediction_count': prediction_count
                            }
                        }
                        await manager.broadcast(error_message)
                    else:
                        ppg_window = np.array(list(ppg_buffer))
                        ecg_window = np.array(list(ecg_buffer))
                        
                        # Use sensor HR if available and non-zero
                        hr_window = None
                        if len(hr_buffer) >= BUFFER_SIZE:
                            hr_array = np.array(list(hr_buffer))
                            # Only use sensor HR if values are reasonable (non-zero)
                            if np.mean(hr_array) > 0:
                                hr_window = hr_array
                        
                        sbp, dbp = predictor.predict(ppg_window, ecg_window, hr_window)
                        
                        if sbp is not None and dbp is not None:
                            prediction_count += 1
                            
                            # Broadcast BP prediction
                            bp_message = {
                                'type': 'bp_prediction',
                                'payload': {
                                    'sbp': round(sbp, 1),
                                    'dbp': round(dbp, 1),
                                    'timestamp': int(time.time() * 1000),
                                    'confidence': signal_quality,
                                    'prediction_count': prediction_count
                                }
                            }
                            await manager.broadcast(bp_message)
                            
                            logger.info(f"[{prediction_count:3d}] BP: {sbp:6.1f}/{dbp:5.1f} mmHg (quality: {signal_quality:.1%}, sent to {len(manager.active_connections)} clients)")
        
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
        "predictions": prediction_count,
        "queue_size": data_queue.qsize(),
        "serial_reader_alive": serial_reader.is_alive() if serial_reader else False
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
