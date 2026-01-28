#!/usr/bin/env python3
"""
Mock ESP32 WebSocket Server for Development

This server simulates an ESP32 device for testing the cuffless blood pressure frontend.
It provides realistic PPG and ECG signal data over WebSocket connections.

Usage:
    python mock_esp32_server.py

The server will start on http://localhost:8080 and provide:
- WebSocket endpoint at ws://localhost:8080/signals
- Static web interface at http://localhost:8080
- Real-time PPG and ECG signal simulation
"""

import asyncio
import json
import time
import math
import random
from typing import Dict, List
import logging
from pathlib import Path

# Check if FastAPI and uvicorn are available
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("❌ Missing dependencies. Please install them with:")
    print("   pip install fastapi uvicorn websockets")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock ESP32 Blood Pressure Monitor", version="1.0.0")

class SignalGenerator:
    """Generates realistic PPG and ECG signals for testing."""
    
    def __init__(self):
        self.sample_rate = 250  # Hz
        self.time = 0.0
        self.heart_rate = 72  # BPM
        self.breath_rate = 16  # Breaths per minute
        self.noise_level = 0.02
        
        # Signal quality factors
        self.signal_quality = 0.8
        self.motion_artifact = False
        
    def get_ppg_sample(self) -> float:
        """Generate a realistic PPG signal sample."""
        # Primary cardiac component
        cardiac_freq = self.heart_rate / 60.0
        cardiac_signal = math.sin(2 * math.pi * cardiac_freq * self.time)
        
        # Add harmonics for realistic PPG shape
        harmonic2 = 0.3 * math.sin(2 * math.pi * cardiac_freq * self.time * 2 + math.pi/4)
        harmonic3 = 0.1 * math.sin(2 * math.pi * cardiac_freq * self.time * 3 + math.pi/2)
        
        # Respiratory component
        resp_freq = self.breath_rate / 60.0
        respiratory = 0.1 * math.sin(2 * math.pi * resp_freq * self.time)
        
        # Combine signals
        signal = cardiac_signal + harmonic2 + harmonic3 + respiratory
        
        # Add realistic noise
        noise = random.gauss(0, self.noise_level)
        
        # Add motion artifacts occasionally
        if self.motion_artifact:
            noise += 0.5 * random.gauss(0, 0.1)
        
        return signal + noise
    
    def get_ecg_sample(self) -> float:
        """Generate a realistic ECG signal sample."""
        # Basic ECG waveform (simplified QRS complex)
        cardiac_freq = self.heart_rate / 60.0
        t_cycle = (self.time * cardiac_freq) % 1.0
        
        # QRS complex approximation
        if 0.1 < t_cycle < 0.2:
            # Q wave
            ecg_signal = -0.2 * math.sin(math.pi * (t_cycle - 0.1) / 0.1)
        elif 0.2 <= t_cycle < 0.25:
            # R wave
            ecg_signal = 2.0 * math.sin(math.pi * (t_cycle - 0.2) / 0.05)
        elif 0.25 <= t_cycle < 0.3:
            # S wave
            ecg_signal = -0.3 * math.sin(math.pi * (t_cycle - 0.25) / 0.05)
        elif 0.35 <= t_cycle < 0.5:
            # T wave
            ecg_signal = 0.3 * math.sin(math.pi * (t_cycle - 0.35) / 0.15)
        else:
            # Baseline
            ecg_signal = 0.0
        
        # Add noise
        noise = random.gauss(0, self.noise_level * 0.5)
        
        return ecg_signal + noise
    
    def update_time(self):
        """Update the time for next sample."""
        self.time += 1.0 / self.sample_rate
        
        # Occasionally change heart rate slightly
        if random.random() < 0.001:  # 0.1% chance per sample
            self.heart_rate += random.gauss(0, 2)
            self.heart_rate = max(50, min(120, self.heart_rate))  # Keep realistic range
        
        # Occasionally introduce motion artifacts
        if random.random() < 0.005:  # 0.5% chance
            self.motion_artifact = not self.motion_artifact
    
    def get_signal_quality(self) -> float:
        """Get current signal quality (0-1)."""
        base_quality = 0.85
        if self.motion_artifact:
            return base_quality * 0.6
        return base_quality + 0.1 * random.gauss(0, 0.1)

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.is_acquiring = False
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast_signal_data(self, data: dict):
        """Send signal data to all connected clients."""
        if not self.active_connections:
            return
            
        message = json.dumps(data)
        disconnected = []
        
        # Debug: Log message type being sent
        logger.debug(f"Broadcasting {data.get('type', 'unknown')} message to {len(self.active_connections)} clients")
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Global instances
manager = ConnectionManager()
signal_gen = SignalGenerator()

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve a simple status page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mock ESP32 Blood Pressure Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; margin: 10px 0; }
            .info { color: #666; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫀 Mock ESP32 Blood Pressure Monitor</h1>
            
            <div class="status">
                ✅ Server is running and ready to simulate physiological signals
            </div>
            
            <h3>WebSocket Endpoint</h3>
            <div class="endpoint">ws://localhost:8080/signals</div>
            
            <h3>Signal Information</h3>
            <div class="info">
                • <strong>PPG Signal:</strong> Photoplethysmography simulation<br>
                • <strong>ECG Signal:</strong> Electrocardiography simulation<br>
                • <strong>Sample Rate:</strong> 250 Hz<br>
                • <strong>Heart Rate:</strong> ~72 BPM (varies realistically)<br>
                • <strong>Signal Quality:</strong> Includes realistic noise and artifacts
            </div>
            
            <h3>Supported Commands</h3>
            <div class="info">
                • <strong>start_acquisition:</strong> Begin streaming signals<br>
                • <strong>stop_acquisition:</strong> Stop streaming signals<br>
                • <strong>calibrate:</strong> Simulate calibration process
            </div>
            
            <h3>Usage</h3>
            <div class="info">
                Connect your frontend application to the WebSocket endpoint above.<br>
                The server will simulate realistic physiological signals for development and testing.
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.websocket("/signals")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for signal streaming."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Listen for commands from client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                command = json.loads(data)
                await handle_command(command, websocket)
            except asyncio.TimeoutError:
                pass  # No command received, continue
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received from client")
                continue
            
            # Send signal data if acquiring
            if manager.is_acquiring:
                await send_signal_data()
            
            # Small delay to control sample rate
            await asyncio.sleep(1.0 / signal_gen.sample_rate)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def handle_command(command: dict, websocket: WebSocket):
    """Handle control commands from client."""
    cmd_type = command.get("type")
    cmd_command = command.get("command")
    
    if cmd_type == "control":
        if cmd_command == "start_acquisition":
            manager.is_acquiring = True
            logger.info("Signal acquisition started")
            await websocket.send_text(json.dumps({
                "type": "status",
                "message": "Acquisition started",
                "timestamp": time.time()
            }))
            
        elif cmd_command == "stop_acquisition":
            manager.is_acquiring = False
            logger.info("Signal acquisition stopped")
            await websocket.send_text(json.dumps({
                "type": "status", 
                "message": "Acquisition stopped",
                "timestamp": time.time()
            }))
            
        elif cmd_command == "calibrate":
            logger.info("Calibration requested")
            await websocket.send_text(json.dumps({
                "type": "status",
                "message": "Calibration completed",
                "timestamp": time.time()
            }))

async def send_signal_data():
    """Generate and send signal data to all connected clients."""
    # Generate samples
    ppg_value = signal_gen.get_ppg_sample()
    ecg_value = signal_gen.get_ecg_sample()
    quality = signal_gen.get_signal_quality()
    
    # Update time for next sample
    signal_gen.update_time()
    
    # Create signal packet
    packet = {
        "type": "signal_data",
        "payload": {
            "timestamp": time.time() * 1000,  # Convert to milliseconds
            "ppg_value": ppg_value,
            "ecg_value": ecg_value,
            "sample_rate": signal_gen.sample_rate,
            "quality": quality
        }
    }
    
    # Broadcast to all connected clients
    await manager.broadcast_signal_data(packet)

if __name__ == "__main__":
    print("🫀 Starting Mock ESP32 Blood Pressure Monitor Server...")
    print("   📡 WebSocket endpoint: ws://localhost:8080/signals")
    print("   🌐 Web interface: http://localhost:8080")
    print("   🛑 Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8080, 
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
