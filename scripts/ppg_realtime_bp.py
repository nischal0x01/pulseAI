#!/usr/bin/env python3
"""
Real-time Blood Pressure Estimation from ESP32 PPG Data

This script:
1. Receives pre-filtered PPG data from ESP32
2. Buffers data into windows (7 seconds @ 125 Hz = 875 samples)
3. Extracts physiological features (ECG simulated, PAT, HR)
4. Feeds to trained model for SBP/DBP prediction
5. Displays real-time predictions

Usage:
    python ppg_realtime_bp.py --port /dev/ttyUSB0
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
import csv
import time
import argparse
import sys
import os

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

try:
    import tensorflow as tf
    from model_builder_attention import create_phys_informed_model
    from feature_engineering import extract_physiological_features, standardize_feature_length, create_4_channel_input
    from preprocessing import preprocess_signals
    from utils import normalize_data
    print("✓ Model modules loaded")
except ImportError as e:
    print(f"✗ Error importing model modules: {e}")
    print("  Please ensure you're in the project root and dependencies are installed")
    sys.exit(1)

# Constants
WINDOW_SIZE = 875  # 7 seconds @ 125 Hz (model expects this)
PPG_SAMPLE_RATE = 200  # ESP32 sampling rate
MODEL_SAMPLE_RATE = 125  # Model training sample rate
DOWNSAMPLE_FACTOR = PPG_SAMPLE_RATE // MODEL_SAMPLE_RATE  # 200/125 ≈ 1.6
BUFFER_SIZE = int(WINDOW_SIZE * (PPG_SAMPLE_RATE / MODEL_SAMPLE_RATE))  # ~1400 samples
UPDATE_INTERVAL = 250  # Update prediction every N samples (~2 seconds)

class BloodPressurePredictor:
    """Real-time blood pressure prediction from PPG signals."""
    
    def __init__(self, model_path='/home/arwin/codes/python/cuffless-blood-pressure/checkpoints/best_model.h5'):
        """Initialize predictor with trained model."""
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            print("  Please train the model first using: python src/models/model.py")
            sys.exit(1)
        
        try:
            # Load model
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'huber_loss': tf.keras.losses.Huber(delta=1.0)}
            )
            print("✓ Model loaded successfully")
            
            # Running stats for normalization (will be updated with real data)
            self.mean = 0.0
            self.std = 1.0
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def downsample(self, signal, factor=DOWNSAMPLE_FACTOR):
        """Downsample signal to match model's expected sample rate."""
        # Simple decimation
        return signal[::int(factor)]
    
    def simulate_ecg_from_ppg(self, ppg_signal):
        """
        Generate synthetic ECG-like signal from PPG.
        In production, use actual ECG sensor.
        """
        # Simple derivative approximation (not accurate, but allows demo)
        ecg_sim = np.gradient(ppg_signal)
        ecg_sim = ecg_sim - np.mean(ecg_sim)
        # Scale to typical ECG range
        if np.std(ecg_sim) > 0:
            ecg_sim = ecg_sim / np.std(ecg_sim) * 0.5
        return ecg_sim
    
    def preprocess_window(self, ppg_window, hr_data=None):
        """
        Preprocess PPG window for model input.
        Returns 4-channel input: [ECG, PPG, PAT, HR]
        
        Args:
            ppg_window: PPG signal data
            hr_data: Optional heart rate data from sensor (array of HR values).
                    If provided, will use this instead of extracting HR from ECG.
        """
        # Downsample to model's sample rate
        ppg_downsampled = self.downsample(ppg_window)
        
        # Ensure correct length
        if len(ppg_downsampled) < WINDOW_SIZE:
            # Pad with zeros if too short
            padding = WINDOW_SIZE - len(ppg_downsampled)
            ppg_downsampled = np.pad(ppg_downsampled, (0, padding), mode='edge')
        elif len(ppg_downsampled) > WINDOW_SIZE:
            # Truncate if too long
            ppg_downsampled = ppg_downsampled[:WINDOW_SIZE]
        
        # Generate simulated ECG (in production, use real ECG)
        ecg_sim = self.simulate_ecg_from_ppg(ppg_downsampled)
        
        # Update running stats for normalization
        self.mean = 0.9 * self.mean + 0.1 * np.mean(ppg_downsampled)
        self.std = 0.9 * self.std + 0.1 * (np.std(ppg_downsampled) + 1e-6)
        
        # Normalize signals
        ppg_norm = (ppg_downsampled - self.mean) / (self.std + 1e-6)
        ecg_norm = (ecg_sim - np.mean(ecg_sim)) / (np.std(ecg_sim) + 1e-6)
        
        # Prepare signals in format expected by model: (samples, channels, timesteps)
        # We have: PPG (channel 0), ECG (channel 1)
        signals = np.stack([ppg_norm, ecg_norm], axis=0)  # (2, 875)
        signals = signals[np.newaxis, :, :]  # (1, 2, 875)
        
        try:
            # Extract physiological features (PAT, HR)
            # Note: extract_physiological_features expects (n_samples, timesteps)
            ecg_for_feat = ecg_norm[np.newaxis, :]  # (1, 875)
            ppg_for_feat = ppg_norm[np.newaxis, :]  # (1, 875)
            
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
            
            # Standardize PAT and HR to same length
            pat_seqs = standardize_feature_length(pat_seqs, WINDOW_SIZE)
            hr_seqs = standardize_feature_length(hr_seqs, WINDOW_SIZE)
            
            # Create 4-channel input: [ECG, PPG, PAT, HR]
            # Transpose to (timesteps, channels) for model
            input_4ch = create_4_channel_input(
                signals.transpose(0, 2, 1),  # (1, 875, 2)
                pat_seqs,
                hr_seqs
            )
            
            return input_4ch  # Shape: (1, 875, 4)
            
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            # Fallback: use zero PAT and HR channels
            zeros = np.zeros((1, WINDOW_SIZE, 1))
            ppg_ch = ppg_norm.reshape(1, WINDOW_SIZE, 1)
            ecg_ch = ecg_norm.reshape(1, WINDOW_SIZE, 1)
            return np.concatenate([ecg_ch, ppg_ch, zeros, zeros], axis=2)
    
    def predict(self, ppg_window, hr_data=None):
        """
        Predict SBP and DBP from PPG window.
        
        Args:
            ppg_window: PPG signal data
            hr_data: Optional heart rate data from sensor
            
        Returns: (sbp, dbp) tuple
        """
        try:
            # Preprocess
            input_data = self.preprocess_window(ppg_window, hr_data)
            
            # Predict
            predictions = self.model.predict(input_data, verbose=0)
            
            # Model outputs [SBP, DBP] or just SBP (depends on training)
            if predictions.shape[1] == 2:
                sbp, dbp = predictions[0]
            else:
                # If model only predicts SBP, estimate DBP
                sbp = predictions[0, 0]
                dbp = sbp * 0.67  # Rough approximation
            
            return sbp, dbp
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None


def setup_serial_port(port='/dev/ttyUSB0', baudrate=115200):
    """Setup serial connection to ESP32."""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1
        )
        print(f"✓ Connected to {ser.name}")
        return ser
    except serial.SerialException as e:
        print(f"✗ Error opening serial port: {e}")
        return None


def setup_plots():
    """Setup matplotlib for real-time visualization."""
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # PPG signal plot
    ax_ppg = fig.add_subplot(gs[0, :])
    line_ppg, = ax_ppg.plot([], [], 'b-', linewidth=1.5, label="Filtered PPG")
    ax_ppg.set_title("Real-time PPG Signal")
    ax_ppg.set_xlabel("Sample")
    ax_ppg.set_ylabel("Amplitude")
    ax_ppg.legend()
    ax_ppg.grid(True, alpha=0.3)
    
    # Blood pressure history
    ax_bp = fig.add_subplot(gs[1, :])
    line_sbp, = ax_bp.plot([], [], 'r-', linewidth=2, label="SBP", marker='o')
    line_dbp, = ax_bp.plot([], [], 'b-', linewidth=2, label="DBP", marker='o')
    ax_bp.set_title("Blood Pressure Predictions")
    ax_bp.set_xlabel("Prediction #")
    ax_bp.set_ylabel("Blood Pressure (mmHg)")
    ax_bp.legend()
    ax_bp.grid(True, alpha=0.3)
    ax_bp.set_ylim(40, 200)
    
    # Latest BP display
    ax_display = fig.add_subplot(gs[2, 0])
    ax_display.axis('off')
    text_bp = ax_display.text(0.5, 0.5, "Waiting for data...", 
                              fontsize=20, ha='center', va='center',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Status display
    ax_status = fig.add_subplot(gs[2, 1])
    ax_status.axis('off')
    text_status = ax_status.text(0.5, 0.5, "Status: Initializing", 
                                fontsize=12, ha='center', va='center',
                                family='monospace')
    
    return fig, (ax_ppg, ax_bp, ax_display, ax_status), (line_ppg, line_sbp, line_dbp, text_bp, text_status)


def main():
    parser = argparse.ArgumentParser(description='Real-time blood pressure estimation from ESP32 PPG')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    parser.add_argument('--model', default='/home/arwin/codes/python/cuffless-blood-pressure/checkpoints/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--output', default='bp_predictions.csv', help='Output CSV file')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BloodPressurePredictor(args.model)
    
    # Setup serial
    ser = setup_serial_port(args.port, args.baudrate)
    if not ser:
        sys.exit(1)
    
    # Setup plots
    if not args.no_plot:
        fig, axes, lines = setup_plots()
        ax_ppg, ax_bp, ax_display, ax_status = axes
        line_ppg, line_sbp, line_dbp, text_bp, text_status = lines
    
    # Data buffers
    ppg_buffer = deque(maxlen=BUFFER_SIZE)
    hr_buffer = deque(maxlen=BUFFER_SIZE)  # Buffer for sensor-provided heart rate
    sbp_history = []
    dbp_history = []
    sample_count = 0
    prediction_count = 0
    
    # CSV logging
    with open(args.output, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["prediction_id", "timestamp", "sbp", "dbp", "samples_used"])
        
        print("\n" + "="*60)
        print("  REAL-TIME BLOOD PRESSURE ESTIMATION")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Port: {args.port}")
        print(f"Window size: {WINDOW_SIZE} samples (~7 sec)")
        print(f"Update every: {UPDATE_INTERVAL} samples (~2 sec)")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    # Read serial data
                    line = ser.readline().decode('utf-8').strip()
                    if not line or ',' not in line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) not in [3, 4]:  # timestamp,ir_raw,filtered or timestamp,ir_raw,filtered,hr
                        continue
                    
                    timestamp, ir_raw, filtered = parts[0:3]
                    filtered_value = float(filtered)
                    heart_rate = float(parts[3]) if len(parts) == 4 else None
                    
                    # Add to buffers
                    ppg_buffer.append(filtered_value)
                    if heart_rate is not None:
                        hr_buffer.append(heart_rate)
                    sample_count += 1
                    
                    # Update PPG plot
                    if not args.no_plot and len(ppg_buffer) > 100:
                        ppg_array = np.array(list(ppg_buffer)[-1000:])  # Show last 1000 samples
                        x = np.arange(len(ppg_array))
                        line_ppg.set_data(x, ppg_array)
                        ax_ppg.set_xlim(0, len(ppg_array))
                        ax_ppg.set_ylim(ppg_array.min() - 0.1, ppg_array.max() + 0.1)
                    
                    # Make prediction when buffer is full and at intervals
                    if len(ppg_buffer) >= BUFFER_SIZE and sample_count % UPDATE_INTERVAL == 0:
                        ppg_window = np.array(list(ppg_buffer))
                        
                        # Prepare HR data if available
                        hr_window = None
                        if len(hr_buffer) >= BUFFER_SIZE:
                            hr_array = np.array(list(hr_buffer))
                            # Only use sensor HR if values are reasonable (non-zero)
                            if np.mean(hr_array) > 0:
                                hr_window = hr_array
                        
                        # Predict
                        sbp, dbp = predictor.predict(ppg_window, hr_window)
                        
                        if sbp is not None and dbp is not None:
                            prediction_count += 1
                            sbp_history.append(sbp)
                            dbp_history.append(dbp)
                            
                            # Log to CSV
                            csv_writer.writerow([
                                prediction_count,
                                time.time(),
                                f"{sbp:.1f}",
                                f"{dbp:.1f}",
                                len(ppg_window)
                            ])
                            
                            # Update displays
                            if not args.no_plot:
                                # Update BP history plot
                                pred_nums = np.arange(1, len(sbp_history) + 1)
                                line_sbp.set_data(pred_nums, sbp_history)
                                line_dbp.set_data(pred_nums, dbp_history)
                                ax_bp.set_xlim(max(1, len(sbp_history) - 20), len(sbp_history) + 1)
                                
                                # Update BP display
                                text_bp.set_text(f"BP: {sbp:.0f}/{dbp:.0f} mmHg")
                                
                                # Update status
                                status_text = (
                                    f"Predictions: {prediction_count}\n"
                                    f"Samples: {sample_count}\n"
                                    f"Buffer: {len(ppg_buffer)}/{BUFFER_SIZE}"
                                )
                                text_status.set_text(status_text)
                            
                            # Console output
                            print(f"[{prediction_count:3d}] BP: {sbp:6.1f}/{dbp:5.1f} mmHg  "
                                  f"(samples: {sample_count})")
                    
                    # Update plot
                    if not args.no_plot:
                        plt.pause(0.001)
                    
                except ValueError:
                    continue
                except UnicodeDecodeError:
                    continue
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("  SUMMARY")
            print("="*60)
            print(f"Total predictions: {prediction_count}")
            print(f"Total samples: {sample_count}")
            print(f"Data saved to: {args.output}")
            if len(sbp_history) > 0:
                print(f"\nAverage BP: {np.mean(sbp_history):.1f}/{np.mean(dbp_history):.1f} mmHg")
                print(f"SBP range: {np.min(sbp_history):.1f} - {np.max(sbp_history):.1f} mmHg")
                print(f"DBP range: {np.min(dbp_history):.1f} - {np.max(dbp_history):.1f} mmHg")
        
        finally:
            ser.close()
            print("\n✓ Serial port closed")


if __name__ == "__main__":
    main()
