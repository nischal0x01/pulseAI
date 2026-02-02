#!/usr/bin/env python3
"""
PPG Reader - Receives pre-filtered data from ESP32

This script reads PPG data that has already been filtered on the ESP32 device.
The ESP32 performs:
- DC component removal
- Bandpass filtering (0.7-4 Hz)

This script only handles:
- Data reception
- Visualization
- Data logging
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv
import time
import argparse
import sys

WINDOW_SIZE = 2000
AXIS_UPDATE_INTERVAL = 200

def setup_serial_port(port='/dev/ttyUSB0', baudrate=115200):
    """Setup serial connection to ESP32."""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=None  # Blocking reads
        )
        print(f"✓ Connected to {ser.name}")
        return ser
    except serial.SerialException as e:
        print(f"✗ Error opening serial port: {e}")
        print("  Please check:")
        print("  - Port is correct (try: ls /dev/ttyUSB* or /dev/ttyACM*)")
        print("  - User has permissions (add to 'dialout' group)")
        print("  - ESP32 is connected and powered")
        return None

def setup_plot():
    """Setup matplotlib for real-time plotting."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line_filtered, = ax.plot([], [], label="Filtered PPG (ESP32)", linewidth=1.5)
    
    ax.set_title("Pre-Filtered PPG Waveform (ESP32 Processing)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Normalized Filtered Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax, plt, line_filtered

def main():
    parser = argparse.ArgumentParser(description='Read pre-filtered PPG data from ESP32')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--output', default='ppg_stream.csv', help='Output CSV file (default: ppg_stream.csv)')
    parser.add_argument('--no-plot', action='store_true', help='Disable real-time plotting')
    
    args = parser.parse_args()

    # Setup serial connection
    ser = setup_serial_port(args.port, args.baudrate)
    if not ser:
        sys.exit(1)

    # Setup plotting
    if not args.no_plot:
        ax, plt, line_filtered = setup_plot()

    filtered_buffer = deque(maxlen=WINDOW_SIZE)
    sample_count = 0

    # Open CSV file for logging
    with open(args.output, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header if file is new
        if csv_file.tell() == 0:
            csv_writer.writerow(["timestamp", "ir_raw", "ir_filtered"])
            print(f"✓ Created new CSV file: {args.output}")
        else:
            print(f"✓ Appending to existing CSV file: {args.output}")

        print("\nStarting data acquisition...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                try:
                    # Read line from serial
                    line_bytes = ser.readline()
                    if not line_bytes:
                        continue

                    line_str = line_bytes.decode('utf-8').strip()
                    
                    # Skip non-data lines (e.g., debug messages)
                    if not line_str or ',' not in line_str:
                        continue

                    # Parse: timestamp,ir_raw,filtered_value
                    parts = line_str.split(',')
                    if len(parts) != 3:
                        continue

                    timestamp, ir_raw, filtered = parts
                    filtered_value = float(filtered)
                    
                    # Store in buffer
                    filtered_buffer.append(filtered_value)
                    
                    # Write to CSV
                    csv_writer.writerow([timestamp, ir_raw, filtered_value])
                    
                    sample_count += 1

                    # Update plot
                    if not args.no_plot and len(filtered_buffer) > 100:
                        y = np.array(filtered_buffer)
                        x = np.arange(len(y))
                        
                        # Normalize for display
                        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-6)

                        line_filtered.set_data(x, y_norm)
                        ax.set_xlim(0, len(y))

                        # Update y-axis limits periodically
                        if sample_count % AXIS_UPDATE_INTERVAL == 0:
                            ax.set_ylim(y_norm.min() - 0.5, y_norm.max() + 0.5)

                        plt.pause(0.001)

                    # Progress indicator
                    if sample_count % 1000 == 0:
                        print(f"Samples collected: {sample_count}")

                except ValueError as e:
                    # Skip malformed data
                    continue
                except UnicodeDecodeError:
                    # Skip invalid bytes
                    continue

        except KeyboardInterrupt:
            print(f"\n\n✓ Stopped by user")
            print(f"✓ Total samples collected: {sample_count}")
            print(f"✓ Data saved to: {args.output}")

        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            ser.close()
            print("✓ Serial port closed")

if __name__ == "__main__":
    main()
