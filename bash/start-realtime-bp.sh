#!/bin/bash
# Quick start script for real-time BP monitoring

echo "=================================="
echo "  Real-time BP Monitoring Setup  "
echo "=================================="
echo ""

# Check if model exists
if [ ! -f "checkpoints/best_model.h5" ]; then
    echo "⚠️  Model not found at checkpoints/best_model.h5"
    echo "   Please train the model first:"
    echo "   python src/models/model.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default values
SERIAL_PORT=${1:-/dev/ttyUSB0}
SERIAL_BAUDRATE=${2:-115200}
SERVER_PORT=${3:-8080}

echo "Configuration:"
echo "  Serial Port: $SERIAL_PORT"
echo "  Baud Rate: $SERIAL_BAUDRATE"
echo "  Server Port: $SERVER_PORT"
echo ""

# Check if serial port exists
if [ ! -e "$SERIAL_PORT" ]; then
    echo "⚠️  Serial port $SERIAL_PORT not found"
    echo "   Available ports:"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "   None found"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting bridge server..."
echo ""
echo "To connect:"
echo "  1. Open browser: http://localhost:3000"
echo "  2. Server will auto-connect to: ws://localhost:$SERVER_PORT/signals"
echo "  3. Place finger on MAX30102 sensor"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start bridge server
python3 bridge_server.py \
    --port $SERVER_PORT \
    --serial-port $SERIAL_PORT \
    --serial-baudrate $SERIAL_BAUDRATE
