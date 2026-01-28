#!/usr/bin/env python3
"""
Test script to verify WebSocket communication with the mock ESP32 server.
This helps debug the message format and connection issues.
"""

import asyncio
import websockets
import json
import time

async def test_websocket_connection():
    """Test the WebSocket connection and message handling."""
    uri = "ws://localhost:8080/signals"
    
    try:
        print("🔗 Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Send start acquisition command
            start_command = {
                "type": "control",
                "command": "start_acquisition"
            }
            
            print("📤 Sending start acquisition command...")
            await websocket.send(json.dumps(start_command))
            
            # Listen for messages for 10 seconds
            print("👂 Listening for messages (10 seconds)...")
            message_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 10:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    if message_count <= 3 or message_count % 50 == 0:  # Show first few and every 50th message
                        print(f"📥 Message #{message_count}: {data['type']}")
                        if data['type'] == 'signal_data':
                            print(f"   - PPG: {data['payload']['ppg_value']:.3f}")
                            print(f"   - ECG: {data['payload']['ecg_value']:.3f}")
                            print(f"   - Quality: {data['payload']['quality']:.3f}")
                        elif data['type'] == 'status':
                            print(f"   - Status: {data['message']}")
                        print()
                    
                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"   Raw message: {message}")
            
            # Send stop acquisition command
            stop_command = {
                "type": "control", 
                "command": "stop_acquisition"
            }
            
            print("📤 Sending stop acquisition command...")
            await websocket.send(json.dumps(stop_command))
            
            # Wait for stop confirmation
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                stop_data = json.loads(response)
                print(f"📥 Stop response: {stop_data}")
            except asyncio.TimeoutError:
                print("⚠️  No stop confirmation received")
            
            print(f"✅ Test completed! Received {message_count} messages total.")
            
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the server is running:")
        print("   python mock_esp32_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 WebSocket Test Script")
    print("=======================")
    asyncio.run(test_websocket_connection())
