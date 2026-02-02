# Hardware Wiring Guide - ESP32 + MAX30102 + AD8232

## Components Needed

1. **ESP32 Development Board** (any variant with ADC pins)
2. **MAX30102 Pulse Oximeter Module** (PPG sensor)
3. **AD8232 ECG Sensor Module**
4. **ECG Electrodes** (3 disposable gel electrodes)
5. **Jumper wires**
6. **USB cable** (for ESP32)

---

## Wiring Diagram

```
                    ┌─────────────┐
                    │    ESP32    │
                    │             │
         ┌──────────┤ GPIO 21(SDA)│
         │      ┌───┤ GPIO 22(SCL)│
         │      │   │             │
         │      │   │ GPIO 34─────┼──┐
         │      │   │ GPIO 32─────┼──┼──┐
         │      │   │ GPIO 33─────┼──┼──┼──┐
         │      │   │             │  │  │  │
         │      │   │ 3.3V ───────┼──┼──┼──┼──┐
         │      │   │ GND  ───────┼──┼──┼──┼──┼──┐
         │      │   └─────────────┘  │  │  │  │  │
         │      │                    │  │  │  │  │
         ↓      ↓                    ↓  ↓  ↓  ↓  ↓
    ┌──────────────┐          ┌────────────────────┐
    │  MAX30102    │          │      AD8232        │
    │              │          │                    │
    │  SDA ────────┘          │  OUTPUT ───────────┘  │  │  │  │
    │  SCL ───────────────────┘  LO+  ──────────────┘  │  │  │
    │  VIN ──────────────────────┘  LO-  ─────────────┘  │  │
    │  GND ───────────────────────────┘  3.3V ───────────┘  │
    └──────────────┘          │  GND  ──────────────────────┘
                              │                    │
                              │  RA (Red)    ──────┼── Right Arm/Chest
                              │  LA (Yellow) ──────┼── Left Arm/Chest
                              │  RL (Green)  ──────┼── Right Leg/Abdomen
                              └────────────────────┘
```

---

## Pin Connections Table

### MAX30102 → ESP32

| MAX30102 Pin | ESP32 Pin | Description |
|--------------|-----------|-------------|
| VIN          | 3.3V      | Power supply |
| GND          | GND       | Ground |
| SDA          | GPIO 21   | I2C Data (default) |
| SCL          | GPIO 22   | I2C Clock (default) |

### AD8232 → ESP32

| AD8232 Pin | ESP32 Pin | Description |
|------------|-----------|-------------|
| OUTPUT     | GPIO 34   | ECG analog output (ADC1_CH6) |
| LO+        | GPIO 32   | Leads-off detection (+) |
| LO-        | GPIO 33   | Leads-off detection (-) |
| 3.3V       | 3.3V      | Power supply |
| GND        | GND       | Ground |

### ECG Electrodes → AD8232

| AD8232 Pin | Electrode Color | Body Placement |
|------------|----------------|----------------|
| RA (SDN)   | Red            | Right arm or right side of chest |
| LA         | Yellow         | Left arm or left side of chest |
| RL (GND)   | Green          | Right leg or lower right abdomen |

---

## Detailed Connection Steps

### Step 1: Connect MAX30102

1. Connect **VIN** to ESP32 **3.3V**
2. Connect **GND** to ESP32 **GND**
3. Connect **SDA** to ESP32 **GPIO 21**
4. Connect **SCL** to ESP32 **GPIO 22**

**Note**: Some MAX30102 modules have 5V tolerance, but use 3.3V for safety.

### Step 2: Connect AD8232

1. Connect **3.3V** to ESP32 **3.3V** (share with MAX30102)
2. Connect **GND** to ESP32 **GND** (share with MAX30102)
3. Connect **OUTPUT** to ESP32 **GPIO 34**
4. Connect **LO+** to ESP32 **GPIO 32**
5. Connect **LO-** to ESP32 **GPIO 33**

### Step 3: Attach ECG Electrodes

1. **Clean skin** with alcohol wipes where electrodes will be placed
2. **Peel backing** from disposable gel electrodes
3. **Attach electrodes** to body:
   - **Red (RA)**: Right side of chest, below collarbone
   - **Yellow (LA)**: Left side of chest, below collarbone
   - **Green (RL)**: Lower right abdomen (reference/ground)

**Alternative placement for testing:**
- Red: Right wrist
- Yellow: Left wrist
- Green: Right ankle or abdomen

4. **Connect electrode cables** to AD8232:
   - Red cable → RA pin
   - Yellow cable → LA pin
   - Green cable → RL pin

---

## Power Considerations

**Total Current Draw:**
- ESP32: ~80-160mA (with WiFi)
- MAX30102: ~20-50mA (depending on LED brightness)
- AD8232: ~170μA (0.17mA)
- **Total**: ~100-210mA

**Power Sources:**
- ✅ USB cable from computer (500mA available)
- ✅ USB power bank (1A+ available)
- ✅ Wall adapter with USB (1A+ recommended)
- ❌ Coin cell batteries (insufficient current)

---

## GPIO Pin Selection Notes

### GPIO 34 (ECG OUTPUT)
- **ADC1_CH6** - 12-bit ADC (0-4095 range)
- Input-only pin (perfect for analog reading)
- No PWM, no I2C conflicts

### GPIO 32 & 33 (Leads-off Detection)
- Can be used as digital inputs
- Can be changed to any available GPIO if needed
- Pull-up resistors not required (AD8232 handles)

### Why These Pins?
- **GPIO 34**: Dedicated ADC pin, high resolution
- **GPIO 32/33**: General purpose, no special functions conflicting
- **GPIO 21/22**: Standard I2C pins for ESP32

---

## Alternative Pin Configurations

If you need to use different pins:

### For ECG OUTPUT (Analog):
Any ADC1 pin can be used:
- GPIO 32, 33, 34, 35, 36, 37, 38, 39

**In code, change:**
```cpp
const int ECG_PIN = 34;  // Change to your chosen ADC pin
```

### For LO+/LO- (Digital):
Any GPIO can be used:
- GPIO 2, 4, 5, 12, 13, 14, 15, 25, 26, 27, 32, 33

**In code, change:**
```cpp
const int LO_PLUS_PIN = 32;   // Change to your chosen pin
const int LO_MINUS_PIN = 33;  // Change to your chosen pin
```

### For I2C (MAX30102):
Can use different pins if needed:

**In code, before `particleSensor.begin()`:**
```cpp
Wire.begin(SDA_PIN, SCL_PIN);  // e.g., Wire.begin(25, 26)
```

---

## Testing Setup

### 1. Check MAX30102 Connection

```
Upload sketch → Open Serial Monitor (115200 baud)
Expected: "MAX30102 initialized"
If error: Check I2C connections, try different module
```

### 2. Check AD8232 Connection

```
Attach all 3 electrodes → Check serial output
Expected: Non-zero ecg_raw and ecg_filtered values
If zero: Check leads-off detection or electrode contact
```

### 3. Verify Both Signals

```
Place finger on MAX30102 AND attach ECG electrodes
Expected serial output:
1234,65000,0.234567,2048,0.012345
      ↑PPG          ↑ECG
```

---

## Common Issues

### "MAX30102 was not found"
- Check I2C wiring (SDA/SCL)
- Verify 3.3V power
- Try different MAX30102 module
- Check solder joints on module

### ECG always shows 0
- Leads not attached → Attach all 3 electrodes properly
- Check OUTPUT pin connection to GPIO 34
- Verify LO+/LO- connections
- Electrodes may be dry → Use fresh electrodes

### Noisy ECG signal
- Poor electrode contact → Press electrodes firmly
- Electrical interference → Move away from power supplies
- Motion artifacts → Stay still during reading
- Loose wires → Secure all connections

### PPG works but ECG doesn't
- Independent circuits, check AD8232 power (3.3V)
- Verify GPIO 34 is not used elsewhere
- Test with multimeter: OUTPUT should show ~1.65V at rest

---

## Safety Notes

⚠️ **IMPORTANT SAFETY INFORMATION:**

1. **Not for medical diagnosis** - This is for research/educational purposes only
2. **Battery powered only during human testing** - Do NOT use USB power connected to mains when attached to body
3. **Keep away from** - Pacemakers, defibrillators, other medical devices
4. **No medical decisions** - Do not use for medical treatment decisions
5. **Proper electrodes** - Use only gel electrodes designed for ECG
6. **Skin irritation** - Remove electrodes if skin becomes irritated
7. **Electrical safety** - Ensure all connections are insulated and secure

---

## Next Steps

After wiring:
1. Upload the Arduino sketch: `esp32/ppg_reader_filtered/ppg_reader_filtered.ino`
2. Open Serial Monitor (115200 baud) to verify data
3. Run bridge server: `python bridge_server.py --serial-port /dev/ttyUSB0`
4. Start frontend: `cd frontend && pnpm dev`
5. Open browser: `http://localhost:3000`

See [REALTIME_SETUP.md](REALTIME_SETUP.md) for complete software setup.
