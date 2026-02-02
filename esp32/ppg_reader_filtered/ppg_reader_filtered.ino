/*
 * ESP32 PPG + ECG Reader (Raw Data)
 * 
 * This sketch reads:
 * - PPG data from MAX30102 sensor (IR channel)
 * - ECG data from AD8232 sensor
 * 
 * Sends RAW data for filtering in Python (bridge_server.py)
 * 
 * Output format: timestamp,ppg_raw,ecg_raw
 */

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// AD8232 ECG sensor pins
const int ECG_PIN = 34;           // Analog input pin for ECG signal
const int LO_PLUS_PIN = 32;       // Leads-off detection LO+
const int LO_MINUS_PIN = 33;      // Leads-off detection LO-

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing sensors...");

  // Initialize MAX30102 PPG sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1);
  }

  // Configure sensor
  byte ledBrightness = 60;  // Options: 0=Off to 255=50mA
  byte sampleAverage = 1;   // Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2;         // Options: 1 = Red only, 2 = Red + IR
  int sampleRate = 200;     // Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411;     // Options: 69, 118, 215, 411
  int adcRange = 4096;      // Options: 2048, 4096, 8192, 16384

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeIR(0x0A);

  // Configure AD8232 pins
  pinMode(ECG_PIN, INPUT);
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);

  Serial.println("Sensors initialized. Starting data stream...");
  Serial.println("Format: timestamp,ppg_raw,ecg_raw");
}

void loop() {
  // Read PPG sensor data
  particleSensor.check();
  
  if (particleSensor.available()) {
    long ir = particleSensor.getIR();
    particleSensor.nextSample();
    
    if (ir < 50000) {
      // No finger detected
      return;
    }
    
    // Check ECG leads-off detection
    bool ecg_valid = (digitalRead(LO_PLUS_PIN) == 0) && (digitalRead(LO_MINUS_PIN) == 0);
    
    // Read ECG data
    int ecg_raw = 0;
    if (ecg_valid) {
      ecg_raw = analogRead(ECG_PIN);
    }
    
    // Send: timestamp,ppg_raw,ecg_raw
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ir);
    Serial.print(",");
    Serial.println(ecg_raw);
  }
}