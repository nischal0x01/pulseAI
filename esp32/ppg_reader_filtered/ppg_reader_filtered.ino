/*
 * ESP32 PPG + ECG Reader with On-Device Filtering
 * 
 * This sketch reads:
 * - PPG data from MAX30102 sensor
 * - ECG data from AD8232 sensor
 * 
 * Filtering applied:
 * - PPG: DC removal + Bandpass (0.7-4 Hz)
 * - ECG: Bandpass (0.5-40 Hz)
 * 
 * Output format: timestamp,ppg_raw,ppg_filtered,ecg_raw,ecg_filtered
 */

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// AD8232 ECG sensor pins
const int ECG_PIN = 34;           // Analog input pin for ECG signal
const int LO_PLUS_PIN = 32;       // Leads-off detection LO+
const int LO_MINUS_PIN = 33;      // Leads-off detection LO-

// Filter parameters
const int FILTER_ORDER = 3;
const float DC_ALPHA = 0.01;
const float SAMPLE_RATE = 200.0;

// PPG Bandpass filter coefficients for 0.7-4 Hz at 200 Hz sampling rate
// Generated using scipy.signal.butter(3, [0.7, 4], 'band', fs=200)
const float ppg_b_coeff[7] = {0.00048429, 0.0, -0.00145287, 0.0, 0.00145287, 0.0, -0.00048429};
const float ppg_a_coeff[7] = {1.0, -5.73472347, 13.79441094, -17.56330439, 12.59909153, -4.77994664, 0.77398774};

// ECG Bandpass filter coefficients for 0.5-40 Hz at 200 Hz sampling rate
// Generated using scipy.signal.butter(3, [0.5, 40], 'band', fs=200)
const float ecg_b_coeff[7] = {0.11957729, 0.0, -0.35873187, 0.0, 0.35873187, 0.0, -0.11957729};
const float ecg_a_coeff[7] = {1.0, -2.37409474, 3.32826312, -2.98588214, 1.81104923, -0.60619881, 0.10359967};

// PPG Filter state variables
float ppg_x_buffer[7] = {0};
float ppg_y_buffer[7] = {0};
float ppg_dc_estimate = 0.0;

// ECG Filter state variables
float ecg_x_buffer[7] = {0};
float ecg_y_buffer[7] = {0};

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing PPG sensor with on-device filtering...");

  // Initialize sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1);
  }

  // Configure sensor
  byte ledBrightness = 60;  // Options: 0=Off to 255=50mA
  byte sampleAverage = 1;   // Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2;         // Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
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
  Serial.println("Format: timestamp,ppg_raw,ppg_filtered,ecg_raw,ecg_filtered");
}

float applyPPGBandpassFilter(float input) {
  // Shift buffers (move history)
  for (int i = 6; i > 0; i--) {
    ppg_x_buffer[i] = ppg_x_buffer[i-1];
    ppg_y_buffer[i] = ppg_y_buffer[i-1];
  }
  
  ppg_x_buffer[0] = input;
  
  // Apply IIR filter (Direct Form II Transposed)
  float y = 0.0;
  
  // Apply numerator (b coefficients)
  for (int i = 0; i < 7; i++) {
    y += ppg_b_coeff[i] * ppg_x_buffer[i];
  }
  
  // Apply denominator (a coefficients, excluding a[0] which is 1.0)
  for (int i = 1; i < 7; i++) {
    y -= ppg_a_coeff[i] * ppg_y_buffer[i];
  }
  
  ppg_y_buffer[0] = y;
  return y;
}

float applyECGBandpassFilter(float input) {
  // Shift buffers (move history)
  for (int i = 6; i > 0; i--) {
    ecg_x_buffer[i] = ecg_x_buffer[i-1];
    ecg_y_buffer[i] = ecg_y_buffer[i-1];
  }
  
  ecg_x_buffer[0] = input;
  
  // Apply IIR filter (Direct Form II Transposed)
  float y = 0.0;
  
  // Apply numerator (b coefficients)
  for (int i = 0; i < 7; i++) {
    y += ecg_b_coeff[i] * ecg_x_buffer[i];
  }
  
  // Apply denominator (a coefficients, excluding a[0] which is 1.0)
  for (int i = 1; i < 7; i++) {
    y -= ecg_a_coeff[i] * ecg_y_buffer[i];
  }
  
  ecg_y_buffer[0] = y;
  return y;
}

void loop() {
  uint32_t ir, red;
  
  // Read PPG sensor data
  particleSensor.check();
  
  if (particleSensor.available()) {
    ir = particleSensor.getIR();
    red = particleSensor.getRed();
    particleSensor.nextSample();
    
    // Check ECG leads-off detection
    bool ecg_valid = (digitalRead(LO_PLUS_PIN) == 0) && (digitalRead(LO_MINUS_PIN) == 0);
    
    // Read ECG data
    int ecg_raw = 0;
    float ecg_filtered = 0.0;
    
    if (ecg_valid) {
      ecg_raw = analogRead(ECG_PIN);
      
      // Normalize ECG to -1 to 1 range (assuming 12-bit ADC)
      float ecg_normalized = (ecg_raw - 2048.0) / 2048.0;
      
      // Apply ECG bandpass filter
      ecg_filtered = applyECGBandpassFilter(ecg_normalized);
    }
    
    // Process PPG with DC component removal
    ppg_dc_estimate += DC_ALPHA * ((float)ir - ppg_dc_estimate);
    float ppg_ac_component = (float)ir - ppg_dc_estimate;
    
    // Apply PPG bandpass filter
    float ppg_filtered = applyPPGBandpassFilter(ppg_ac_component);
    
    // Send: timestamp,ppg_raw,ppg_filtered,ecg_raw,ecg_filtered
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ir);
    Serial.print(",");
    Serial.print(ppg_filtered, 6);
    Serial.print(",");
    Serial.print(ecg_raw);
    Serial.print(",");
    Serial.println(ecg_filtered, 6);
  }
}
