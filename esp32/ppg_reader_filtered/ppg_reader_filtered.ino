/*
 * ESP32 PPG Reader with On-Device Filtering
 * 
 * This sketch reads PPG data from MAX30102 sensor and applies:
 * - DC component removal
 * - Bandpass filtering (0.7-4 Hz)
 * 
 * Output format: timestamp,ir_raw,filtered_value
 */

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// Filter parameters
const int FILTER_ORDER = 3;
const float DC_ALPHA = 0.01;
const float SAMPLE_RATE = 200.0;

// Bandpass filter coefficients for 0.7-4 Hz at 200 Hz sampling rate
// Generated using scipy.signal.butter(3, [0.7, 4], 'band', fs=200)
const float b_coeff[7] = {0.00048429, 0.0, -0.00145287, 0.0, 0.00145287, 0.0, -0.00048429};
const float a_coeff[7] = {1.0, -5.73472347, 13.79441094, -17.56330439, 12.59909153, -4.77994664, 0.77398774};

// Filter state variables
float x_buffer[7] = {0}; // Input history
float y_buffer[7] = {0}; // Output history
float dc_estimate = 0.0;

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

  Serial.println("Sensor initialized. Starting data stream...");
  Serial.println("Format: timestamp,ir_raw,filtered_value");
}

float applyBandpassFilter(float input) {
  // Shift buffers (move history)
  for (int i = 6; i > 0; i--) {
    x_buffer[i] = x_buffer[i-1];
    y_buffer[i] = y_buffer[i-1];
  }
  
  x_buffer[0] = input;
  
  // Apply IIR filter (Direct Form II Transposed)
  float y = 0.0;
  
  // Apply numerator (b coefficients)
  for (int i = 0; i < 7; i++) {
    y += b_coeff[i] * x_buffer[i];
  }
  
  // Apply denominator (a coefficients, excluding a[0] which is 1.0)
  for (int i = 1; i < 7; i++) {
    y -= a_coeff[i] * y_buffer[i];
  }
  
  y_buffer[0] = y;
  return y;
}

void loop() {
  uint32_t ir, red;
  
  // Read sensor data
  particleSensor.check();
  
  if (particleSensor.available()) {
    ir = particleSensor.getIR();
    red = particleSensor.getRed();
    particleSensor.nextSample();
    
    // DC component removal (high-pass filter using exponential moving average)
    dc_estimate += DC_ALPHA * ((float)ir - dc_estimate);
    float ac_component = (float)ir - dc_estimate;
    
    // Apply bandpass filter
    float filtered_value = applyBandpassFilter(ac_component);
    
    // Send: timestamp,ir_raw,filtered_value
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ir);
    Serial.print(",");
    Serial.println(filtered_value, 6);
  }
}
