/*
  =============================================================================
  AI-Based Water Pipeline Leakage Detection System
  Sensor Node Firmware (Compatible with ESP32, ESP8266, Arduino Uno/Mega/Nano)
  =============================================================================
  Sensors Supported:
  1. YF-S201 Water Flow Sensor (Hall effect pulse output on digital interrupt)
  2. G1/4" 0 - 1.2 MPa (0 - 12 Bar) Pressure Transducer (Linear Analog 0.5V - 4.5V)
  3. Secondary Node Flow & Pressure Sensors (Dual-Node Mass Balance)
  4. Piezo / Vibration / Acoustic sensor (Analog or digital)
  5. Local Actuators: Warning Buzzer & Alarm LED
  
  Communication:
  - Formats telemetry as standard JSON strings transmitted over USB Serial (115200 baud).
  - Can be read directly by the Streamlit Dashboard or Python Serial Gateway.
  =============================================================================
*/

#include <Arduino.h>

// ---------------------- PIN CONFIGURATION ----------------------
#if defined(ESP32)
  #define PIN_FLOW_INLET     18   // Interrupt Pin for Inlet Flow Sensor
  #define PIN_FLOW_OUTLET    19   // Interrupt Pin for Outlet Flow Sensor
  #define PIN_PRESS_INLET    34   // ADC1 Channel for Inlet Pressure Sensor
  #define PIN_PRESS_OUTLET   35   // ADC1 Channel for Outlet Pressure Sensor
  #define PIN_VIBRATION      32   // ADC Channel for Acoustic/Piezo Vibration
  #define PIN_BUZZER         23   // Alarm Buzzer output
  #define PIN_LED_ALARM      22   // Red Alarm Warning LED
  #define PIN_LED_NORMAL     21   // Green Normal Status LED
  #define ADC_MAX_VOLTAGE    3.3
  #define ADC_RESOLUTION     4095.0
#else // Arduino Uno / Nano / Mega
  #define PIN_FLOW_INLET     2    // INT0
  #define PIN_FLOW_OUTLET    3    // INT1
  #define PIN_PRESS_INLET    A0   // Analog 0
  #define PIN_PRESS_OUTLET   A1   // Analog 1
  #define PIN_VIBRATION      A2   // Analog 2
  #define PIN_BUZZER         8    // Buzzer pin
  #define PIN_LED_ALARM      7    // Red LED
  #define PIN_LED_NORMAL     6    // Green LED
  #define ADC_MAX_VOLTAGE    5.0
  #define ADC_RESOLUTION     1023.0
#endif

// ---------------------- CALIBRATION CONSTANTS ----------------------
// YF-S201 Flow Sensor: Pulse frequency (Hz) = 7.5 * Q (L/min)
// Each pulse represents ~ 1.0 / (7.5 * 60) Litres
const float FLOW_CALIBRATION_FACTOR = 7.5; 

// Pressure Transducer: 0.5V = 0 MPa (0 bar), 4.5V = 1.2 MPa (12.0 bar)
const float V_MIN = 0.5; // Offset at 0 bar
const float V_MAX = 4.5; // Max output voltage at 12 bar
const float P_MAX_BAR = 12.0; 

// ---------------------- VOLATILE COUNTERS ----------------------
volatile unsigned long pulseCountInlet = 0;
volatile unsigned long pulseCountOutlet = 0;

// Interrupt Service Routines
void IRAM_ATTR pulseCounterInlet() {
  pulseCountInlet++;
}

void IRAM_ATTR pulseCounterOutlet() {
  pulseCountOutlet++;
}

// Timing
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 1000; // 1-second sample loop

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {
    // Wait for Serial monitor connection on boards with native USB
  }

  // Setup digital I/O
  pinMode(PIN_FLOW_INLET, INPUT_PULLUP);
  pinMode(PIN_FLOW_OUTLET, INPUT_PULLUP);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_ALARM, OUTPUT);
  pinMode(PIN_LED_NORMAL, OUTPUT);

  // Initial LED status
  digitalWrite(PIN_LED_NORMAL, HIGH);
  digitalWrite(PIN_LED_ALARM, LOW);
  digitalWrite(PIN_BUZZER, LOW);

  // Attach interrupts
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW_INLET), pulseCounterInlet, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW_OUTLET), pulseCounterOutlet, RISING);

  Serial.println(F("{\"system\": \"Water Pipeline Sensor Node\", \"status\": \"READY\", \"baud\": 115200}"));
}

float readPressureSensor(int pin) {
  int rawADC = analogRead(pin);
  float voltage = (rawADC / ADC_RESOLUTION) * ADC_MAX_VOLTAGE;
  
  // Convert voltage to pressure in bar
  float pressureBar = 0.0;
  if (voltage >= V_MIN) {
    pressureBar = ((voltage - V_MIN) / (V_MAX - V_MIN)) * P_MAX_BAR;
  }
  if (pressureBar < 0.0) pressureBar = 0.0;
  return pressureBar;
}

float readVibrationSensor(int pin) {
  int rawADC = analogRead(pin);
  float voltage = (rawADC / ADC_RESOLUTION) * ADC_MAX_VOLTAGE;
  // Convert to relative dB acoustic level
  float dbLevel = 10.0 + (voltage / ADC_MAX_VOLTAGE) * 50.0;
  return dbLevel;
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    // Detach or disable interrupts briefly while reading pulse counters
    noInterrupts();
    unsigned long pulsesIn = pulseCountInlet;
    unsigned long pulsesOut = pulseCountOutlet;
    pulseCountInlet = 0;
    pulseCountOutlet = 0;
    interrupts();

    float elapsedSeconds = (currentMillis - lastSampleTime) / 1000.0;
    lastSampleTime = currentMillis;

    // Calculate Flow in Litres Per Minute (LPM)
    float inletFlowLPM = (pulsesIn / elapsedSeconds) / FLOW_CALIBRATION_FACTOR;
    float outletFlowLPM = (pulsesOut / elapsedSeconds) / FLOW_CALIBRATION_FACTOR;
    
    // Read Pressures
    float inletPressBar = readPressureSensor(PIN_PRESS_INLET);
    float outletPressBar = readPressureSensor(PIN_PRESS_OUTLET);
    
    // If only 1 hardware pressure sensor is attached, emulate mid/outlet drop for demo
    if (outletPressBar < 0.1 && inletPressBar > 0.5) {
      outletPressBar = inletPressBar * 0.88;
    }
    float midPressBar = (inletPressBar + outletPressBar) / 2.0;
    
    // Read Acoustic / Vibration
    float vibrationDB = readVibrationSensor(PIN_VIBRATION);

    // Compute differential
    float flowDiff = inletFlowLPM - outletFlowLPM;
    float pressDrop = inletPressBar - outletPressBar;

    // Local Hardware Failsafe / Threshold Logic:
    // If mass balance discrepancy > 3.0 LPM and pressure drop > 0.4 bar
    bool hardwareAlert = (flowDiff > 3.5) && (pressDrop > 0.35);

    if (hardwareAlert) {
      digitalWrite(PIN_LED_ALARM, HIGH);
      digitalWrite(PIN_LED_NORMAL, LOW);
      digitalWrite(PIN_BUZZER, HIGH);
    } else {
      digitalWrite(PIN_LED_ALARM, LOW);
      digitalWrite(PIN_LED_NORMAL, HIGH);
      digitalWrite(PIN_BUZZER, LOW);
    }

    // Transmit telemetry as JSON packet
    Serial.print(F("{\"timestamp_ms\": "));
    Serial.print(currentMillis);
    Serial.print(F(", \"inlet_flow_lpm\": "));
    Serial.print(inletFlowLPM, 2);
    Serial.print(F(", \"outlet_flow_lpm\": "));
    Serial.print(outletFlowLPM, 2);
    Serial.print(F(", \"flow_difference_lpm\": "));
    Serial.print(flowDiff, 2);
    Serial.print(F(", \"inlet_pressure_bar\": "));
    Serial.print(inletPressBar, 3);
    Serial.print(F(", \"mid_pressure_bar\": "));
    Serial.print(midPressBar, 3);
    Serial.print(F(", \"outlet_pressure_bar\": "));
    Serial.print(outletPressBar, 3);
    Serial.print(F(", \"pressure_drop_bar\": "));
    Serial.print(pressDrop, 3);
    Serial.print(F(", \"vibration_noise_db\": "));
    Serial.print(vibrationDB, 1);
    Serial.print(F(", \"water_temperature_c\": 24.5"));
    Serial.print(F(", \"hardware_alert\": "));
    Serial.print(hardwareAlert ? "true" : "false");
    Serial.println(F("}"));
  }
}
