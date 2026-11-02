# Hardware Prototype Guide & Circuit Architecture

This module details the physical implementation of the sensor node for the **AI-Based Water Pipeline Leakage Detection System**.

---

## 1. Bill of Materials (BOM)

| Component | Specification | Function | Est. Cost (INR / USD) |
| :--- | :--- | :--- | :--- |
| **Microcontroller** | ESP32 DevKit V1 (or Arduino Uno / Nano) | Edge telemetry processing, ADC sampling, serial JSON output | ₹450 / $6.00 |
| **Water Flow Sensor** | YF-S201 (1/2" Hall Turbine, 1–30 LPM) | Inlet & Outlet mass-flow measurement | ₹250 each / $3.50 |
| **Pressure Transducer** | G1/4" 0–1.2 MPa (0–12 Bar) Linear Analog (0.5V–4.5V) | Upstream and downstream pipeline static pressure | ₹650 each / $8.00 |
| **Vibration / Acoustic Sensor**| SW-420 or Piezoelectric film sensor | Acoustic leak detection / pipe vibration signature | ₹60 / $0.80 |
| **Active Buzzer** | 5V DC Active Buzzer | Local auditory warning alert | ₹25 / $0.35 |
| **Status LEDs** | Red (5mm) and Green (5mm) with 220Ω resistors | Visual normal / alert indication | ₹10 / $0.15 |
| **Power Supply** | 5V 2A DC Adapter or USB 5V | Sensor and microcontroller excitation | ₹150 / $2.00 |
| **PVC Pipe Assembly** | 1/2" or 3/4" PVC pipes with Tee-joints & Ball Valves | Experimental water loop with controlled leak valve | ₹300 / $4.00 |

---

## 2. Pinout Connections

### A. ESP32 DevKit V1 Pinout
| Sensor / Peripheral | Sensor Pin | ESP32 GPIO | Notes |
| :--- | :--- | :--- | :--- |
| **Inlet Flow Sensor (YF-S201)** | VCC (Red) | 5V (VIN) | Needs 5V power |
| | GND (Black) | GND | Common ground |
| | Signal (Yellow)| GPIO 18 | Configured as Interrupt Pin |
| **Outlet Flow Sensor** | Signal (Yellow)| GPIO 19 | Configured as Interrupt Pin |
| **Inlet Pressure Sensor** | VCC | 5V (VIN) | 5V excitation |
| | GND | GND | Common ground |
| | Output (0.5–4.5V) | GPIO 34 | Analog input (ADC1) *Use voltage divider to 3.3V max!* |
| **Outlet Pressure Sensor** | Output | GPIO 35 | Analog input (ADC1) |
| **Acoustic Sensor** | DO / AO | GPIO 32 | Analog / digital vibration |
| **Alarm Buzzer** | (+) | GPIO 23 | Active HIGH alarm |
| **Alarm LED (Red)** | Anode (+) | GPIO 22 | Active HIGH with 220Ω resistor |
| **Normal LED (Green)**| Anode (+) | GPIO 21 | Active HIGH with 220Ω resistor |

> **IMPORTANT FOR ESP32**: The pressure sensor outputs up to 4.5V. Since ESP32 ADC pins accept a maximum of 3.3V, pass the pressure sensor signal through a simple 2-resistor voltage divider (e.g., $R_1 = 1.8\text{ k}\Omega$, $R_2 = 3.3\text{ k}\Omega$) or use a 3.3V-compatible sensor!

---

### B. Arduino Uno / Nano Pinout
| Sensor / Peripheral | Sensor Pin | Arduino Pin | Notes |
| :--- | :--- | :--- | :--- |
| **Inlet Flow Sensor** | Signal | Digital Pin 2 | External Interrupt INT0 |
| **Outlet Flow Sensor** | Signal | Digital Pin 3 | External Interrupt INT1 |
| **Inlet Pressure Sensor** | Output | Analog Pin A0 | 0 to 5V (10-bit ADC: 0–1023) |
| **Outlet Pressure Sensor**| Output | Analog Pin A1 | 0 to 5V |
| **Vibration Sensor** | Output | Analog Pin A2 | Analog acoustic listening |
| **Alarm Buzzer** | (+) | Digital Pin 8 | Digital Output |
| **Alarm LED (Red)** | Anode (+) | Digital Pin 7 | Digital Output |
| **Normal LED (Green)**| Anode (+) | Digital Pin 6 | Digital Output |

---

## 3. Mathematical Calibration Models

### Flow Measurement
$$Q (\text{L/min}) = \frac{f (\text{Hz})}{7.5}$$
Where $f$ is pulse frequency counted over a 1-second sampling window.

### Pressure Measurement
$$V_{out} = \frac{\text{ADC Reading}}{\text{ADC Max}} \times V_{ref}$$
$$P (\text{Bar}) = \left(\frac{V_{out} - 0.5\text{ V}}{4.0\text{ V}}\right) \times 12.0\text{ Bar}$$

---

## 4. Software Gateway Operation
1. Upload `arduino_esp32_sensor_node.ino` using the Arduino IDE.
2. Connect the microcontroller via USB to your computer.
3. Launch the Streamlit dashboard (`streamlit run app.py`).
4. Navigate to the **"IoT & Hardware Gateway"** tab.
5. Select your COM port (e.g., `COM3` on Windows) and Baud rate `115200`.
6. Click **"Connect"** — real-time physical sensor readings will stream into the AI inference engine!
