# AI-Based Water Pipeline Leakage Detection System
## Presentation Slide Deck Outline (Viva / Project Defense)

---

### Slide 1: Title Slide
- **Title**: AI-Based Water Pipeline Leakage Detection Using Sensors
- **Subtitle**: A Cyber-Physical System Integrating Multi-Sensor Telemetry, Mass Conservation Hydraulics, and Machine Learning
- **Presenter**: Project Team
- **Key Focus**: Non-Revenue Water (NRW) Reduction, Real-Time Anomaly Detection, False Alarm Mitigation

---

### Slide 2: Problem Statement & Motivation
- **Global Challenge**: 30% to 45% of treated drinking water is lost to pipeline leaks globally (World Bank: \$14B/year lost).
- **Current Limitations**:
  - Manual acoustic listening rods are labor-intensive and weather-dependent.
  - Conventional static SCADA threshold alarms cause excessive false alarms during normal consumer morning/evening usage spikes.
  - Underground pinhole leaks remain undetected until catastrophic road collapses / sinkholes occur.
- **Goal**: Autonomous, AI-driven, continuous multi-node pipeline integrity monitoring with <1% false alarm rate.

---

### Slide 3: Hydraulic & Physical Principles
- **Continuity Law (Conservation of Mass)**:
  $$Q_{in}(t) - Q_{out}(t) = Q_{leak}(t) + \epsilon(t)$$
  When a leak occurs, mass balance is disrupted ($Q_{in} > Q_{out}$).
- **Darcy-Weisbach Friction & Hydraulic Gradient**:
  Accelerated fluid velocity upstream of a rupture steepens the pressure gradient, inducing localized pressure drops ($\Delta P$).
- **Torricelli's Orifice Discharge**:
  $$Q_{leak} = C_d A \sqrt{2g \Delta P / \rho}$$

---

### Slide 4: System Architecture & End-to-End Workflow
- **Sensor Tier**:
  - YF-S201 Turbine Flow Sensors (Digital pulse counting)
  - 0–1.2 MPa Linear Pressure Transducers (Analog voltage)
  - SW-420 / Piezoelectric Vibration Sensor (Acoustic jetting noise)
- **Edge Microcontroller Tier**:
  - ESP32 / Arduino Uno processing raw sensor pulses & ADC readings
  - JSON packet serialization over USB Serial / Wi-Fi
- **AI Processing & Decision Engine**:
  - Domain-specific feature engineering (mass balance ratios, rolling statistics)
  - Random Forest, Decision Tree, Gradient Boosting, Isolation Forest
- **SCADA Web Dashboard (Streamlit)**:
  - Real-time gauge telemetry, animated alarm triggers, what-if simulator, batch CSV analyzer.

---

### Slide 5: Hardware Prototype & Circuit Design
- **Microcontrollers**: ESP32 DevKit V1 (3.3V) / Arduino Uno (5V).
- **Interrupt Servicing**:
  - Hardware external interrupts count turbine rotation pulses for precise LPM conversion ($Q = f / 7.5$).
- **Analog Signal Conditioning**:
  - Pressure transducer signal calibrated linearly: $0.5\text{V} \rightarrow 0\text{ Bar}$, $4.5\text{V} \rightarrow 12.0\text{ Bar}$.
  - Voltage divider protection for ESP32 3.3V ADC inputs.
- **Fail-safe Local Actuators**: Onboard buzzer and dual Red/Green status LEDs.

---

### Slide 6: Dataset Engineering & Realistic Physics
- **Sample Size**: 16,000 continuous time-series records (5.5 days of monitoring).
- **Physical Dynamics Modeled**:
  - Diurnal domestic demand fluctuation (morning peak 07:00–09:00, evening peak 19:00–21:00).
  - Background sensor noise ($\sigma = 0.45\text{ LPM}$).
- **Failure Modes Covered**:
  - Normal Flow: 78.3%
  - Minor Pinhole Leak (1.5–4.5 LPM): 7.6%
  - Moderate Crack (5–12 LPM): 9.0%
  - Major Pipe Burst (>16 LPM): 4.5%
  - Transient Non-Leak Valve Spurt (False Alarm Challenge): 0.5%

---

### Slide 7: Feature Engineering Pipeline
- Raw telemetry transformed into 18 domain-informed features:
  1. `flow_loss_ratio`: $\frac{Q_{in} - Q_{out}}{Q_{in}}$
  2. `total_pressure_drop`: $P_{in} - P_{out}$
  3. `inlet_to_mid_drop` & `mid_to_outlet_drop`: Locates pipe rupture quadrant.
  4. `hydraulic_resistance`: $\frac{\Delta P}{Q_{in}^2}$ (Pipe friction signature).
  5. `vib_acoustic_intensity`: Acoustic noise normalized against baseline.
  6. `flow_diff_rolling_mean` & `flow_diff_rolling_std`: 5-step rolling window to filter transient measurement spikes.

---

### Slide 8: Machine Learning Model Benchmarking
- **Supervised Models**:
  - **Random Forest (Champion)**: 120 trees; handles non-linear multi-sensor interactions.
  - **Decision Tree**: Simple, transparent boolean decision rules for SCADA integration.
  - **Gradient Boosting**: Sequential residual minimization.
- **Unsupervised Anomaly Model**:
  - **Isolation Forest**: Isolates anomalies without requiring prior defect labels (zero-day leak detection).

---

### Slide 9: Model Evaluation Metrics Table
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Alarm Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **99.78%** | **99.27%** | **99.71%** | **99.48%** | **1.0000** | **0.20%** |
| **Gradient Boosting** | 99.75% | 99.26% | 99.56% | 99.41% | 0.9996 | 0.20% |
| **Decision Tree** | 99.69% | 99.12% | 99.41% | 99.26% | 0.9985 | 0.24% |
| **Isolation Forest** | 93.06% | 83.04% | 84.51% | 83.77% | 0.9776 | 4.36% |

---

### Slide 10: False Alarm Analysis & Mitigation
- **The False Alarm Dilemma**: Sending emergency repair crews into the field costs thousands of dollars per dispatch.
- **Random Forest Confusion Matrix on 3,200 Test Samples**:
  - Normal Flow Tested: 2,522 samples
  - Correct Normal (TN): 2,517 (99.80%)
  - **False Alarms (FP): Only 5 samples (0.20% False Alarm Rate!)**
  - Leaks Detected (TP): 676 of 678 (99.71% Recall)
- **Mitigation Strategy**: Rolling moving average temporal filter prevents transient water-hammer spikes from triggering premature alarms.

---

### Slide 11: Feature Importance & Physical Interpretation
- **Gini Impurity Reduction Rankings**:
  1. `flow_difference_lpm` (32.4%) — Primary violation of mass balance.
  2. `flow_loss_ratio` (24.8%) — Normalized leak fraction.
  3. `pressure_drop_rolling_mean` (14.2%) — Persistent hydraulic head loss.
  4. `total_pressure_drop` (11.6%) — Static differential across monitored nodes.
  5. `vibration_noise_db` (7.8%) — Acoustic jetting vibration.

---

### Slide 12: Streamlit SCADA Dashboard Features
- **Tab 1: Live Telemetry**: Dynamic multi-node pipeline schematic, real-time gauges, animated red alarm banner, audio siren trigger.
- **Tab 2: Manual Diagnostic Simulator**: Interactive sliders for what-if scenarios with real-time confidence scores and physical diagnostics.
- **Tab 3: Batch CSV Log Analyzer**: Ingests historical logs, flags leak timestamps, and estimates total lost water volume ($m^3$).
- **Tab 4: IoT Hardware Gateway**: Connects live physical microcontrollers over COM serial or runs virtual hardware node.
- **Tab 5 & 6: ML Analytics & Architecture**: Integrated ROC curves, confusion matrices, and hydraulic formulas.

---

### Slide 13: Sample Test Results
- **Normal Flow Test (`sample_normal_test.csv`)**: 200 records classified with 100% stability; zero false alarms triggered.
- **Leak Incident Transition (`sample_leakage_test.csv`)**: Correctly flagged transition from normal flow $\rightarrow$ pinhole leak (confidence 74%) $\rightarrow$ major burst (confidence 99.8%).
- **Transient Sensor Noise Challenge (`sample_sensor_noise_test.csv`)**: Model resisted momentary demand spikes without triggering false alarms.

---

### Slide 14: Future Enhancements & Industrial Roadmap
1. **LoRaWAN & NB-IoT**: Wireless low-power telemetry over 10–25 km transmission lines.
2. **Acoustic Wave Time-of-Flight**: Cross-correlating acoustic signal delay between sensors to pinpoint leak coordinates to within $\pm 1$ meter.
3. **Edge TinyML**: Deploying the quantized Decision Tree directly into the ESP32 firmware for standalone micro-controller inference.

---

### Slide 15: Conclusion & Q&A
- **Summary**: Delivered a complete, verified, cyber-physical pipeline leakage detection platform with 99.48% F1-score and 0.20% false alarm rate.
- **Deliverables**: Trained models, firmware sketch, hardware circuit schematics, dataset generator, and live Streamlit dashboard.
- **Thank you! Questions & Discussion**.
