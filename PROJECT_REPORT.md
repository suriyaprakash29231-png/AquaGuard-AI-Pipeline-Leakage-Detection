# AI-Based Water Pipeline Leakage Detection Using Multi-Sensor Telemetry
## Comprehensive Engineering Project Report & Academic Thesis Documentation

---

### **Executive Summary / Abstract**
Non-Revenue Water (NRW) caused by municipal and industrial pipeline leakages accounts for over 30–45% of treated fresh water loss globally, inflicting massive financial losses, structural sinkhole risks, and water contamination hazards. Traditional manual leak acoustic listening and periodic visual surveys are labor-intensive, reactive, and incapable of detecting localized incipient or micro-pinhole leaks before catastrophic pipeline ruptures occur.

This project delivers **AquaGuard AI**, an end-to-end cyber-physical, artificial intelligence-driven pipeline integrity monitoring and leakage detection system. By fusing hydraulic mass-balance conservation with real-time sensor telemetry—including inlet/outlet volumetric flow rates (YF-S201 turbine sensors), upstream, intermediate, and downstream static hydraulic pressure gradients (0–1.2 MPa transducers), and high-frequency acoustic/piezoelectric vibration signatures—the system detects and localizes fractures, orifice leaks, and bursts in real time. 

A high-performance **Random Forest Classifier** ensemble was engineered and trained alongside an interpretable **Decision Tree Classifier**, a **Gradient Boosting Classifier**, and an unsupervised **Isolation Forest Anomaly Detector**. The primary model achieved a **99.78% Classification Accuracy**, **99.27% Precision**, **99.71% Recall**, **99.48% F1-Score**, and **1.0000 ROC-AUC**, with a **False Alarm Rate (FPR) of only 0.20% (5 out of 2,522 normal operational test samples)**. A full-featured industrial SCADA dashboard was constructed using **Streamlit**, featuring real-time visual and auditory alert mechanisms, what-if diagnostic simulation, batch CSV log classification, and a hardware serial gateway interfacing with an **ESP32 / Arduino Uno** sensor node prototype.

---

## Chapter 1: Introduction & Problem Statement

### 1.1 Context and Background
Water distribution networks (WDNs) form critical infrastructure for human civilization. However, aging pipe infrastructure, soil shifts, corrosion, pressure surges (water hammer), and joint failures lead to persistent water loss. According to the World Bank, worldwide non-revenue water losses exceed 32 billion cubic meters per year, with an estimated global economic cost of over \$14 billion annually.

### 1.2 Limitations of Conventional Detection Techniques
1. **Acoustic Listening Rods & Geophones**: Require skilled human operators listening along the surface. Highly prone to ambient urban traffic noise and ineffective for non-metallic (PVC/HDPE) pipes.
2. **Ground Penetrating Radar (GPR)**: High capital expense, complex post-processing, and non-continuous coverage.
3. **Fixed Thresholding (Static Alarm Rules)**: Traditional SCADA systems rely on hardcoded flow or pressure thresholds. These trigger massive false alarm frequencies during normal domestic consumption spikes (e.g., morning showers, commercial peak hours).

### 1.3 Project Objectives
- Synthesize physically consistent hydraulic telemetry capturing normal diurnal flow, transient valve operations, minor pinhole leaks (1.5–4.5 LPM), moderate structural cracks (5–12 LPM), and catastrophic pipe bursts (>15 LPM).
- Engineer domain-specific hydraulic features that capture mass continuity violations, Darcy-Weisbach head loss, and hydraulic resistance variations.
- Train, evaluate, and benchmark multiple machine learning architectures (Random Forest, Decision Tree, Gradient Boosting, Isolation Forest).
- Perform in-depth **False Alarm Analysis** to optimize operational decision boundaries for municipal deployment.
- Develop an industrial-grade **Streamlit Dashboard** featuring real-time telemetry streaming, interactive what-if simulation, batch historical log analysis, and automated water loss volume estimation.
- Provide production-ready firmware (`.ino`) and a serial gateway for physical **Arduino Uno / ESP32** hardware sensor nodes.

---

## Chapter 2: Mathematical Hydraulics & Physical Principles

Pipeline flow is governed by fundamental fluid mechanics laws:

### 2.1 Conservation of Mass (The Continuity Equation)
In a closed, pressurized pipe section without intermediate legal off-takes:
$$Q_{in}(t) - Q_{out}(t) = Q_{leak}(t) + \epsilon_{measurement}(t)$$

Where:
- $Q_{in}(t)$ = Volumetric flow entering the monitored section at time $t$ ($\text{L/min}$).
- $Q_{out}(t)$ = Volumetric flow exiting to consumers ($\text{L/min}$).
- $Q_{leak}(t)$ = Volumetric discharge rate through the defect orifice.
- $\epsilon_{measurement}(t)$ = Gaussian white noise originating from sensor turbine fluctuations ($\sigma \approx 0.45\text{ LPM}$).

Under nominal conditions, $Q_{leak} = 0 \implies Q_{in} \approx Q_{out}$.

### 2.2 Orifice Discharge Dynamics (Torricelli's Law)
The fluid discharge rate through an orifice or fracture depends on internal gauge pressure:
$$Q_{leak} = C_d \cdot A_{orifice} \cdot \sqrt{\frac{2 \Delta P_{orifice}}{\rho}}$$
Where $C_d$ is the discharge coefficient ($0.60 - 0.65$ for circular orifices), $A_{orifice}$ is the rupture area, and $\Delta P$ is internal pressure relative to atmospheric ambient pressure.

### 2.3 Darcy-Weisbach Friction Head Loss
Friction along the inner wall causes pressure to drop across the pipeline length $L$:
$$h_f = f \cdot \frac{L}{D} \cdot \frac{v^2}{2g}$$
When a rupture occurs downstream, upstream velocity $v$ surges to feed both consumer demand and leak discharge, steepening the hydraulic gradient line (HGL) and depressing midpoint and downstream static pressure readings.

---

## Chapter 3: System Architecture & Hardware Prototype

```
+-------------------------------------------------------------------------------+
|                        PHYSICAL PIPELINE SENSOR NODE                          |
|                                                                               |
|   +-----------------------+     +-------------------+     +---------------+   |
|   | YF-S201 Flow Sensor 1 |     | Pressure Sensor P1|     | ESP32/Arduino |   |
|   | (Inlet Digital Pulse) |     | (Inlet 0-1.2 MPa) |     | Edge Micro    |   |
|   +-----------+-----------+     +---------+---------+     +-------+-------+   |
|               |                           |                       |           |
|               v                           v                       v           |
|      External Interrupt 0             ADC1 Ch 34             JSON Streamer    |
|       (GPIO 18 / Pin 2)           (Linear 0.5-4.5V)       (115200 Baud USB)   |
+-------------------------------------------+-----------------------------------+
                                            | JSON Packets over Serial
                                            v
+-------------------------------------------------------------------------------+
|                       PYTHON EDGE / SCADA INFERENCE LAYER                     |
|                                                                               |
|   +-----------------------+     +-------------------+     +---------------+   |
|   | Serial Gateway Reader | --> | Feature Engine    | --> | Pre-trained   |   |
|   | (pyserial bridge)     |     | (Hydraulics &     |     | Random Forest |   |
|   |                       |     |  Rolling Stats)   |     | Classifier    |   |
|   +-----------------------+     +-------------------+     +-------+-------+   |
+-------------------------------------------------------------------|-----------+
                                                                    | Probability
                                                                    v
+-------------------------------------------------------------------------------+
|                       STREAMLIT INTERACTIVE WEB SCADA                         |
|                                                                               |
|   - Real-Time Dynamic Gauges & Multi-Node Line Charts                         |
|   - Audio/Visual Red Warning Siren Trigger                                    |
|   - What-If Diagnostic Simulator & Decision Boundary Sliders                  |
|   - Batch CSV Historical Log Ingestion & Water Loss Estimator                 |
+-------------------------------------------------------------------------------+
```

### 3.1 Hardware Components
1. **YF-S201 Turbine Flow Sensor**:
   - Working Range: $1 - 30\text{ LPM}$ ($\pm 3\%$ accuracy)
   - Working Pressure: $< 1.75\text{ MPa}$
   - Output: Hall effect pulse frequency $f = 7.5 \cdot Q$.
2. **G1/4" Piezoresistive Pressure Transducer**:
   - Range: $0 - 1.2\text{ MPa}$ ($0 - 12\text{ Bar}$)
   - Output: Linear analog voltage $0.5\text{V}$ (0 bar) to $4.5\text{V}$ (12 bar).
3. **SW-420 / Piezoelectric Vibration Sensor**:
   - Captures acoustic jetting noise ($10 - 70\text{ dB}$) emitted by pressurized water escaping through a fractured pipe wall.
4. **Microcontroller (ESP32 DevKit V1 / Arduino Uno)**:
   - Configured with interrupt-driven pulse counting and multi-channel analog sampling.
   - Emits structured JSON frames at 1-second sample intervals over 115200 baud serial.

---

## Chapter 4: Data Engineering & Feature Extraction

### 4.1 Dataset Generation & Physical Consistency
To evaluate machine learning models under diverse operational contexts, a 16,000-record dataset was generated reflecting 5.5 days of continuous multi-node pipeline operation at 30-second sampling intervals:
- **Diurnal Municipal Demand Curve**: Modeled with dual peaks at 07:00–09:00 (morning domestic peak) and 19:00–21:00 (evening peak), with baseline demand oscillating between 8 and 50 LPM.
- **Incident Severities**:
  - *Normal Operation* (78.29%, 12,526 samples)
  - *Minor Pinhole Leak* (7.65%, 1,224 samples, leak rate 1.5–4.5 LPM)
  - *Moderate Crack* (9.04%, 1,446 samples, leak rate 5.0–12.0 LPM)
  - *Major Pipe Burst* (4.49%, 719 samples, leak rate 16.0–32.0 LPM)
  - *Transient Non-Leak Challenge* (0.53%, 85 samples, sharp consumer valve closure or pump transient to test false positive resistance)

### 4.2 Feature Engineering
From the raw sensor signals, 18 predictive features were engineered:
1. `flow_loss_ratio`: Relative mass balance violation $\frac{Q_{in} - Q_{out}}{Q_{in} + 10^{-4}}$.
2. `total_pressure_drop`: $\Delta P = P_{in} - P_{out}$ across the entire monitored segment.
3. `inlet_to_mid_drop` & `mid_to_outlet_drop`: Localized pressure gradients used to isolate leak location.
4. `hydraulic_resistance`: Apparent section friction index $\frac{\Delta P}{Q_{in}^2 + 10^{-4}}$.
5. `vib_acoustic_intensity`: Normalized acoustic noise level $\frac{\text{Noise (dB)}}{12.0\text{ dB baseline}}$.
6. `flow_diff_rolling_mean` & `flow_diff_rolling_std`: 5-step rolling window moving average and standard deviation to dampen instantaneous high-frequency sensor noise.
7. `pressure_drop_rolling_mean` & `pressure_drop_rolling_std`: Rolling metrics capturing persistent pressure drops versus temporary pump ripples.

---

## Chapter 5: Machine Learning Models & Training

Four diverse model paradigms were implemented and benchmarked on an 80/20 stratified split (12,800 train / 3,200 test samples):

1. **Random Forest Classifier**:
   - 120 estimators, max depth 12, minimum samples split 4.
   - Robust against non-linear sensor cross-correlations and feature scale discrepancies.
2. **Decision Tree Classifier**:
   - Max depth 6, minimum leaf size 5.
   - High interpretability; provides transparent conditional rules suitable for programmable logic controllers (PLCs).
3. **Gradient Boosting Classifier**:
   - 100 boosting stages, learning rate 0.08, max depth 5.
   - Sequentially corrects residuals of preceding weak learners.
4. **Isolation Forest (Unsupervised Anomaly Detector)**:
   - Contamination parameter set to 0.21.
   - Operates without labeled leak examples by measuring the average path length required to isolate a point in randomized feature partitions. Serves as a zero-day leak detector for unexpected structural failure modes.

---

## Chapter 6: Experimental Results & False Alarm Analysis

### 6.1 Performance Benchmark Summary

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Alarm Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.78%** | **99.27%** | **99.71%** | **99.48%** | **1.0000** | **0.20%** |
| **Gradient Boosting** | 99.75% | 99.26% | 99.56% | 99.41% | 0.9996 | 0.20% |
| **Decision Tree (Interpretable)** | 99.69% | 99.12% | 99.41% | 99.26% | 0.9985 | 0.24% |
| **Isolation Forest (Anomaly)** | 93.06% | 83.04% | 84.51% | 83.77% | 0.9776 | 4.36% |

### 6.2 Confusion Matrix (Random Forest on 3,200 Test Samples)
- **True Negatives (TN)**: 2,517 (Correct normal operations)
- **False Positives (FP - False Alarms)**: 5 (Only 0.20% false alert rate)
- **False Negatives (FN - Missed Leaks)**: 2 (0.29% missed leak rate)
- **True Positives (TP - Detected Leaks)**: 676 (99.71% sensitivity)

### 6.3 Feature Importance Analysis
Gini impurity reduction ranked the most decisive features as:
1. `flow_difference_lpm` (32.4%) — Primary direct indicator of mass continuity disruption.
2. `flow_loss_ratio` (24.8%) — Normalized leak fraction relative to total throughput.
3. `pressure_drop_rolling_mean` (14.2%) — Persistent hydraulic head loss.
4. `total_pressure_drop` (11.6%) — Static differential between upstream and downstream transducers.
5. `vibration_noise_db` (7.8%) — Acoustic signature of turbulent orifice ejection.

### 6.4 False Alarm Sensitivity vs Decision Threshold
In municipal water systems, sending repair crews to investigate false alarms wastes critical operational expenditure. 
- At threshold **0.30**: Recall is 99.85%, but False Alarms = 9.
- At default threshold **0.50**: Recall is 99.71%, with only **5 False Alarms** across 2,522 normal operational periods.
- At threshold **0.70**: Precision reaches 99.85%, with only **1 False Alarm** (FPR = 0.04%), while maintaining a high 98.96% recall.

---

## Chapter 7: Streamlit Monitoring Dashboard Features

The dashboard application (`app.py`) provides an industrial SCADA operator interface:
1. **Live Pipeline Telemetry**:
   - Dynamic SVG pipeline topology schematic showing nodes and sensor states.
   - Real-time gauge metrics for $Q_{in}, Q_{out}, \Delta Q, P_{in}, \Delta P$, and acoustic vibration.
   - Live streaming charts tracking flow balance and hydraulic gradients.
   - High-contrast visual alarm banner with animated CSS pulse and HTML5 Web Audio siren.
2. **Manual Diagnostic & What-If Analyzer**:
   - Interactive sliders allowing hydraulic engineers to simulate pump flutter, valve throttling, or burst scenarios and immediately inspect model confidence scores.
3. **Batch CSV Log Analyzer**:
   - Ingests field telemetry logs, calculates cumulative lost water volume ($m^3$ / Litres), flags incident timestamps, and exports annotated CSV audit reports.
4. **IoT & Hardware Gateway**:
   - Live serial port autodetector interfacing directly with USB-connected ESP32 / Arduino microcontrollers.
   - Interactive virtual hardware simulator enabling leak injection testing when physical hardware is not plugged in.
5. **Model Analytics & EDA Visualizer**:
   - Embeds pre-rendered confusion matrices, ROC curves, feature importance charts, and correlation heatmaps.

---

## Chapter 8: Conclusion & Future Work

### 8.1 Summary of Achievements
The **AquaGuard AI** project successfully demonstrates that combining hydraulic conservation laws with ensemble machine learning provides near-perfect leak detection accuracy (99.78%) while maintaining an ultra-low false alarm rate (0.20%). The system addresses both supervised scenarios (where historical defect data exists) and unsupervised zero-day anomalies via Isolation Forests.

### 8.2 Future Scope & Industrial Extensions
1. **Long-Range LoRaWAN / Cellular IoT**: Deploy battery-powered LoRaWAN sensor nodes across 10–50 km trunk distribution mains.
2. **Cross-Correlation Acoustic Ranging**: Fusing acoustic sensor arrival time differences ($\Delta t$) between two adjacent nodes to locate crack coordinates within $\pm 1.5$ meters.
3. **Edge TinyML Deployment**: Quantizing the Decision Tree or lightweight neural net into C++ byte arrays for execution directly inside the ESP32 microcontroller, eliminating cloud/PC dependency.
