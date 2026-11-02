"""
Serial Gateway & Hardware Bridge for Water Pipeline Leakage Detection
Allows reading real ESP32 / Arduino sensor data via USB Serial,
or running an interactive virtual sensor simulator when hardware is offline.
"""

import time
import json
import random
import serial
import serial.tools.list_ports
import pandas as pd
import numpy as np

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]

class SensorHardwareBridge:
    def __init__(self, port=None, baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.simulated = False
        self.sim_leak_mode = False
        
    def connect(self):
        if self.port and self.port != "Simulated Node (No COM Port)":
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
                time.sleep(1.5) # Allow Arduino reset
                self.simulated = False
                return True, f"Connected to physical sensor node on {self.port}"
            except Exception as e:
                self.simulated = True
                return False, f"Could not open {self.port}: {str(e)}. Switched to Virtual Node."
        else:
            self.simulated = True
            return True, "Running in Virtual Sensor Node Simulation Mode."

    def set_leak_simulation(self, leak_state: bool, severity="Minor"):
        self.sim_leak_mode = leak_state
        self.sim_severity = severity

    def read_packet(self):
        if not self.simulated and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    return data
            except Exception:
                pass
        
        # Virtual Simulation Mode
        base_demand = 28.0 + random.uniform(-1.5, 1.5)
        inlet_pressure = 4.6 + random.uniform(-0.05, 0.05)
        
        if self.sim_leak_mode:
            if getattr(self, 'sim_severity', 'Moderate') == 'Minor':
                leak_flow = random.uniform(2.5, 4.0)
                p_drop = random.uniform(0.35, 0.55)
                vib = random.uniform(18.0, 24.0)
            elif getattr(self, 'sim_severity', 'Moderate') == 'Major':
                leak_flow = random.uniform(18.0, 26.0)
                p_drop = random.uniform(1.2, 1.8)
                vib = random.uniform(35.0, 50.0)
            else: # Moderate
                leak_flow = random.uniform(7.0, 11.0)
                p_drop = random.uniform(0.65, 0.95)
                vib = random.uniform(24.0, 32.0)
                
            inlet_flow = base_demand + leak_flow
            outlet_flow = base_demand
            outlet_pressure = max(0.8, inlet_pressure - p_drop)
            mid_pressure = inlet_pressure - (p_drop * 0.6)
        else:
            # Normal flow
            inlet_flow = base_demand + random.uniform(-0.25, 0.25)
            outlet_flow = base_demand + random.uniform(-0.25, 0.25)
            p_drop = 0.12 + random.uniform(-0.02, 0.02)
            outlet_pressure = inlet_pressure - p_drop
            mid_pressure = inlet_pressure - (p_drop * 0.45)
            vib = random.uniform(10.0, 13.5)
            
        packet = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inlet_flow_lpm": round(inlet_flow, 2),
            "outlet_flow_lpm": round(outlet_flow, 2),
            "flow_difference_lpm": round(inlet_flow - outlet_flow, 2),
            "inlet_pressure_bar": round(inlet_pressure, 3),
            "mid_pressure_bar": round(mid_pressure, 3),
            "outlet_pressure_bar": round(outlet_pressure, 3),
            "pressure_drop_bar": round(inlet_pressure - outlet_pressure, 3),
            "water_temperature_c": round(23.5 + random.uniform(-0.2, 0.2), 1),
            "vibration_noise_db": round(vib, 1),
            "is_virtual": self.simulated
        }
        return packet

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
