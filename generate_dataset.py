"""
Water Pipeline Leakage Dataset Generator
Physics-informed synthetic dataset generator simulating realistic pipeline hydraulic behavior:
- Conservation of mass: Inlet flow - Outlet flow = Leakage rate + noise
- Pressure dynamics: Darcy-Weisbach head loss, sudden pressure drop during burst or orifice leak
- Diurnal water consumption demand cycles (morning and evening domestic peaks)
- Operational variations: Normal fluctuations, minor pinhole leaks, moderate cracks, major pipe bursts,
  and sensor noise / valve transient events to test false alarm resistance.
"""

import os
import numpy as np
import pandas as pd

def generate_pipeline_data(n_samples=16000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Time indexing (1 reading every 30 seconds -> ~5.5 days of continuous monitoring)
    start_time = pd.Timestamp("2026-03-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(seconds=30 * i) for i in range(n_samples)]
    timestamps = pd.DatetimeIndex(timestamps)
    
    # 2. Time-of-day diurnal factor (Normal municipal water demand cycle)
    # Peak at 07:00 - 09:00 (morning) and 18:00 - 21:00 (evening), lowest at 02:00 - 04:00
    hours = timestamps.hour + timestamps.minute / 60.0
    diurnal_demand = (
        25.0
        + 12.0 * np.sin(2 * np.pi * (hours - 5.5) / 24)
        + 8.0 * np.sin(4 * np.pi * (hours - 8.0) / 24)
        + np.random.normal(0, 1.2, n_samples)
    )
    diurnal_demand = np.clip(diurnal_demand, 8.0, 50.0) # Litres per minute baseline consumption
    
    # 3. Baseline Hydraulic Parameters
    # Inlet base pressure: 4.5 bar, fluctuates inversely with demand (high demand -> slight pump pressure drop)
    base_inlet_pressure = 4.8 - 0.02 * diurnal_demand + np.random.normal(0, 0.06, n_samples)
    base_inlet_pressure = np.clip(base_inlet_pressure, 3.5, 5.5)
    
    # Water temperature (slight daily cycle between 20°C and 28°C)
    water_temp = 23.0 + 3.5 * np.sin(2 * np.pi * (hours - 10) / 24) + np.random.normal(0, 0.4, n_samples)
    
    # 4. Generate States and Events
    # States:
    # 0 = Normal Operation (~68%)
    # 1 = Minor Pinhole Leak (~12%)
    # 2 = Moderate Crack Leak (~11%)
    # 3 = Major Pipe Burst (~6%)
    # 4 = Operational Transient / Valve Operation / High Demand Spurt (~3% - False alarm challenge)
    
    leak_status = np.zeros(n_samples, dtype=int)
    leak_severity = np.array(["None"] * n_samples, dtype=object)
    leak_rate = np.zeros(n_samples)
    vibration_noise = np.random.normal(12.0, 1.5, n_samples) # dB ambient sensor noise
    
    # We create realistic episodic blocks (leaks persist over multiple time steps, rather than random single steps)
    i = 0
    while i < n_samples:
        event_roll = np.random.rand()
        
        if event_roll < 0.68:
            # Normal operation segment (duration: 30 to 120 steps)
            duration = min(np.random.randint(40, 150), n_samples - i)
            leak_status[i:i+duration] = 0
            leak_severity[i:i+duration] = "None"
            leak_rate[i:i+duration] = 0.0
            vibration_noise[i:i+duration] += np.random.normal(0, 0.8, duration)
            i += duration
            
        elif event_roll < 0.80:
            # Minor Pinhole Leak (leak rate ~ 1.5 - 4.0 LPM, subtle pressure drop ~ 0.15 - 0.35 bar)
            duration = min(np.random.randint(30, 90), n_samples - i)
            leak_status[i:i+duration] = 1
            leak_severity[i:i+duration] = "Minor Leak"
            # Leak develops with a slight ramp
            l_rate = np.linspace(1.2, np.random.uniform(2.5, 4.5), duration)
            leak_rate[i:i+duration] = l_rate
            vibration_noise[i:i+duration] += np.random.uniform(6.0, 12.0) + np.random.normal(0, 1.0, duration)
            i += duration
            
        elif event_roll < 0.91:
            # Moderate Crack Leak (leak rate ~ 5.0 - 12.0 LPM, clear pressure drop ~ 0.4 - 0.9 bar)
            duration = min(np.random.randint(25, 80), n_samples - i)
            leak_status[i:i+duration] = 1
            leak_severity[i:i+duration] = "Moderate Leak"
            l_rate = np.random.uniform(6.0, 12.0, duration) + np.random.normal(0, 0.5, duration)
            leak_rate[i:i+duration] = np.clip(l_rate, 4.5, 15.0)
            vibration_noise[i:i+duration] += np.random.uniform(14.0, 24.0) + np.random.normal(0, 1.2, duration)
            i += duration
            
        elif event_roll < 0.97:
            # Major Pipe Burst (catastrophic: leak rate ~ 16.0 - 32.0 LPM, sudden pressure drop > 1.2 bar)
            duration = min(np.random.randint(20, 60), n_samples - i)
            leak_status[i:i+duration] = 1
            leak_severity[i:i+duration] = "Major Burst"
            l_rate = np.random.uniform(18.0, 30.0, duration) + np.random.normal(0, 1.0, duration)
            leak_rate[i:i+duration] = np.clip(l_rate, 15.0, 35.0)
            vibration_noise[i:i+duration] += np.random.uniform(25.0, 42.0) + np.random.normal(0, 2.0, duration)
            i += duration
            
        else:
            # Transient spike / Sudden consumer valve shut or pump flutter (Non-leak anomaly)
            duration = min(np.random.randint(5, 20), n_samples - i)
            leak_status[i:i+duration] = 0 # NOT A LEAK, tests model false positive resistance
            leak_severity[i:i+duration] = "Transient Non-Leak"
            leak_rate[i:i+duration] = 0.0
            # Higher ambient noise but conservation of mass holds
            vibration_noise[i:i+duration] += np.random.uniform(8.0, 15.0)
            i += duration
            
    # 5. Apply Hydraulic Laws
    # Continuity: Inlet Flow = Demand + Leak Rate + Sensor Noise
    inlet_flow = diurnal_demand + leak_rate + np.random.normal(0, 0.45, n_samples)
    # Outlet Flow = Demand (what reaches consumer) + Sensor Noise
    outlet_flow = diurnal_demand + np.random.normal(0, 0.45, n_samples)
    
    # Pipe friction pressure drop (Hazen-Williams / Darcy proportional to Flow^1.85)
    friction_factor = 0.012
    normal_pressure_drop = friction_factor * (inlet_flow ** 1.35)
    
    # Additional pressure loss caused by leakage orifice outflow (Torricelli's law / hydraulic drop)
    leak_pressure_drop = 0.08 * (leak_rate ** 1.1)
    
    inlet_pressure = base_inlet_pressure
    mid_pressure = inlet_pressure - (normal_pressure_drop * 0.45) - (leak_pressure_drop * 0.6) - np.random.normal(0, 0.04, n_samples)
    outlet_pressure = inlet_pressure - normal_pressure_drop - leak_pressure_drop - np.random.normal(0, 0.05, n_samples)
    
    # Flow difference: Q_in - Q_out
    flow_difference = inlet_flow - outlet_flow
    total_pressure_drop = inlet_pressure - outlet_pressure
    
    # Clip realistic physical values
    inlet_flow = np.clip(inlet_flow, 1.0, 100.0)
    outlet_flow = np.clip(outlet_flow, 0.5, 95.0)
    flow_difference = inlet_flow - outlet_flow
    inlet_pressure = np.clip(inlet_pressure, 1.5, 7.0)
    mid_pressure = np.clip(mid_pressure, 0.8, inlet_pressure - 0.05)
    outlet_pressure = np.clip(outlet_pressure, 0.5, mid_pressure - 0.05)
    total_pressure_drop = inlet_pressure - outlet_pressure
    vibration_noise = np.clip(vibration_noise, 5.0, 80.0)
    
    # Binary classification target (0 = Normal, 1 = Leakage)
    # Binary leak label ensures strict evaluation metrics
    df = pd.DataFrame({
        "timestamp": timestamps,
        "inlet_flow_lpm": np.round(inlet_flow, 2),
        "outlet_flow_lpm": np.round(outlet_flow, 2),
        "flow_difference_lpm": np.round(flow_difference, 2),
        "inlet_pressure_bar": np.round(inlet_pressure, 3),
        "mid_pressure_bar": np.round(mid_pressure, 3),
        "outlet_pressure_bar": np.round(outlet_pressure, 3),
        "pressure_drop_bar": np.round(total_pressure_drop, 3),
        "water_temperature_c": np.round(water_temp, 1),
        "vibration_noise_db": np.round(vibration_noise, 1),
        "leak_severity": leak_severity,
        "leak_rate_lpm": np.round(leak_rate, 2),
        "leak_detected": leak_status
    })
    
    return df

def save_datasets(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating comprehensive water pipeline dataset (16,000 records)...")
    full_df = generate_pipeline_data(n_samples=16000, random_seed=42)
    
    # Full dataset path
    full_path = os.path.join(output_dir, "pipeline_leakage_dataset.csv")
    full_df.to_csv(full_path, index=False)
    print(f"Saved primary dataset to: {full_path}")
    print(f"Total rows: {len(full_df)}")
    print(f"Class distribution:\n{full_df['leak_detected'].value_counts(normalize=True)}")
    print(f"Severity breakdown:\n{full_df['leak_severity'].value_counts()}")
    
    # Sample Test Sets for Interactive Dashboard Evaluation:
    # 1. Normal Operation Sample (300 consecutive records with no leaks)
    normal_indices = full_df[full_df["leak_severity"] == "None"].index
    # Find a contiguous slice of normal data
    start_idx = normal_indices[500]
    sample_normal = full_df.iloc[start_idx:start_idx+200].copy()
    sample_normal_path = os.path.join(output_dir, "sample_normal_test.csv")
    sample_normal.to_csv(sample_normal_path, index=False)
    print(f"Saved sample normal test file to: {sample_normal_path}")
    
    # 2. Leakage Incident Sample (200 records transitioning from Normal -> Minor -> Major Burst)
    burst_indices = full_df[full_df["leak_severity"] == "Major Burst"].index
    if len(burst_indices) > 0:
        burst_idx = burst_indices[0]
        # Take 50 rows before the burst and 150 rows during/after
        incident_start = max(0, burst_idx - 50)
        sample_leak = full_df.iloc[incident_start:incident_start+200].copy()
    else:
        sample_leak = full_df[full_df["leak_detected"] == 1].head(200).copy()
        
    sample_leak_path = os.path.join(output_dir, "sample_leakage_test.csv")
    sample_leak.to_csv(sample_leak_path, index=False)
    print(f"Saved sample leakage test file to: {sample_leak_path}")
    
    # 3. Sensor Noise & Transient Spike Sample (Tests robustness against false alarms)
    transient_indices = full_df[full_df["leak_severity"] == "Transient Non-Leak"].index
    if len(transient_indices) > 0:
        t_idx = transient_indices[0]
        sample_noise = full_df.iloc[max(0, t_idx-30):t_idx+120].copy()
    else:
        sample_noise = full_df.sample(n=150, random_state=101).copy()
    sample_noise_path = os.path.join(output_dir, "sample_sensor_noise_test.csv")
    sample_noise.to_csv(sample_noise_path, index=False)
    print(f"Saved sample noise/transient test file to: {sample_noise_path}")

if __name__ == "__main__":
    save_datasets()
