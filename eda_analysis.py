"""
Exploratory Data Analysis (EDA) for Water Pipeline Sensor Dataset
Generates comprehensive visual insights into hydraulic features:
- Sensor distributions across normal vs leakage states
- Correlation matrix of pipeline parameters
- Hydraulic pressure drop vs flow differential dynamics
- Diurnal demand fluctuation patterns
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_eda(data_path="data/pipeline_leakage_dataset.csv", output_dir="models/evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = [
        "inlet_flow_lpm", "outlet_flow_lpm", "flow_difference_lpm",
        "inlet_pressure_bar", "mid_pressure_bar", "outlet_pressure_bar",
        "pressure_drop_bar", "water_temperature_c", "vibration_noise_db",
        "leak_detected"
    ]
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Hydraulic Sensor Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")
    
    # 2. Hydraulic Signature: Flow Difference vs Pressure Drop by Leak Severity
    plt.figure(figsize=(10, 6))
    palette = {
        "None": "#2b8cbe",
        "Minor Leak": "#fd8d3c",
        "Moderate Leak": "#e6550d",
        "Major Burst": "#bd0026",
        "Transient Non-Leak": "#756bb1"
    }
    sample_df = df.sample(n=min(3000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x="flow_difference_lpm",
        y="pressure_drop_bar",
        hue="leak_severity",
        palette=palette,
        alpha=0.7,
        s=45
    )
    plt.title("Hydraulic Characteristic Signature: Flow Difference vs Pressure Drop", fontsize=13, fontweight="bold")
    plt.xlabel("Flow Differential ($Q_{in} - Q_{out}$) [LPM]", fontsize=11)
    plt.ylabel("Total Pressure Drop ($\Delta P$) [Bar]", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Pipeline State", loc="upper left")
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "hydraulic_signature_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")

    # 3. Sensor Value Distributions (Normal vs Leakage)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    sns.kdeplot(data=df, x="flow_difference_lpm", hue="leak_detected", common_norm=False, fill=True, palette=["#2b8cbe", "#de2d26"], ax=axes[0, 0])
    axes[0, 0].set_title("Distribution: Flow Difference (LPM)", fontweight="bold")
    axes[0, 0].legend(["Leak Detected", "Normal Flow"])
    
    sns.kdeplot(data=df, x="pressure_drop_bar", hue="leak_detected", common_norm=False, fill=True, palette=["#2b8cbe", "#de2d26"], ax=axes[0, 1])
    axes[0, 1].set_title("Distribution: Pressure Drop (Bar)", fontweight="bold")
    axes[0, 1].legend(["Leak Detected", "Normal Flow"])
    
    sns.kdeplot(data=df, x="inlet_pressure_bar", hue="leak_detected", common_norm=False, fill=True, palette=["#2b8cbe", "#de2d26"], ax=axes[1, 0])
    axes[1, 0].set_title("Distribution: Inlet Pressure (Bar)", fontweight="bold")
    axes[1, 0].legend(["Leak Detected", "Normal Flow"])
    
    sns.kdeplot(data=df, x="vibration_noise_db", hue="leak_detected", common_norm=False, fill=True, palette=["#2b8cbe", "#de2d26"], ax=axes[1, 1])
    axes[1, 1].set_title("Distribution: Acoustic / Vibration Noise (dB)", fontweight="bold")
    axes[1, 1].legend(["Leak Detected", "Normal Flow"])
    
    plt.suptitle("Sensor Telemetry Probability Densities: Normal Flow vs Leak Condition", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "sensor_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

if __name__ == "__main__":
    run_eda()
