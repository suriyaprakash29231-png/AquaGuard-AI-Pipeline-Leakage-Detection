"""
Model Training, Feature Engineering, and Evaluation Pipeline for Water Pipeline Leakage Detection
Includes:
- Physical and statistical feature engineering
- Model training: Random Forest, Decision Tree, Isolation Forest, Gradient Boosting
- Exhaustive evaluation: Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
- In-depth False-Alarm Analysis
- Plot generation (Matplotlib / Seaborn) and Model Serialization
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

def engineer_features(df):
    """
    Computes hydraulic and statistical features for water pipeline leakage identification.
    """
    df = df.copy()
    
    # 1. Flow mass balance violation
    df["flow_diff_abs"] = np.abs(df["inlet_flow_lpm"] - df["outlet_flow_lpm"])
    df["flow_loss_ratio"] = (df["inlet_flow_lpm"] - df["outlet_flow_lpm"]) / (df["inlet_flow_lpm"] + 1e-4)
    
    # 2. Hydraulic gradients
    df["total_pressure_drop"] = df["inlet_pressure_bar"] - df["outlet_pressure_bar"]
    df["inlet_to_mid_drop"] = df["inlet_pressure_bar"] - df["mid_pressure_bar"]
    df["mid_to_outlet_drop"] = df["mid_pressure_bar"] - df["outlet_pressure_bar"]
    
    # Hydraulic resistance index: Head loss relative to kinetic energy (Darcy-Weisbach / Orifice analogy)
    df["hydraulic_resistance"] = df["total_pressure_drop"] / ((df["inlet_flow_lpm"] / 60.0)**2 + 1e-4)
    
    # 3. Acoustic/Vibration anomaly index relative to expected baseline
    df["vib_acoustic_intensity"] = df["vibration_noise_db"] / 12.0 # 12 dB is baseline ambient
    
    # 4. Rolling statistical features (short window = 5 time-steps ~ 2.5 minutes)
    df["flow_diff_rolling_mean"] = df["flow_difference_lpm"].rolling(window=5, min_periods=1).mean()
    df["flow_diff_rolling_std"] = df["flow_difference_lpm"].rolling(window=5, min_periods=1).std().fillna(0)
    df["pressure_drop_rolling_mean"] = df["total_pressure_drop"].rolling(window=5, min_periods=1).mean()
    df["pressure_drop_rolling_std"] = df["total_pressure_drop"].rolling(window=5, min_periods=1).std().fillna(0)
    
    # Fill any remaining NaNs
    df = df.bfill().ffill()
    return df

def train_and_evaluate():
    data_path = os.path.join("data", "pipeline_leakage_dataset.csv")
    results_dir = os.path.join("models", "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    raw_df = pd.read_csv(data_path)
    df = engineer_features(raw_df)
    
    # Feature set selection
    feature_cols = [
        "inlet_flow_lpm",
        "outlet_flow_lpm",
        "flow_difference_lpm",
        "flow_loss_ratio",
        "inlet_pressure_bar",
        "mid_pressure_bar",
        "outlet_pressure_bar",
        "total_pressure_drop",
        "inlet_to_mid_drop",
        "mid_to_outlet_drop",
        "hydraulic_resistance",
        "water_temperature_c",
        "vibration_noise_db",
        "vib_acoustic_intensity",
        "flow_diff_rolling_mean",
        "flow_diff_rolling_std",
        "pressure_drop_rolling_mean",
        "pressure_drop_rolling_std"
    ]
    
    X = df[feature_cols]
    y = df["leak_detected"].values
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Dataset shape: {X.shape}, Leak prevalence: {np.mean(y)*100:.2f}%")
    
    # Train-Test Split (stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ----------------------------------------------------
    # 1. Random Forest Classifier (Primary Model)
    # ----------------------------------------------------
    print("\n--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # ----------------------------------------------------
    # 2. Decision Tree Classifier (Interpretable Model)
    # ----------------------------------------------------
    print("--- Training Decision Tree Classifier ---")
    dt_model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
    
    # ----------------------------------------------------
    # 3. Gradient Boosting Classifier
    # ----------------------------------------------------
    print("--- Training Gradient Boosting Classifier ---")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
    
    # ----------------------------------------------------
    # 4. Isolation Forest (Unsupervised Anomaly Detection)
    # ----------------------------------------------------
    print("--- Training Isolation Forest (Anomaly Detection) ---")
    # contamination set close to empirical leak rate
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.21,
        random_state=42,
        n_jobs=-1
    )
    # Fit on training set
    iso_model.fit(X_train_scaled)
    # Isolation forest outputs -1 for anomaly, 1 for inlier
    iso_raw_pred = iso_model.predict(X_test_scaled)
    y_pred_iso = np.where(iso_raw_pred == -1, 1, 0)
    # Anomaly score: higher negative score -> higher anomaly
    iso_scores = -iso_model.decision_function(X_test_scaled)
    # Normalize to 0-1 range for probability-like comparison
    iso_prob = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6)
    
    # ----------------------------------------------------
    # Model Comparison Metrics Table
    # ----------------------------------------------------
    models_dict = {
        "Random Forest": (y_pred_rf, y_prob_rf),
        "Decision Tree": (y_pred_dt, y_prob_dt),
        "Gradient Boosting": (y_pred_gb, y_prob_gb),
        "Isolation Forest": (y_pred_iso, iso_prob)
    }
    
    metrics_summary = {}
    print("\n================ MODEL PERFORMANCE COMPARISON ================")
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)
    
    for name, (pred, prob) in models_dict.items():
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, prob)
        
        # Calculate False Alarm Rate (False Positive Rate = FP / (FP + TN))
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_discovery_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        
        metrics_summary[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4),
            "False_Alarm_Rate_FPR": round(float(fpr), 4),
            "False_Discovery_Rate": round(float(false_discovery_rate), 4),
            "Confusion_Matrix": {
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
            }
        }
        print(f"{name:<20} | {acc:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1:.4f}     | {auc:.4f}")
    
    # ----------------------------------------------------
    # Detailed False Alarm Analysis
    # ----------------------------------------------------
    print("\n--- In-Depth False Alarm Analysis (Random Forest) ---")
    tn, fp, fn, tp = metrics_summary["Random Forest"]["Confusion_Matrix"].values()
    total_normal_tested = tn + fp
    total_leaks_tested = tp + fn
    print(f"Total Normal Samples in Test Set: {total_normal_tested}")
    print(f"Correctly Classified as Normal (TN): {tn} ({(tn/total_normal_tested)*100:.2f}%)")
    print(f"False Alarms (FP): {fp} ({(fp/total_normal_tested)*100:.2f}%)")
    print(f"Actual Leaks Detected (TP): {tp} ({(tp/total_leaks_tested)*100:.2f}%)")
    print(f"Missed Leaks (FN): {fn} ({(fn/total_leaks_tested)*100:.2f}%)")
    
    # Threshold Tuning for False Alarm Mitigation:
    thresholds = np.linspace(0.1, 0.9, 17)
    threshold_analysis = []
    for th in thresholds:
        th_pred = (y_prob_rf >= th).astype(int)
        th_cm = confusion_matrix(y_test, th_pred)
        t_tn, t_fp, t_fn, t_tp = th_cm.ravel()
        t_fpr = t_fp / (t_fp + t_tn)
        t_prec = precision_score(y_test, th_pred, zero_division=0)
        t_rec = recall_score(y_test, th_pred, zero_division=0)
        t_f1 = f1_score(y_test, th_pred, zero_division=0)
        threshold_analysis.append({
            "threshold": round(float(th), 2),
            "false_alarm_rate_fpr": round(float(t_fpr), 4),
            "precision": round(float(t_prec), 4),
            "recall": round(float(t_rec), 4),
            "f1": round(float(t_f1), 4),
            "false_alarms_count": int(t_fp),
            "missed_leaks_count": int(t_fn)
        })
    
    # ----------------------------------------------------
    # PLOT 1: Confusion Matrices
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Normal Flow", "Leak Detected"],
                yticklabels=["Normal Flow", "Leak Detected"], cbar=False)
    axes[0].set_title(f"Random Forest Confusion Matrix\nPrecision: {metrics_summary['Random Forest']['Precision']:.3f} | Recall: {metrics_summary['Random Forest']['Recall']:.3f}", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("True Status", fontsize=11)
    axes[0].set_xlabel("Predicted Status", fontsize=11)
    
    # Isolation Forest Confusion Matrix
    cm_iso = confusion_matrix(y_test, y_pred_iso)
    sns.heatmap(cm_iso, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                xticklabels=["Normal Flow", "Leak Detected"],
                yticklabels=["Normal Flow", "Leak Detected"], cbar=False)
    axes[1].set_title(f"Isolation Forest (Unsupervised) Confusion Matrix\nPrecision: {metrics_summary['Isolation Forest']['Precision']:.3f} | Recall: {metrics_summary['Isolation Forest']['Recall']:.3f}", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("True Status", fontsize=11)
    axes[1].set_xlabel("Predicted Status", fontsize=11)
    
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_plot_path}")
    
    # ----------------------------------------------------
    # PLOT 2: ROC Curves
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    for name, (_, prob) in models_dict.items():
        fpr_curve, tpr_curve, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        plt.plot(fpr_curve, tpr_curve, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Baseline (AUC = 0.500)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (False Alarm Rate)", fontsize=11)
    plt.ylabel("True Positive Rate (Leak Recall)", fontsize=11)
    plt.title("ROC Curves - Pipeline Leakage Detection Models", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_plot_path}")
    
    # ----------------------------------------------------
    # PLOT 3: Feature Importance (Random Forest)
    # ----------------------------------------------------
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(indices)), sorted_importances[::-1], color="#2b5c8f", edgecolor="#1a365d")
    plt.yticks(range(len(indices)), sorted_features[::-1], fontsize=10)
    plt.xlabel("Relative Feature Importance (Gini Impurity Reduction)", fontsize=11)
    plt.title("Random Forest: Pipeline Leakage Feature Importance", fontsize=13, fontweight='bold')
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%",
                 ha="left", va="center", fontsize=9, color="#1a365d", fontweight="bold")
        
    plt.xlim(0, max(sorted_importances) * 1.15)
    plt.tight_layout()
    fi_plot_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {fi_plot_path}")
    
    # ----------------------------------------------------
    # PLOT 4: False Alarm Analysis vs Decision Threshold
    # ----------------------------------------------------
    th_df = pd.DataFrame(threshold_analysis)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    
    color = '#1f77b4'
    ax1.set_xlabel('Classification Decision Threshold', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Metric Score (0 - 1.0)', color=color, fontsize=11, fontweight='bold')
    l1, = ax1.plot(th_df['threshold'], th_df['precision'], marker='o', color='#2ca02c', label='Precision (Low False Alarms)', lw=2)
    l2, = ax1.plot(th_df['threshold'], th_df['recall'], marker='s', color='#d62728', label='Recall (Leak Sensitivity)', lw=2)
    l3, = ax1.plot(th_df['threshold'], th_df['f1'], marker='^', color='#1f77b4', label='F1-Score', lw=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Secondary axis for False Alarm Count
    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Number of False Alarms in Test Set (Count)', color=color, fontsize=11, fontweight='bold')
    l4, = ax2.plot(th_df['threshold'], th_df['false_alarms_count'], marker='x', color=color, linestyle='--', lw=2, label='False Alarm Count')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight recommended optimal threshold (0.50)
    ax1.axvline(x=0.50, color='purple', linestyle=':', lw=2, label='Default Threshold (0.50)')
    
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.1, 0.45), framealpha=0.9)
    plt.title("False Alarm Trade-off Analysis Across Decision Thresholds", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fa_plot_path = os.path.join(results_dir, "false_alarm_analysis.png")
    plt.savefig(fa_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {fa_plot_path}")
    
    # ----------------------------------------------------
    # PLOT 5: Model Comparison Bar Chart
    # ----------------------------------------------------
    comp_df = pd.DataFrame([
        {"Model": k, "F1-Score": v["F1_Score"], "Precision": v["Precision"], "Recall": v["Recall"], "ROC-AUC": v["ROC_AUC"]}
        for k, v in metrics_summary.items()
    ])
    melted = pd.melt(comp_df, id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="crest")
    plt.ylim(0, 1.1)
    plt.title("Algorithm Comparison: Supervised vs Anomaly Detection", fontsize=13, fontweight='bold')
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.ylabel("Score", fontsize=11)
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_plot_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_plot_path}")
    
    # ----------------------------------------------------
    # Save Models and Artifacts
    # ----------------------------------------------------
    joblib.dump(rf_model, os.path.join("models", "best_model.pkl"))
    joblib.dump(dt_model, os.path.join("models", "decision_tree_model.pkl"))
    joblib.dump(gb_model, os.path.join("models", "gradient_boosting_model.pkl"))
    joblib.dump(iso_model, os.path.join("models", "isolation_forest_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    print("\nSaved all trained models to 'models/' directory.")
    
    # Save metadata JSON for the dashboard and reports
    metadata = {
        "features": feature_cols,
        "n_features": len(feature_cols),
        "feature_importances": {feat: float(imp) for feat, imp in zip(feature_cols, rf_model.feature_importances_)},
        "metrics": metrics_summary,
        "threshold_analysis": threshold_analysis,
        "optimal_threshold": 0.50,
        "test_samples_count": len(y_test),
        "total_dataset_rows": len(df)
    }
    
    meta_path = os.path.join("models", "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {meta_path}")
    
    return metadata

if __name__ == "__main__":
    train_and_evaluate()
