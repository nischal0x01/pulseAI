"""
Analyze predictions for a specific patient.
Shows actual vs predicted BP with error metrics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import glob
import h5py
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

import tensorflow as tf
from preprocessing import preprocess_signals, create_subject_wise_splits
from feature_engineering import extract_physiological_features, standardize_feature_length, normalize_pat_subject_wise, create_4_channel_input
from utils import normalize_data

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Pearson correlation
    if len(y_true) > 1:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        pearson = 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Pearson': pearson,
        'Mean Error': np.mean(y_pred - y_true),
        'Std Error': np.std(y_pred - y_true)
    }


def analyze_patient(patient_id=None, checkpoint_path='checkpoints/best_model.h5'):
    """Analyze predictions for a specific patient."""
    
    print("="*70)
    print("  PATIENT-SPECIFIC ANALYSIS")
    print("="*70)
    
    # Load model
    print(f"\n📦 Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        return
    
    # Load with safe_mode=False to allow Lambda layers
    try:
        model = tf.keras.models.load_model(checkpoint_path, safe_mode=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Try loading with compile=False...")
        try:
            model = tf.keras.models.load_model(checkpoint_path, compile=False, safe_mode=False)
            print("✅ Model loaded successfully (without compilation)")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            return
    
    # Load data
    print("\n📊 Loading data...")
    data_dir = 'data/processed'
    
    # If patient_id is not specified, list available patients and pick one
    if patient_id is None:
        # Just list available patient files
        import glob
        patient_files = sorted(glob.glob(os.path.join(data_dir, 'p*.mat')))
        if not patient_files:
            print(f"❌ No patient data found in {data_dir}")
            return
        unique_patients = [Path(f).stem for f in patient_files]
        print(f"   - Available patients: {', '.join(unique_patients[:5])}...")
        patient_id = unique_patients[-1]  # Use the last one
    
    print(f"\n🔍 Analyzing patient: {patient_id}")
    
    # Load only this patient's data using the same function from data_loader
    from data_loader import _read_subj_wins
    patient_file = os.path.join(data_dir, f"{patient_id}.mat")
    if not os.path.exists(patient_file):
        print(f"❌ Patient file not found: {patient_file}")
        return
    
    print(f"   - Loading single patient file: {patient_file}")
    try:
        signals_agg, y_sbp, y_dbp, demographics = _read_subj_wins(patient_file)
        n_samples = len(y_sbp)
        print(f"   - Samples: {n_samples}")
    except Exception as e:
        print(f"❌ Failed to load patient data: {e}")
        return
    
    # Preprocess
    print("\n🔄 Preprocessing signals...")
    processed_signals = preprocess_signals(signals_agg)
    
    # Extract features
    print("\n💓 Extracting physiological features...")
    ppg_raw = signals_agg[:, 0, :]
    ecg_raw = signals_agg[:, 1, :]
    pat_seqs, hr_seqs, quality_mask = extract_physiological_features(ecg_raw, ppg_raw)
    
    # Standardize
    target_len = processed_signals.shape[1]
    pat_seqs = standardize_feature_length(pat_seqs, target_len)
    hr_seqs = standardize_feature_length(hr_seqs, target_len)
    
    # Create patient_ids array for this single patient
    patient_ids = np.array([patient_id] * n_samples)
    
    # Create train mask (we'll normalize using all data for simplicity in analysis)
    train_mask = np.ones(n_samples, dtype=bool)
    pat_seqs_scaled, _ = normalize_pat_subject_wise(pat_seqs, patient_ids, train_mask)
    
    # Normalize HR sequences using simple standardization (z-score)
    hr_mean = np.mean(hr_seqs)
    hr_std = np.std(hr_seqs)
    hr_seqs_scaled = (hr_seqs - hr_mean) / (hr_std if hr_std > 0 else 1.0)
    
    # Create 4-channel input
    X_phys_informed = create_4_channel_input(processed_signals, pat_seqs_scaled, hr_seqs_scaled)
    
    # All data is for this patient, so no masking needed
    X_patient = X_phys_informed
    y_sbp_patient = y_sbp
    y_dbp_patient = y_dbp
    quality_patient = quality_mask
    
    print(f"\n   - Valid samples: {quality_patient.sum()}/{len(quality_patient)}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = model.predict(X_patient, verbose=0)
    y_pred_sbp = predictions[0].flatten()
    y_pred_dbp = predictions[1].flatten()
    
    # Calculate metrics
    print("\n📊 RESULTS FOR PATIENT:", patient_id)
    print("="*70)
    
    print("\n📈 SBP (Systolic Blood Pressure):")
    print("-" * 70)
    sbp_metrics = calculate_metrics(y_sbp_patient, y_pred_sbp)
    for metric, value in sbp_metrics.items():
        print(f"   {metric:15s}: {value:8.2f}")
    
    print("\n📉 DBP (Diastolic Blood Pressure):")
    print("-" * 70)
    dbp_metrics = calculate_metrics(y_dbp_patient, y_pred_dbp)
    for metric, value in dbp_metrics.items():
        print(f"   {metric:15s}: {value:8.2f}")
    
    # Summary statistics
    print("\n📊 Summary Statistics:")
    print("-" * 70)
    print(f"   Actual SBP:    {np.mean(y_sbp_patient):.1f} ± {np.std(y_sbp_patient):.1f} mmHg")
    print(f"   Predicted SBP: {np.mean(y_pred_sbp):.1f} ± {np.std(y_pred_sbp):.1f} mmHg")
    print(f"   Actual DBP:    {np.mean(y_dbp_patient):.1f} ± {np.std(y_dbp_patient):.1f} mmHg")
    print(f"   Predicted DBP: {np.mean(y_pred_dbp):.1f} ± {np.std(y_pred_dbp):.1f} mmHg")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Patient {patient_id} - BP Prediction Analysis', fontsize=16, fontweight='bold')
    
    # SBP: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_sbp_patient, y_pred_sbp, alpha=0.6, s=50)
    min_val = min(y_sbp_patient.min(), y_pred_sbp.min())
    max_val = max(y_sbp_patient.max(), y_pred_sbp.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual SBP (mmHg)', fontsize=12)
    ax.set_ylabel('Predicted SBP (mmHg)', fontsize=12)
    ax.set_title(f'SBP: R² = {sbp_metrics["R²"]:.3f}, MAE = {sbp_metrics["MAE"]:.1f} mmHg', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # DBP: Predicted vs Actual
    ax = axes[0, 1]
    ax.scatter(y_dbp_patient, y_pred_dbp, alpha=0.6, s=50, color='orange')
    min_val = min(y_dbp_patient.min(), y_pred_dbp.min())
    max_val = max(y_dbp_patient.max(), y_pred_dbp.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual DBP (mmHg)', fontsize=12)
    ax.set_ylabel('Predicted DBP (mmHg)', fontsize=12)
    ax.set_title(f'DBP: R² = {dbp_metrics["R²"]:.3f}, MAE = {dbp_metrics["MAE"]:.1f} mmHg', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # SBP Error Distribution
    ax = axes[1, 0]
    sbp_errors = y_pred_sbp - y_sbp_patient
    ax.hist(sbp_errors, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.axvline(np.mean(sbp_errors), color='green', linestyle='--', lw=2, label=f'Mean: {np.mean(sbp_errors):.1f}')
    ax.set_xlabel('Prediction Error (mmHg)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('SBP Error Distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # DBP Error Distribution
    ax = axes[1, 1]
    dbp_errors = y_pred_dbp - y_dbp_patient
    ax.hist(dbp_errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.axvline(np.mean(dbp_errors), color='green', linestyle='--', lw=2, label=f'Mean: {np.mean(dbp_errors):.1f}')
    ax.set_xlabel('Prediction Error (mmHg)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('DBP Error Distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'checkpoints/patient_{patient_id}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {output_path}")
    
    # Show a few sample predictions
    print("\n📋 Sample Predictions (first 10 samples):")
    print("-" * 70)
    print(f"{'Index':<6} {'Actual SBP':<12} {'Pred SBP':<12} {'Error':<8} | {'Actual DBP':<12} {'Pred DBP':<12} {'Error':<8}")
    print("-" * 70)
    for i in range(min(10, len(y_sbp_patient))):
        sbp_err = y_pred_sbp[i] - y_sbp_patient[i]
        dbp_err = y_pred_dbp[i] - y_dbp_patient[i]
        print(f"{i:<6} {y_sbp_patient[i]:<12.1f} {y_pred_sbp[i]:<12.1f} {sbp_err:>7.1f} | "
              f"{y_dbp_patient[i]:<12.1f} {y_pred_dbp[i]:<12.1f} {dbp_err:>7.1f}")
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    
    plt.show()
    
    return {
        'patient_id': patient_id,
        'n_samples': n_samples,
        'sbp_metrics': sbp_metrics,
        'dbp_metrics': dbp_metrics,
        'y_true_sbp': y_sbp_patient,
        'y_pred_sbp': y_pred_sbp,
        'y_true_dbp': y_dbp_patient,
        'y_pred_dbp': y_pred_dbp
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BP predictions for a specific patient')
    parser.add_argument('--patient', type=str, default=None, 
                       help='Patient ID (e.g., p003232). If not provided, uses last patient.')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.h5',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    analyze_patient(patient_id=args.patient, checkpoint_path=args.model)
