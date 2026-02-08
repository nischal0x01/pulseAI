"""
Patient-specific calibration with multiple strategies.
"""

import numpy as np
import sys
import os
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

import tensorflow as tf
from preprocessing import preprocess_signals
from feature_engineering import (extract_physiological_features, 
                                 standardize_feature_length, 
                                 normalize_pat_subject_wise,
                                 create_4_channel_input)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def calibrate_patient(patient_id, checkpoint_path='checkpoints/best_model.h5', 
                     n_calibration_samples=50, method='linear', 
                     variance_correction=True):
    """
    Calibrate model for a specific patient using their initial readings.
    
    Args:
        patient_id: Patient ID (e.g., 'p003357')
        checkpoint_path: Path to trained model
        n_calibration_samples: Number of initial samples to use for calibration
        method: 'linear' (slope+intercept) or 'offset' (intercept only)
        variance_correction: Whether to correct prediction variance
    
    Returns:
        Calibration parameters dictionary
    """
    
    print("="*70)
    print("  PATIENT-SPECIFIC CALIBRATION")
    print("="*70)
    print(f"\n👤 Patient: {patient_id}")
    print(f"📊 Calibration samples: {n_calibration_samples}")
    print(f"🔧 Method: {method}")
    print(f"📏 Variance correction: {'ON' if variance_correction else 'OFF'}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = tf.keras.models.load_model(checkpoint_path, safe_mode=False)
    
    # Load patient data
    from data_loader import _read_subj_wins
    patient_file = f"data/processed/{patient_id}.mat"
    
    if not os.path.exists(patient_file):
        print(f"❌ Patient file not found: {patient_file}")
        return None
    
    signals_agg, y_sbp, y_dbp, demographics = _read_subj_wins(patient_file)
    n_total = len(y_sbp)
    
    if n_total < n_calibration_samples:
        print(f"⚠️  Only {n_total} samples available, using all for calibration")
        n_calibration_samples = n_total
    
    print(f"   Total samples: {n_total}")
    print(f"   Using first {n_calibration_samples} for calibration")
    
    # Preprocess all data
    print("\n🔄 Preprocessing...")
    processed_signals = preprocess_signals(signals_agg)
    
    # Extract features
    ppg_raw = signals_agg[:, 0, :]
    ecg_raw = signals_agg[:, 1, :]
    pat_seqs, hr_seqs, quality_mask = extract_physiological_features(ecg_raw, ppg_raw)
    
    target_len = processed_signals.shape[1]
    pat_seqs = standardize_feature_length(pat_seqs, target_len)
    hr_seqs = standardize_feature_length(hr_seqs, target_len)
    
    # Normalize features
    patient_ids = np.array([patient_id] * n_total)
    train_mask = np.ones(n_total, dtype=bool)
    pat_seqs_scaled, _ = normalize_pat_subject_wise(pat_seqs, patient_ids, train_mask)
    
    hr_mean = np.mean(hr_seqs)
    hr_std = np.std(hr_seqs)
    hr_seqs_scaled = (hr_seqs - hr_mean) / (hr_std if hr_std > 0 else 1.0)
    
    X_phys = create_4_channel_input(processed_signals, pat_seqs_scaled, hr_seqs_scaled)
    
    # Get predictions for calibration set
    print("\n🔮 Computing predictions for calibration set...")
    X_cal = X_phys[:n_calibration_samples]
    y_sbp_cal = y_sbp[:n_calibration_samples]
    y_dbp_cal = y_dbp[:n_calibration_samples]
    
    predictions = model.predict(X_cal, verbose=0)
    y_pred_sbp_cal = predictions[0].flatten()
    y_pred_dbp_cal = predictions[1].flatten()
    
    # Compute calibration parameters
    print("\n🔧 Computing calibration parameters...")
    
    if method == 'linear':
        # Full linear calibration: y_true = slope * y_pred + intercept
        A_sbp = np.vstack([y_pred_sbp_cal, np.ones(len(y_pred_sbp_cal))]).T
        sbp_slope, sbp_intercept = np.linalg.lstsq(A_sbp, y_sbp_cal, rcond=None)[0]
        
        A_dbp = np.vstack([y_pred_dbp_cal, np.ones(len(y_pred_dbp_cal))]).T
        dbp_slope, dbp_intercept = np.linalg.lstsq(A_dbp, y_dbp_cal, rcond=None)[0]
        
    elif method == 'offset':
        # Offset only: y_true = y_pred + offset
        sbp_slope = 1.0
        sbp_intercept = np.mean(y_sbp_cal - y_pred_sbp_cal)
        
        dbp_slope = 1.0
        dbp_intercept = np.mean(y_dbp_cal - y_pred_dbp_cal)
    
    # Variance correction (optional)
    if variance_correction:
        # Adjust slope to match variance if needed
        std_actual_sbp = np.std(y_sbp_cal)
        std_pred_sbp = np.std(y_pred_sbp_cal)
        std_actual_dbp = np.std(y_dbp_cal)
        std_pred_dbp = np.std(y_pred_dbp_cal)
        
        if std_pred_sbp > 0:
            variance_ratio_sbp = std_actual_sbp / std_pred_sbp
            # Blend with existing slope
            sbp_slope = 0.5 * sbp_slope + 0.5 * variance_ratio_sbp
            
        if std_pred_dbp > 0:
            variance_ratio_dbp = std_actual_dbp / std_pred_dbp
            dbp_slope = 0.5 * dbp_slope + 0.5 * variance_ratio_dbp
        
        # Recompute intercept to preserve mean
        sbp_intercept = np.mean(y_sbp_cal) - sbp_slope * np.mean(y_pred_sbp_cal)
        dbp_intercept = np.mean(y_dbp_cal) - dbp_slope * np.mean(y_pred_dbp_cal)
    
    calibration_params = {
        'patient_id': patient_id,
        'n_calibration_samples': n_calibration_samples,
        'method': method,
        'variance_correction': variance_correction,
        'sbp_slope': float(sbp_slope),
        'sbp_intercept': float(sbp_intercept),
        'dbp_slope': float(dbp_slope),
        'dbp_intercept': float(dbp_intercept),
        'sbp_baseline': float(np.mean(y_sbp_cal)),
        'dbp_baseline': float(np.mean(y_dbp_cal)),
        'sbp_std': float(np.std(y_sbp_cal)),
        'dbp_std': float(np.std(y_dbp_cal))
    }
    
    print(f"\n   📈 SBP calibration: y = {sbp_slope:.3f} × pred + {sbp_intercept:.1f}")
    print(f"      Baseline: {calibration_params['sbp_baseline']:.1f} ± {calibration_params['sbp_std']:.1f} mmHg")
    print(f"\n   📉 DBP calibration: y = {dbp_slope:.3f} × pred + {dbp_intercept:.1f}")
    print(f"      Baseline: {calibration_params['dbp_baseline']:.1f} ± {calibration_params['dbp_std']:.1f} mmHg")
    
    # Test on remaining samples (if any)
    if n_total > n_calibration_samples:
        print(f"\n📊 Testing on remaining {n_total - n_calibration_samples} samples...")
        
        X_test = X_phys[n_calibration_samples:]
        y_sbp_test = y_sbp[n_calibration_samples:]
        y_dbp_test = y_dbp[n_calibration_samples:]
        
        predictions_test = model.predict(X_test, verbose=0)
        y_pred_sbp_test = predictions_test[0].flatten()
        y_pred_dbp_test = predictions_test[1].flatten()
        
        # Apply calibration
        y_pred_sbp_cal_test = sbp_slope * y_pred_sbp_test + sbp_intercept
        y_pred_dbp_cal_test = dbp_slope * y_pred_dbp_test + dbp_intercept
        
        # Metrics before/after
        mae_sbp_before = np.mean(np.abs(y_sbp_test - y_pred_sbp_test))
        mae_sbp_after = np.mean(np.abs(y_sbp_test - y_pred_sbp_cal_test))
        
        mae_dbp_before = np.mean(np.abs(y_dbp_test - y_pred_dbp_test))
        mae_dbp_after = np.mean(np.abs(y_dbp_test - y_pred_dbp_cal_test))
        
        # Variance metrics
        std_pred_sbp_before = np.std(y_pred_sbp_test)
        std_pred_sbp_after = np.std(y_pred_sbp_cal_test)
        std_actual_sbp = np.std(y_sbp_test)
        
        std_pred_dbp_before = np.std(y_pred_dbp_test)
        std_pred_dbp_after = np.std(y_pred_dbp_cal_test)
        std_actual_dbp = np.std(y_dbp_test)
        
        print(f"\n   📈 SBP:")
        print(f"      MAE:      {mae_sbp_before:.2f} → {mae_sbp_after:.2f} mmHg")
        print(f"      Std Dev:  {std_pred_sbp_before:.2f} → {std_pred_sbp_after:.2f} mmHg (actual: {std_actual_sbp:.2f})")
        
        print(f"\n   📉 DBP:")
        print(f"      MAE:      {mae_dbp_before:.2f} → {mae_dbp_after:.2f} mmHg")
        print(f"      Std Dev:  {std_pred_dbp_before:.2f} → {std_pred_dbp_after:.2f} mmHg (actual: {std_actual_dbp:.2f})")
        
        improvement_sbp = mae_sbp_before - mae_sbp_after
        improvement_dbp = mae_dbp_before - mae_dbp_after
        
        if improvement_sbp > 0 and improvement_dbp > 0:
            print(f"\n   ✅ Calibration improved both SBP and DBP!")
        elif improvement_sbp > 0 or improvement_dbp > 0:
            print(f"\n   ⚠️  Calibration improved one metric but not both")
        else:
            print(f"\n   ⚠️  Calibration did not improve (may need different settings)")
    
    # Save calibration
    os.makedirs('calibration', exist_ok=True)
    cal_path = f'calibration/calibration_{patient_id}.pkl'
    with open(cal_path, 'wb') as f:
        pickle.dump(calibration_params, f)
    
    print(f"\n💾 Calibration saved to: {cal_path}")
    
    return calibration_params


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate model for specific patient')
    parser.add_argument('--patient', type=str, required=True,
                       help='Patient ID (e.g., p003357)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of initial samples for calibration (default: 50)')
    parser.add_argument('--method', type=str, default='linear',
                       choices=['linear', 'offset'],
                       help='Calibration method: linear (slope+intercept) or offset (intercept only)')
    parser.add_argument('--variance-correction', action='store_true', default=True,
                       help='Apply variance correction to match prediction spread (default: True)')
    parser.add_argument('--no-variance-correction', action='store_false', dest='variance_correction',
                       help='Disable variance correction')
    
    args = parser.parse_args()
    
    calibrate_patient(args.patient, args.model, args.n_samples, 
                     args.method, args.variance_correction)
    
    print("\n" + "="*70)
    print("  Now run: python analyze_patient_predictions.py --patient", args.patient)
    print("="*70)
