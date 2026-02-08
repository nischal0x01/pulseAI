"""
Patient-specific calibration using stratified sampling across BP range.
This ensures calibration samples represent the full BP distribution.
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


def stratified_sample(y_values, n_samples, n_bins=5):
    """
    Sample indices stratified across BP value bins.
    
    Args:
        y_values: Array of BP values
        n_samples: Number of samples to select
        n_bins: Number of bins for stratification
        
    Returns:
        Array of selected indices
    """
    # Create bins
    bins = np.linspace(y_values.min(), y_values.max(), n_bins + 1)
    bin_indices = np.digitize(y_values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Sample from each bin
    samples_per_bin = n_samples // n_bins
    extra_samples = n_samples % n_bins
    
    selected_indices = []
    for bin_idx in range(n_bins):
        bin_mask = bin_indices == bin_idx
        bin_sample_indices = np.where(bin_mask)[0]
        
        if len(bin_sample_indices) == 0:
            continue
            
        # Number of samples from this bin
        n_bin_samples = samples_per_bin + (1 if bin_idx < extra_samples else 0)
        n_bin_samples = min(n_bin_samples, len(bin_sample_indices))
        
        # Random sample from this bin
        selected = np.random.choice(bin_sample_indices, n_bin_samples, replace=False)
        selected_indices.extend(selected)
    
    return np.array(sorted(selected_indices))


def calibrate_patient_stratified(patient_id, checkpoint_path='checkpoints/best_model.h5', 
                                  n_calibration_samples=50, method='offset', 
                                  stratify='sbp', seed=42, incremental=True):
    """
    Calibrate model using stratified sampling across BP range.
    
    Args:
        patient_id: Patient ID (e.g., 'p003357')
        checkpoint_path: Path to trained model
        n_calibration_samples: Number of samples for calibration
        method: 'linear' or 'offset'
        stratify: 'sbp', 'dbp', or 'both' - which BP to stratify on
        seed: Random seed for reproducibility
        incremental: If True, refine existing calibration if available
    
    Returns:
        Calibration parameters dictionary
    """
    
    np.random.seed(seed)
    
    # Check for existing calibration
    cal_path = f'calibration/calibration_{patient_id}.pkl'
    existing_calibration = None
    if incremental and os.path.exists(cal_path):
        try:
            with open(cal_path, 'rb') as f:
                existing_calibration = pickle.load(f)
            print("="*70)
            print("  INCREMENTAL CALIBRATION (Refining Existing)")
            print("="*70)
            print(f"\n📋 Found existing calibration:")
            print(f"   Previous samples: {existing_calibration['n_calibration_samples']}")
            print(f"   Previous SBP offset: {existing_calibration['sbp_intercept']:.1f} mmHg")
            print(f"   Previous DBP offset: {existing_calibration['dbp_intercept']:.1f} mmHg")
        except Exception as e:
            print(f"⚠️  Could not load existing calibration: {e}")
            existing_calibration = None
    
    if existing_calibration is None:
        print("="*70)
        print("  STRATIFIED PATIENT-SPECIFIC CALIBRATION")
        print("="*70)
    
    print(f"\n👤 Patient: {patient_id}")
    print(f"📊 New calibration samples: {n_calibration_samples}")
    print(f"🔧 Method: {method}")
    print(f"📈 Stratified by: {stratify.upper()}")
    
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
    
    # Stratified sampling for calibration indices
    print(f"\n🎯 Performing stratified sampling...")
    if stratify == 'sbp':
        cal_indices = stratified_sample(y_sbp, n_calibration_samples)
    elif stratify == 'dbp':
        cal_indices = stratified_sample(y_dbp, n_calibration_samples)
    elif stratify == 'both':
        # Stratify on both - take half based on SBP, half on DBP
        n_half = n_calibration_samples // 2
        sbp_indices = stratified_sample(y_sbp, n_half)
        dbp_indices = stratified_sample(y_dbp, n_calibration_samples - n_half)
        cal_indices = np.unique(np.concatenate([sbp_indices, dbp_indices]))
        # If we got duplicates, fill up to n_calibration_samples
        if len(cal_indices) < n_calibration_samples:
            remaining = set(range(n_total)) - set(cal_indices)
            extra_indices = np.random.choice(list(remaining), 
                                           n_calibration_samples - len(cal_indices), 
                                           replace=False)
            cal_indices = np.sort(np.concatenate([cal_indices, extra_indices]))
    
    # Create test mask
    test_mask = np.ones(n_total, dtype=bool)
    test_mask[cal_indices] = False
    test_indices = np.where(test_mask)[0]
    
    print(f"   Calibration indices: {len(cal_indices)} samples")
    print(f"   Test indices: {len(test_indices)} samples")
    
    # Get calibration data
    X_cal = X_phys[cal_indices]
    y_sbp_cal = y_sbp[cal_indices]
    y_dbp_cal = y_dbp[cal_indices]
    
    print(f"\n   📈 Calibration SBP range: {y_sbp_cal.min():.1f} - {y_sbp_cal.max():.1f} mmHg")
    print(f"      Mean: {y_sbp_cal.mean():.1f} ± {y_sbp_cal.std():.1f}")
    print(f"   📉 Calibration DBP range: {y_dbp_cal.min():.1f} - {y_dbp_cal.max():.1f} mmHg")
    print(f"      Mean: {y_dbp_cal.mean():.1f} ± {y_dbp_cal.std():.1f}")
    
    # Get predictions for calibration set
    print("\n🔮 Computing predictions for calibration set...")
    predictions = model.predict(X_cal, verbose=0)
    y_pred_sbp_cal = predictions[0].flatten()
    y_pred_dbp_cal = predictions[1].flatten()
    
    # Compute calibration parameters
    print("\n🔧 Computing calibration parameters...")
    
    if method == 'linear':
        # Full linear calibration
        A_sbp = np.vstack([y_pred_sbp_cal, np.ones(len(y_pred_sbp_cal))]).T
        sbp_slope, sbp_intercept = np.linalg.lstsq(A_sbp, y_sbp_cal, rcond=None)[0]
        
        A_dbp = np.vstack([y_pred_dbp_cal, np.ones(len(y_pred_dbp_cal))]).T
        dbp_slope, dbp_intercept = np.linalg.lstsq(A_dbp, y_dbp_cal, rcond=None)[0]
        
    elif method == 'offset':
        # Offset only
        sbp_slope = 1.0
        sbp_intercept = np.mean(y_sbp_cal - y_pred_sbp_cal)
        
        dbp_slope = 1.0
        dbp_intercept = np.mean(y_dbp_cal - y_pred_dbp_cal)
    
    # If incremental calibration, combine with existing parameters
    if existing_calibration is not None:
        old_n = existing_calibration['n_calibration_samples']
        new_n = len(cal_indices)
        total_n = old_n + new_n
        
        # Weighted average: more weight to more samples
        weight_old = old_n / total_n
        weight_new = new_n / total_n
        
        print(f"\n🔄 Combining with existing calibration:")
        print(f"   Old weight: {weight_old:.2%} ({old_n} samples)")
        print(f"   New weight: {weight_new:.2%} ({new_n} samples)")
        
        # Combine slopes and intercepts
        old_sbp_slope = existing_calibration['sbp_slope']
        old_sbp_intercept = existing_calibration['sbp_intercept']
        old_dbp_slope = existing_calibration['dbp_slope']
        old_dbp_intercept = existing_calibration['dbp_intercept']
        
        sbp_slope = weight_old * old_sbp_slope + weight_new * sbp_slope
        sbp_intercept = weight_old * old_sbp_intercept + weight_new * sbp_intercept
        dbp_slope = weight_old * old_dbp_slope + weight_new * dbp_slope
        dbp_intercept = weight_old * old_dbp_intercept + weight_new * dbp_intercept
        
        print(f"\n   📈 SBP: {old_sbp_intercept:.1f} → {sbp_intercept:.1f} mmHg offset")
        print(f"   📉 DBP: {old_dbp_intercept:.1f} → {dbp_intercept:.1f} mmHg offset")
        
        # Update total samples
        n_calibration_samples = total_n
    else:
        n_calibration_samples = len(cal_indices)
    
    calibration_params = {
        'patient_id': patient_id,
        'n_calibration_samples': n_calibration_samples,
        'method': method,
        'stratify': stratify,
        'sbp_slope': float(sbp_slope),
        'sbp_intercept': float(sbp_intercept),
        'dbp_slope': float(dbp_slope),
        'dbp_intercept': float(dbp_intercept),
        'sbp_baseline': float(np.mean(y_sbp_cal)),
        'dbp_baseline': float(np.mean(y_dbp_cal)),
        'sbp_std': float(np.std(y_sbp_cal)),
        'dbp_std': float(np.std(y_dbp_cal))
    }
    
    print(f"\n   📈 Final SBP calibration: y = {sbp_slope:.3f} × pred + {sbp_intercept:.1f}")
    print(f"   📉 Final DBP calibration: y = {dbp_slope:.3f} × pred + {dbp_intercept:.1f}")
    
    # Test on remaining samples
    if len(test_indices) > 0:
        print(f"\n📊 Testing on remaining {len(test_indices)} samples...")
        
        X_test = X_phys[test_indices]
        y_sbp_test = y_sbp[test_indices]
        y_dbp_test = y_dbp[test_indices]
        
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
        std_pred_sbp_after = np.std(y_pred_sbp_cal_test)
        std_actual_sbp = np.std(y_sbp_test)
        
        std_pred_dbp_after = np.std(y_pred_dbp_cal_test)
        std_actual_dbp = np.std(y_dbp_test)
        
        print(f"\n   📈 SBP:")
        print(f"      MAE:      {mae_sbp_before:.2f} → {mae_sbp_after:.2f} mmHg")
        print(f"      Pred Std: {std_pred_sbp_after:.2f} (actual: {std_actual_sbp:.2f})")
        
        print(f"\n   📉 DBP:")
        print(f"      MAE:      {mae_dbp_before:.2f} → {mae_dbp_after:.2f} mmHg")
        print(f"      Pred Std: {std_pred_dbp_after:.2f} (actual: {std_actual_dbp:.2f})")
        
        improvement_sbp = mae_sbp_before - mae_sbp_after
        improvement_dbp = mae_dbp_before - mae_dbp_after
        
        if improvement_sbp > 0 and improvement_dbp > 0:
            print(f"\n   ✅ Calibration improved both SBP and DBP!")
        elif improvement_sbp > 0 or improvement_dbp > 0:
            total_improvement = improvement_sbp + improvement_dbp
            print(f"\n   {'✅' if total_improvement > 0 else '⚠️ '} Overall improvement: {total_improvement:.2f} mmHg")
        else:
            print(f"\n   ⚠️  Calibration did not improve")
    
    # Save calibration
    os.makedirs('calibration', exist_ok=True)
    cal_path = f'calibration/calibration_{patient_id}.pkl'
    with open(cal_path, 'wb') as f:
        pickle.dump(calibration_params, f)
    
    if existing_calibration:
        print(f"\n💾 Updated calibration saved to: {cal_path}")
        print(f"   Total calibration samples: {n_calibration_samples}")
    else:
        print(f"\n💾 Calibration saved to: {cal_path}")
    
    return calibration_params


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate model with stratified sampling')
    parser.add_argument('--patient', type=str, required=True,
                       help='Patient ID (e.g., p003357)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.h5',
                       help='Path to trained model')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples for calibration (default: 50)')
    parser.add_argument('--method', type=str, default='offset',
                       choices=['linear', 'offset'],
                       help='Calibration method (default: offset)')
    parser.add_argument('--stratify', type=str, default='both',
                       choices=['sbp', 'dbp', 'both'],
                       help='Stratify by SBP, DBP, or both (default: both)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-incremental', action='store_true',
                       help='Ignore existing calibration and start fresh')
    
    args = parser.parse_args()
    
    calibrate_patient_stratified(args.patient, args.model, args.n_samples, 
                                 args.method, args.stratify, args.seed,
                                 incremental=not args.no_incremental)
    
    print("\n" + "="*70)
    print("  Now run: python analyze_patient_predictions.py --patient", args.patient)
    print("="*70)
