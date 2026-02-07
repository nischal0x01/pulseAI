"""
Calibrate model predictions to fix systematic bias.
Uses linear calibration on validation set.
"""
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

def train_calibration(model, X_val, y_val_sbp, y_val_dbp):
    """
    Train calibration layer to correct systematic bias.
    
    Calibration formula:
        SBP_corrected = α × SBP_predicted + β
        DBP_corrected = γ × DBP_predicted + δ
    """
    print("\n🔧 Training calibration layer...")
    
    # Get predictions on validation set
    predictions = model.predict(X_val, verbose=0)
    y_pred_sbp = predictions[0].flatten()
    y_pred_dbp = predictions[1].flatten()
    
    # Train linear calibration for SBP
    sbp_calibrator = LinearRegression()
    sbp_calibrator.fit(y_pred_sbp.reshape(-1, 1), y_val_sbp)
    
    # Train linear calibration for DBP
    dbp_calibrator = LinearRegression()
    dbp_calibrator.fit(y_pred_dbp.reshape(-1, 1), y_val_dbp)
    
    print(f"   SBP calibration: y = {sbp_calibrator.coef_[0]:.3f} × pred + {sbp_calibrator.intercept_:.1f}")
    print(f"   DBP calibration: y = {dbp_calibrator.coef_[0]:.3f} × pred + {dbp_calibrator.intercept_:.1f}")
    
    # Save calibrators
    calibration = {
        'sbp_calibrator': sbp_calibrator,
        'dbp_calibrator': dbp_calibrator
    }
    
    with open('checkpoints/calibration.pkl', 'wb') as f:
        pickle.dump(calibration, f)
    
    print("   ✅ Calibration saved to checkpoints/calibration.pkl")
    
    return calibration


def apply_calibration(y_pred_sbp, y_pred_dbp, calibration=None):
    """Apply calibration to predictions."""
    if calibration is None:
        # Load saved calibration
        with open('checkpoints/calibration.pkl', 'rb') as f:
            calibration = pickle.load(f)
    
    sbp_calibrator = calibration['sbp_calibrator']
    dbp_calibrator = calibration['dbp_calibrator']
    
    # Apply calibration
    y_pred_sbp_cal = sbp_calibrator.predict(y_pred_sbp.reshape(-1, 1))
    y_pred_dbp_cal = dbp_calibrator.predict(y_pred_dbp.reshape(-1, 1))
    
    return y_pred_sbp_cal, y_pred_dbp_cal


if __name__ == '__main__':
    import tensorflow as tf
    
    # Load model
    print("📦 Loading model...")
    model = tf.keras.models.load_model('checkpoints/best_model.h5', safe_mode=False, compile=False)
    
    # Load cached validation data
    print("📊 Loading validation data...")
    cache = np.load('checkpoints/preprocessed_data_cache.npz')
    X_val = cache['X_val']
    y_val_sbp = cache['y_val_sbp']
    y_val_dbp = cache['y_val_dbp']
    
    print(f"   - Validation samples: {len(y_val_sbp)}")
    
    # Train calibration
    calibration = train_calibration(model, X_val, y_val_sbp, y_val_dbp)
    
    # Test on validation set
    print("\n📊 Testing calibration on validation set...")
    predictions = model.predict(X_val[:1000], verbose=0)
    y_pred_sbp = predictions[0].flatten()
    y_pred_dbp = predictions[1].flatten()
    
    # Before calibration
    mae_sbp_before = np.mean(np.abs(y_val_sbp[:1000] - y_pred_sbp))
    mae_dbp_before = np.mean(np.abs(y_val_dbp[:1000] - y_pred_dbp))
    mean_err_sbp_before = np.mean(y_pred_sbp - y_val_sbp[:1000])
    mean_err_dbp_before = np.mean(y_pred_dbp - y_val_dbp[:1000])
    
    # After calibration
    y_pred_sbp_cal, y_pred_dbp_cal = apply_calibration(y_pred_sbp, y_pred_dbp, calibration)
    mae_sbp_after = np.mean(np.abs(y_val_sbp[:1000] - y_pred_sbp_cal))
    mae_dbp_after = np.mean(np.abs(y_val_dbp[:1000] - y_pred_dbp_cal))
    mean_err_sbp_after = np.mean(y_pred_sbp_cal - y_val_sbp[:1000])
    mean_err_dbp_after = np.mean(y_pred_dbp_cal - y_val_dbp[:1000])
    
    print(f"\n   📈 SBP Results:")
    print(f"      MAE:        {mae_sbp_before:.2f} → {mae_sbp_after:.2f} mmHg")
    print(f"      Mean Error: {mean_err_sbp_before:.2f} → {mean_err_sbp_after:.2f} mmHg")
    
    print(f"\n   📉 DBP Results:")
    print(f"      MAE:        {mae_dbp_before:.2f} → {mae_dbp_after:.2f} mmHg")
    print(f"      Mean Error: {mean_err_dbp_before:.2f} → {mean_err_dbp_after:.2f} mmHg")
    
    if mae_sbp_after < mae_sbp_before and mae_dbp_after < mae_dbp_before:
        print("\n   ✅ Calibration successfully improves predictions!")
    else:
        print("\n   ⚠️  Calibration may not help - check model training")
    
    print("\n" + "="*70)
    print("  Calibration training complete!")
    print("  Use analyze_patient_predictions.py to test calibrated predictions")
    print("="*70)