"""
Main script for training blood pressure prediction models.
Orchestrates data loading, preprocessing, feature engineering, and model training.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import from our modules
from config import EPOCHS, BATCH_SIZE, VERBOSE, PROCESSED_DATA_DIR
from data_loader import load_aggregate_data
from preprocessing import preprocess_signals, create_subject_wise_splits
from feature_engineering import (
    extract_physiological_features, standardize_feature_length,
    create_baseline_features, create_4_channel_input
)
from model_builder import create_phys_informed_model
from utils import _ensure_finite, _ensure_finite_1d, normalize_data

# Plotting configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


def train_baseline_model(X_baseline, y_baseline, train_mask, val_mask, test_mask):
    """Train and evaluate a baseline linear regression model."""
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    
    if X_baseline.shape[0] <= 10:
        print("⚠️  Not enough data to train and evaluate the baseline model.")
        return None
    
    # Check if we have subject-wise masks
    have_subject_masks = (
        train_mask is not None and val_mask is not None and test_mask is not None
    )
    
    if have_subject_masks:
        X_base_train, y_base_train = X_baseline[train_mask], y_baseline[train_mask]
        X_base_val, y_base_val = X_baseline[val_mask], y_baseline[val_mask]
        X_base_test, y_base_test = X_baseline[test_mask], y_baseline[test_mask]
        split_name = "subject-wise"
    else:
        # Fallback: random split (less strict; can overestimate performance)
        X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(
            X_baseline, y_baseline, test_size=0.3, random_state=42
        )
        X_base_val, y_base_val = None, None
        split_name = "random"

    if X_base_train.shape[0] < 2 or X_base_test.shape[0] < 1:
        print("\n⚠️  Not enough data after splitting to train/evaluate the baseline model.")
        return None

    # Train model
    print(f"\n🤖 Training baseline Linear Regression model ({split_name} split)...")
    baseline_model = LinearRegression()
    baseline_model.fit(X_base_train, y_base_train)
    print("   - Baseline model trained.")

    # Evaluate model (test)
    y_base_pred = baseline_model.predict(X_base_test)
    mae = mean_absolute_error(y_base_test, y_base_pred)
    r2 = r2_score(y_base_test, y_base_pred)

    print(f"\n✅ Baseline Model Evaluation (test):")
    print(f"   - Mean Absolute Error (MAE): {mae:.2f}")
    print(f"   - R^2 Score: {r2:.2f}")

    # Optional: validation metrics when available
    if X_base_val is not None and X_base_val.shape[0] > 0:
        y_val_pred = baseline_model.predict(X_base_val)
        mae_val = mean_absolute_error(y_base_val, y_val_pred)
        r2_val = r2_score(y_base_val, y_val_pred)
        print(f"\n✅ Baseline Model Evaluation (val):")
        print(f"   - Mean Absolute Error (MAE): {mae_val:.2f}")
        print(f"   - R^2 Score: {r2_val:.2f}")

    # Plot results (test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_base_test, y_base_pred, alpha=0.6, edgecolors='k')
    plt.plot([min(y_base_test), max(y_base_test)], 
             [min(y_base_test), max(y_base_test)], 'r--', lw=2, label='Ideal Fit')
    plt.title('Baseline Model: Predicted vs. Actual SBP')
    plt.xlabel('Actual SBP (mmHg)')
    plt.ylabel('Predicted SBP (mmHg)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return baseline_model


def train_physiology_informed_model(X_train_phys, y_train, X_val_phys, y_val, X_test_phys, y_test):
    """Train and evaluate the physiology-informed deep learning model."""
    print("\n" + "="*60)
    print("PHYSIOLOGY-INFORMED MODEL TRAINING")
    print("="*60)
    
    if X_train_phys.shape[0] < 2 or X_val_phys.shape[0] < 1:
        print("\n⚠️ Not enough data to train/validate the physiology-informed model after filtering.")
        return None

    phys_input_shape = (X_train_phys.shape[1], X_train_phys.shape[2])
    phys_informed_model = create_phys_informed_model(phys_input_shape)
    phys_informed_model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    effective_batch_size = int(min(BATCH_SIZE, max(1, X_train_phys.shape[0])))
    if effective_batch_size != BATCH_SIZE:
        print(f"\nℹ️ Adjusting batch_size from {BATCH_SIZE} to {effective_batch_size} (tiny dataset).")

    # Train
    print("\n🏃‍♂️ Training Physiology-Informed Model...")
    phys_informed_history = phys_informed_model.fit(
        X_train_phys,
        y_train,
        validation_data=(X_val_phys, y_val),
        epochs=EPOCHS,
        batch_size=effective_batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=VERBOSE,
    )
    print("   - Training complete.")
    
    return phys_informed_model, phys_informed_history


def main():
    """Main execution pipeline."""
    print("\n" + "="*60)
    print("BLOOD PRESSURE PREDICTION MODEL TRAINING")
    print("="*60 + "\n")
    
    # Step 1: Load data
    signals_agg, labels_agg, demographics_agg, patient_ids_agg = load_aggregate_data(PROCESSED_DATA_DIR)

    if signals_agg is None:
        print("\n❌ No data available. Exiting.")
        return

    # Step 2: Preprocess signals
    processed_signals = preprocess_signals(signals_agg)
    y = labels_agg

    # Step 3: Create subject-wise splits
    train_mask, val_mask, test_mask = create_subject_wise_splits(patient_ids_agg)

    # Step 4: Process demographics
    print("\n🔄 Imputing and scaling demographic data...")
    imputer = SimpleImputer(strategy='mean')
    scaler_demo = StandardScaler()

    demo_train_raw = demographics_agg[train_mask]
    demo_val_raw = demographics_agg[val_mask]
    demo_test_raw = demographics_agg[test_mask]

    demo_train = imputer.fit_transform(demo_train_raw)
    demo_train = scaler_demo.fit_transform(demo_train)

    demo_val = imputer.transform(demo_val_raw)
    demo_val = scaler_demo.transform(demo_val)

    demo_test = imputer.transform(demo_test_raw)
    demo_test = scaler_demo.transform(demo_test)

    print("   - Missing values and scaling handled for demographics.")

    # Step 5: Extract physiological features
    # Raw signals are (samples, timesteps, channels): channel 0 PPG, channel 1 ECG
    # Need to transpose back to (samples, channels, timesteps) for feature extraction
    signals_for_features = np.transpose(processed_signals, (0, 2, 1))
    ppg_raw = signals_for_features[:, 0, :]
    ecg_raw = signals_for_features[:, 1, :]

    pat_seqs, hr_seqs, peak_indices = extract_physiological_features(ecg_raw, ppg_raw)

    # Standardize length to match processed_signals
    target_len = processed_signals.shape[1]
    pat_seqs = standardize_feature_length(pat_seqs, target_len)
    hr_seqs = standardize_feature_length(hr_seqs, target_len)
    print(f"   - Physiological features standardized to length: {target_len}")

    # Step 6: Scale physiological features using training split only
    print("\n🔄 Scaling physiological features (PAT, HR)...")
    scaler_pat = StandardScaler()
    scaler_hr = StandardScaler()
    scaler_pat.fit(pat_seqs[train_mask])
    scaler_hr.fit(hr_seqs[train_mask])
    pat_seqs_scaled = scaler_pat.transform(pat_seqs)
    hr_seqs_scaled = scaler_hr.transform(hr_seqs)
    print("   - Physiological features scaled.")

    # Step 7: Train baseline model
    X_baseline, y_baseline = create_baseline_features(pat_seqs, hr_seqs, y)
    baseline_model = train_baseline_model(X_baseline, y_baseline, train_mask, val_mask, test_mask)

    # Step 8: Prepare data for physiology-informed model
    X_phys_informed = create_4_channel_input(processed_signals, pat_seqs_scaled, hr_seqs_scaled)
    X_phys_informed = _ensure_finite("X_phys_informed (pre-split)", X_phys_informed)

    # Apply masks
    X_train_phys, y_train = X_phys_informed[train_mask], y[train_mask]
    X_val_phys, y_val = X_phys_informed[val_mask], y[val_mask]
    X_test_phys, y_test = X_phys_informed[test_mask], y[test_mask]

    # Ensure labels are finite (drop invalid samples if any)
    y_train, train_good = _ensure_finite_1d("y_train", y_train)
    y_val, val_good = _ensure_finite_1d("y_val", y_val)
    y_test, test_good = _ensure_finite_1d("y_test", y_test)
    X_train_phys = X_train_phys[train_good]
    X_val_phys = X_val_phys[val_good]
    X_test_phys = X_test_phys[test_good]

    # Ensure inputs are finite
    X_train_phys = _ensure_finite("X_train_phys (pre-norm)", X_train_phys)
    X_val_phys = _ensure_finite("X_val_phys (pre-norm)", X_val_phys)
    X_test_phys = _ensure_finite("X_test_phys (pre-norm)", X_test_phys)

    # Normalize
    X_train_phys, X_val_phys, X_test_phys = normalize_data(X_train_phys, X_val_phys, X_test_phys)

    print(f"\n🎯 Final data shapes:")
    print(f"   X_train: {X_train_phys.shape}, y_train: {y_train.shape}")
    print(f"   X_val:   {X_val_phys.shape}, y_val: {y_val.shape}")
    print(f"   X_test:  {X_test_phys.shape}, y_test: {y_test.shape}")

    # Step 9: Train physiology-informed model
    phys_model, history = train_physiology_informed_model(
        X_train_phys, y_train, X_val_phys, y_val, X_test_phys, y_test
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
