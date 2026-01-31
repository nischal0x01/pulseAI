"""
Main training script for physiology-informed CNN-LSTM blood pressure prediction model.

This script implements:
- Rigorous data validation (no NaN/Inf values)
- 4-channel physiological input: [ECG, PPG, PAT, HR]
- CNN-LSTM architecture with PAT-based attention mechanism
- Huber loss with Adam optimizer and gradient clipping
- Patient-wise data splitting to prevent leakage
- Comprehensive evaluation with MAE, RMSE, R², Pearson correlation
- Attention visualization to confirm PAT influence
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K

# Import from our modules
from config import EPOCHS, BATCH_SIZE, VERBOSE, PROCESSED_DATA_DIR, CHECKPOINT_DIR, RAW_DATA_DIR, BASE_DATA_DIR
from data_loader import load_aggregate_data

# Training configuration
ENABLE_TWO_PHASE_TRAINING = False  # Set to True to freeze CNN layers and train LSTM+attention only
RESUME_LR_REDUCTION_FACTOR = 0.5  # Reduce learning rate by this factor when resuming
from preprocessing import preprocess_signals, create_subject_wise_splits
from feature_engineering import (
    extract_physiological_features, standardize_feature_length,
    create_4_channel_input
)
from model_builder_attention import (
    create_phys_informed_model,
    create_attention_visualization_model
)
from utils import _ensure_finite, _ensure_finite_1d, normalize_data
from evaluation import comprehensive_evaluation, calculate_comprehensive_metrics, print_metrics

# Plotting configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)


def validate_data_integrity(data, name):
    """
    Assert no NaN or Inf values in data.
    
    Args:
        data: numpy array to validate
        name: name of the data for error messages
        
    Raises:
        AssertionError if NaN or Inf values found
    """
    assert not np.any(np.isnan(data)), f"❌ {name} contains NaN values!"
    assert not np.any(np.isinf(data)), f"❌ {name} contains Inf values!"
    print(f"✅ {name} validated: No NaN or Inf values")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("  PHYSIOLOGY-INFORMED CNN-LSTM BLOOD PRESSURE PREDICTION MODEL")
    print("  With PAT-Based Attention Mechanism")
    print("="*70 + "\n")
    
    # ===== Step 1: Load Data =====
    print("📂 STEP 1: Loading data...")
    signals_agg, sbp_labels_agg, dbp_labels_agg, demographics_agg, patient_ids_agg = load_aggregate_data(
        PROCESSED_DATA_DIR
    )
    
    if signals_agg is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # ===== Step 2: Preprocess Signals =====
    print("\n🔧 STEP 2: Preprocessing signals...")
    print("   - Applying bandpass filters (PPG: 0.5-8 Hz, ECG: 0.5-40 Hz)")
    print("   - Normalizing with Z-score")
    processed_signals = preprocess_signals(signals_agg)
    y_sbp = sbp_labels_agg
    y_dbp = dbp_labels_agg
    
    # Validate preprocessed signals
    validate_data_integrity(processed_signals, "Preprocessed signals")
    validate_data_integrity(y_sbp, "SBP Labels")
    validate_data_integrity(y_dbp, "DBP Labels")
    
    # ===== Step 3: Create Patient-Wise Splits =====
    print("\n📊 STEP 3: Creating patient-wise data splits...")
    train_mask, val_mask, test_mask = create_subject_wise_splits(patient_ids_agg)
    
    # ===== Step 4: Extract Physiological Features =====
    print("\n💓 STEP 4: Extracting physiological features (PAT, HR)...")
    
    # Raw signals are (samples, timesteps, channels): channel 0 PPG, channel 1 ECG
    ppg_raw = signals_agg[:, 0, :]
    ecg_raw = signals_agg[:, 1, :]
    
    pat_seqs, hr_seqs, peak_indices = extract_physiological_features(ecg_raw, ppg_raw)
    
    # Standardize length to match processed_signals
    target_len = processed_signals.shape[1]
    pat_seqs = standardize_feature_length(pat_seqs, target_len)
    hr_seqs = standardize_feature_length(hr_seqs, target_len)
    print(f"   - Physiological features standardized to length: {target_len}")
    
    # Validate features before scaling
    validate_data_integrity(pat_seqs, "PAT sequences (before scaling)")
    validate_data_integrity(hr_seqs, "HR sequences (before scaling)")
    
    # ===== Step 5: Scale Physiological Features =====
    print("\n🔄 STEP 5: Scaling physiological features (using training set statistics)...")
    scaler_pat = StandardScaler()
    scaler_hr = StandardScaler()
    
    scaler_pat.fit(pat_seqs[train_mask])
    scaler_hr.fit(hr_seqs[train_mask])
    
    pat_seqs_scaled = scaler_pat.transform(pat_seqs)
    hr_seqs_scaled = scaler_hr.transform(hr_seqs)
    
    # Validate after scaling
    validate_data_integrity(pat_seqs_scaled, "Scaled PAT sequences")
    validate_data_integrity(hr_seqs_scaled, "Scaled HR sequences")
    print("   - Physiological features scaled successfully")
    
    # ===== Step 6: Create 4-Channel Input =====
    print("\n🔀 STEP 6: Creating 4-channel input [ECG, PPG, PAT, HR]...")
    X_phys_informed = create_4_channel_input(processed_signals, pat_seqs_scaled, hr_seqs_scaled)
    
    # Validate 4-channel input
    validate_data_integrity(X_phys_informed, "4-channel input")
    
    # Apply masks for train/val/test split
    X_train_phys = X_phys_informed[train_mask]
    X_val_phys = X_phys_informed[val_mask]
    X_test_phys = X_phys_informed[test_mask]
    
    y_train_sbp = y_sbp[train_mask]
    y_val_sbp = y_sbp[val_mask]
    y_test_sbp = y_sbp[test_mask]
    
    y_train_dbp = y_dbp[train_mask]
    y_val_dbp = y_dbp[val_mask]
    y_test_dbp = y_dbp[test_mask]
    
    # Final validation before training
    print("\n✅ STEP 7: Final data validation...")
    validate_data_integrity(X_train_phys, "Training features")
    validate_data_integrity(X_val_phys, "Validation features")
    validate_data_integrity(X_test_phys, "Test features")
    validate_data_integrity(y_train_sbp, "Training SBP labels")
    validate_data_integrity(y_val_sbp, "Validation SBP labels")
    validate_data_integrity(y_test_sbp, "Test SBP labels")
    validate_data_integrity(y_train_dbp, "Training DBP labels")
    validate_data_integrity(y_val_dbp, "Validation DBP labels")
    validate_data_integrity(y_test_dbp, "Test DBP labels")
    
    print(f"\n📐 Data shapes:")
    print(f"   X_train: {X_train_phys.shape}")
    print(f"   y_train_sbp: {y_train_sbp.shape}, y_train_dbp: {y_train_dbp.shape}")
    print(f"   X_val: {X_val_phys.shape}")
    print(f"   y_val_sbp: {y_val_sbp.shape}, y_val_dbp: {y_val_dbp.shape}")
    print(f"   X_test: {X_test_phys.shape}")
    print(f"   y_test_sbp: {y_test_sbp.shape}, y_test_dbp: {y_test_dbp.shape}")
    
    # Verify 4 channels
    assert X_train_phys.shape[-1] == 4, "Expected 4 channels: [ECG, PPG, PAT, HR]"
    print(f"   ✅ Verified 4 channels: [ECG, PPG, PAT, HR]")
    
    # ===== Step 8: Build Model =====
    print("\n🏗️  STEP 8: Building physiology-informed CNN-LSTM model with attention...")
    phys_input_shape = (X_train_phys.shape[1], X_train_phys.shape[2])
    phys_informed_model = create_phys_informed_model(phys_input_shape)
    
    print("\n📋 Model Architecture:")
    phys_informed_model.summary()
    
    # Check for existing checkpoint and resume training if available
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.h5')
    is_resuming = False
    initial_lr = K.get_value(phys_informed_model.optimizer.learning_rate)
    
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 CHECKPOINT FOUND: Resuming training from {checkpoint_path}")
        try:
            phys_informed_model.load_weights(checkpoint_path)
            is_resuming = True
            
            # Reduce learning rate for fine-tuning
            new_lr = initial_lr * RESUME_LR_REDUCTION_FACTOR
            K.set_value(phys_informed_model.optimizer.learning_rate, new_lr)
            print(f"   ✅ Weights loaded successfully")
            print(f"   📉 Learning rate adjusted: {initial_lr:.2e} → {new_lr:.2e} (reduced by {RESUME_LR_REDUCTION_FACTOR}x)")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print(f"   🆕 Training from scratch instead")
            is_resuming = False
    else:
        print(f"\n🆕 NO CHECKPOINT FOUND: Training from scratch")
        print(f"   💾 Model will be saved to: {checkpoint_path}")
    
    print(f"   🎯 Current learning rate: {K.get_value(phys_informed_model.optimizer.learning_rate):.2e}")
    
    # ===== Step 9: Setup Callbacks =====
    print("\n⚙️  STEP 9: Setting up training callbacks...")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    callbacks_list = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    # ===== Step 10: Train Model =====
    # Optional: Two-phase training (freeze CNN layers, train LSTM+attention only)
    if ENABLE_TWO_PHASE_TRAINING and is_resuming:
        print("\n🔒 TWO-PHASE TRAINING: Freezing CNN layers, training LSTM+attention only...")
        
        # Freeze CNN layers
        for layer in phys_informed_model.layers:
            if 'conv' in layer.name.lower():
                layer.trainable = False
                print(f"   🔒 Frozen: {layer.name}")
        
        # Recompile model with frozen layers
        phys_informed_model.compile(
            optimizer=phys_informed_model.optimizer,
            loss={'sbp_output': 'huber', 'dbp_output': 'huber'},
            metrics={'sbp_output': 'mae', 'dbp_output': 'mae'}
        )
        
        print("   ✅ CNN layers frozen, model recompiled")
        print(f"   📊 Trainable params: {phys_informed_model.count_params():,}")
    
    print("\n🏃‍♂️ STEP 10: Training physiology-informed CNN-LSTM model...")
    print(f"   - Mode: {'RESUMED (fine-tuning)' if is_resuming else 'FRESH START'}")
    if ENABLE_TWO_PHASE_TRAINING and is_resuming:
        print(f"   - Two-phase: CNN frozen, LSTM+attention trainable")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Loss: Huber (robust to outliers)")
    print(f"   - Optimizer: Adam with gradient clipping")
    print(f"   - Outputs: SBP and DBP (dual prediction)")
    print(f"   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n")
    
    training_start_time = datetime.now()
    
    history = phys_informed_model.fit(
        X_train_phys,
        {'sbp_output': y_train_sbp, 'dbp_output': y_train_dbp},
        validation_data=(X_val_phys, {'sbp_output': y_val_sbp, 'dbp_output': y_val_dbp}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=VERBOSE
    )
    
    training_end_time = datetime.now()
    training_duration = (training_end_time - training_start_time).total_seconds()
    
    print("\n✅ Training completed!")
    print(f"   ⏱️  Duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    
    # ===== Step 11: Plot Training History =====
    print("\n📈 STEP 11: Plotting training history...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss (Huber)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Total Model Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # SBP MAE plot
    axes[0, 1].plot(history.history['sbp_output_mae'], label='Training SBP MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_sbp_output_mae'], label='Validation SBP MAE', linewidth=2)
    axes[0, 1].axhline(y=10, color='r', linestyle='--', label='Target MAE (10 mmHg)', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('MAE (mmHg)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('SBP Mean Absolute Error Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # DBP MAE plot
    axes[1, 0].plot(history.history['dbp_output_mae'], label='Training DBP MAE', linewidth=2)
    axes[1, 0].plot(history.history['val_dbp_output_mae'], label='Validation DBP MAE', linewidth=2)
    axes[1, 0].axhline(y=10, color='r', linestyle='--', label='Target MAE (10 mmHg)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MAE (mmHg)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('DBP Mean Absolute Error Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined MAE comparison
    axes[1, 1].plot(history.history['sbp_output_mae'], label='Training SBP MAE', linewidth=2, linestyle='--')
    axes[1, 1].plot(history.history['dbp_output_mae'], label='Training DBP MAE', linewidth=2, linestyle='--')
    axes[1, 1].plot(history.history['val_sbp_output_mae'], label='Val SBP MAE', linewidth=2)
    axes[1, 1].plot(history.history['val_dbp_output_mae'], label='Val DBP MAE', linewidth=2)
    axes[1, 1].axhline(y=10, color='r', linestyle='--', label='Target (10 mmHg)', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('MAE (mmHg)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('SBP vs DBP Performance', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== Step 12: Evaluate on Validation Set =====
    print("\n📊 STEP 12: Evaluating on validation set...")
    val_results = comprehensive_evaluation(
        phys_informed_model,
        X_val_phys,
        {'sbp': y_val_sbp, 'dbp': y_val_dbp},
        dataset_name="Validation",
        visualize_attention=False,
        save_dir=CHECKPOINT_DIR
    )
    
    # ===== Step 13: Evaluate on Test Set =====
    print("\n📊 STEP 13: Evaluating on test set...")
    
    # Create attention visualization model
    print("   - Creating attention visualization model...")
    attention_model = create_attention_visualization_model(phys_input_shape)
    
    # Copy weights from trained model
    attention_model.set_weights(phys_informed_model.get_weights())
    
    # Get predictions and attention weights
    predictions = attention_model.predict(X_test_phys, verbose=0)
    y_pred_sbp, y_pred_dbp, attention_weights = predictions[0], predictions[1], predictions[2]
    
    # Perform comprehensive evaluation
    test_results = comprehensive_evaluation(
        attention_model,
        X_test_phys,
        {'sbp': y_test_sbp, 'dbp': y_test_dbp},
        dataset_name="Test",
        visualize_attention=True,
        save_dir=CHECKPOINT_DIR
    )
    
    # ===== Step 14: Final Summary =====
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"\n📊 Validation Set Performance:")
    print(f"   SBP - MAE: {val_results['metrics']['sbp']['MAE']:.2f} mmHg, RMSE: {val_results['metrics']['sbp']['RMSE']:.2f} mmHg, R²: {val_results['metrics']['sbp']['R2']:.4f}")
    print(f"   DBP - MAE: {val_results['metrics']['dbp']['MAE']:.2f} mmHg, RMSE: {val_results['metrics']['dbp']['RMSE']:.2f} mmHg, R²: {val_results['metrics']['dbp']['R2']:.4f}")
    
    print(f"\n📊 Test Set Performance:")
    print(f"   SBP - MAE: {test_results['metrics']['sbp']['MAE']:.2f} mmHg, RMSE: {test_results['metrics']['sbp']['RMSE']:.2f} mmHg, R²: {test_results['metrics']['sbp']['R2']:.4f}")
    print(f"   DBP - MAE: {test_results['metrics']['dbp']['MAE']:.2f} mmHg, RMSE: {test_results['metrics']['dbp']['RMSE']:.2f} mmHg, R²: {test_results['metrics']['dbp']['R2']:.4f}")
    
    sbp_success = test_results['metrics']['sbp']['MAE'] < 10.0
    dbp_success = test_results['metrics']['dbp']['MAE'] < 10.0
    
    if sbp_success and dbp_success:
        print(f"\n🎉 SUCCESS! Both SBP and DBP MAE < 10 mmHg target achieved!")
    elif sbp_success:
        print(f"\n✅ SBP target achieved! DBP needs improvement.")
    elif dbp_success:
        print(f"\n✅ DBP target achieved! SBP needs improvement.")
    else:
        print(f"\n⚠️  Targets not yet met. Consider:")
        print(f"   - Increasing model capacity")
        print(f"   - Adding more training data")
        print(f"   - Tuning hyperparameters")
        print(f"   - Improving feature engineering")
    
    print(f"\n💾 Model and results saved to: {CHECKPOINT_DIR}")
    
    # ===== Step 15: Save Run Metadata =====
    print("\n💾 STEP 15: Saving run metadata...")
    
    metadata = {
        'run_timestamp': training_start_time.isoformat(),
        'training_mode': 'resumed' if is_resuming else 'fresh',
        'two_phase_training': ENABLE_TWO_PHASE_TRAINING and is_resuming,
        'initial_learning_rate': float(initial_lr),
        'final_learning_rate': float(K.get_value(phys_informed_model.optimizer.learning_rate)),
        'training_duration_seconds': training_duration,
        'epochs_trained': len(history.history['loss']),
        'final_metrics': {
            'validation': {
                'sbp_mae': float(val_results['metrics']['sbp']['MAE']),
                'sbp_rmse': float(val_results['metrics']['sbp']['RMSE']),
                'sbp_r2': float(val_results['metrics']['sbp']['R2']),
                'dbp_mae': float(val_results['metrics']['dbp']['MAE']),
                'dbp_rmse': float(val_results['metrics']['dbp']['RMSE']),
                'dbp_r2': float(val_results['metrics']['dbp']['R2'])
            },
            'test': {
                'sbp_mae': float(test_results['metrics']['sbp']['MAE']),
                'sbp_rmse': float(test_results['metrics']['sbp']['RMSE']),
                'sbp_r2': float(test_results['metrics']['sbp']['R2']),
                'dbp_mae': float(test_results['metrics']['dbp']['MAE']),
                'dbp_rmse': float(test_results['metrics']['dbp']['RMSE']),
                'dbp_r2': float(test_results['metrics']['dbp']['R2'])
            }
        },
        'target_achieved': {
            'sbp': test_results['metrics']['sbp']['MAE'] < 10.0,
            'dbp': test_results['metrics']['dbp']['MAE'] < 10.0
        },
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'resume_lr_reduction_factor': RESUME_LR_REDUCTION_FACTOR
        }
    }
    
    metadata_path = os.path.join(CHECKPOINT_DIR, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Metadata saved to: {metadata_path}")
    print("="*70 + "\n")
    
    return phys_informed_model, history, test_results


if __name__ == "__main__":
    model, history, results = main()
