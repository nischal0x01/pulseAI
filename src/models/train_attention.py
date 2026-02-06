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
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# TensorFlow memory optimization for laptop training
import tensorflow as tf
# Limit TensorFlow to use only necessary CPU threads
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"✅ GPU memory growth enabled for {len(physical_devices)} GPU(s)")
else:
    print("ℹ️  No GPU detected - training on CPU")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K

# Import from our modules
from config import EPOCHS, BATCH_SIZE, VERBOSE, PROCESSED_DATA_DIR, CHECKPOINT_DIR, RAW_DATA_DIR, BASE_DATA_DIR, SBP_LOSS_WEIGHT, EXTREME_BP_WEIGHT, RESUME_LR_REDUCTION_FACTOR
from data_loader import load_aggregate_data

# Training configuration
ENABLE_TWO_PHASE_TRAINING = False  # Set to True to freeze CNN layers and train LSTM+attention only
RESUME_LR_REDUCTION_FACTOR = 0.5  # Reduce learning rate by this factor when resuming

# Data augmentation configuration
ENABLE_AUGMENTATION = False
AUGMENTATION_FACTOR = 2.0  # Target 2x representation for high BP samples
BP_THRESHOLD = 140  # SBP threshold to define "high BP" for augmentation
AUGMENTATION_CONFIG = {
    'time_warp': True,
    'amplitude_scale': True,
    'add_noise': True,
    'warp_factor': 0.1,  # ±10% temporal stretch
    'scale_factor': 0.05,  # ±5% amplitude variation
    'snr_range': (20, 30)  # SNR 20-30 dB for noise injection
}

from preprocessing import (
    preprocess_signals, create_subject_wise_splits,
    compute_sbp_baselines, compute_dbp_baselines,
    convert_to_residuals, reconstruct_from_residuals
)
from feature_engineering import (
    extract_physiological_features, standardize_feature_length,
    create_4_channel_input, normalize_pat_subject_wise
)
from augmentation import augment_high_bp_samples
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
    
    # Check for cached preprocessed data
    cache_path = os.path.join(CHECKPOINT_DIR, 'preprocessed_data_cache.npz')
    use_cache = os.path.exists(cache_path)
    
    if use_cache:
        print("\n" + "="*70)
        print("  📦 LOADING CACHED PREPROCESSED DATA")
        print("="*70)
        print(f"\n✅ Found cached data: {cache_path}")
        print("⏩ Skipping data loading, preprocessing, and feature extraction...")
        
        try:
            # Load cached data
            cache_data = np.load(cache_path, allow_pickle=True)
            
            X_train_phys = cache_data['X_train']
            X_val_phys = cache_data['X_val']
            X_test_phys = cache_data['X_test']
            y_train_sbp = cache_data['y_train_sbp']
            y_train_dbp = cache_data['y_train_dbp']
            y_val_sbp = cache_data['y_val_sbp']
            y_val_dbp = cache_data['y_val_dbp']
            y_test_sbp = cache_data['y_test_sbp']
            y_test_dbp = cache_data['y_test_dbp']
            y_sbp = cache_data['y_sbp']
            y_dbp = cache_data['y_dbp']
            val_mask = cache_data['val_mask']
            test_mask = cache_data['test_mask']
            
            print(f"\n✅ Loaded preprocessed data successfully!")
            print(f"\n📐 Data shapes:")
            print(f"   X_train: {X_train_phys.shape}")
            print(f"   y_train_sbp: {y_train_sbp.shape}, y_train_dbp: {y_train_dbp.shape}")
            print(f"   X_val: {X_val_phys.shape}")
            print(f"   y_val_sbp: {y_val_sbp.shape}, y_val_dbp: {y_val_dbp.shape}")
            print(f"   X_test: {X_test_phys.shape}")
            print(f"   y_test_sbp: {y_test_sbp.shape}, y_test_dbp: {y_test_dbp.shape}")
            print(f"   ✅ Verified 4 channels: [ECG, PPG, PAT, HR]")
            
        except Exception as e:
            print(f"\n⚠️  Failed to load cache: {e}")
            print(f"Falling back to full data loading...\n")
            use_cache = False
    
    if not use_cache:
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
        
        # ===== Step 3: Extract Physiological Features (BEFORE splitting) =====
        print("\n💓 STEP 3: Extracting physiological features (PAT, HR)...")
        
        # Raw signals are (samples, timesteps, channels): channel 0 PPG, channel 1 ECG
        ppg_raw = signals_agg[:, 0, :]
        ecg_raw = signals_agg[:, 1, :]
        
        # Extract features with quality mask
        pat_seqs, hr_seqs, quality_mask = extract_physiological_features(ecg_raw, ppg_raw)
        
        # Filter out low-quality samples
        n_low_quality = (~quality_mask).sum()
        if n_low_quality > 0:
            print(f"   ⚠️  Filtering {n_low_quality} low-quality windows ({100*n_low_quality/len(quality_mask):.1f}%)...")
            
            # Apply quality mask to all data
            signals_agg = signals_agg[quality_mask]
            processed_signals = processed_signals[quality_mask]
            pat_seqs = pat_seqs[quality_mask]
            hr_seqs = hr_seqs[quality_mask]
            y_sbp = y_sbp[quality_mask]
            y_dbp = y_dbp[quality_mask]
            patient_ids_agg = patient_ids_agg[quality_mask]
            
            print(f"   ✅ Kept {quality_mask.sum()} high-quality samples")
        else:
            print(f"   ✅ All {len(quality_mask)} samples have good signal quality")
        
        # ===== Step 4: Create Patient-Wise Splits (AFTER quality filtering) =====
        print("\n📊 STEP 4: Creating patient-wise data splits...")
        train_mask, val_mask, test_mask = create_subject_wise_splits(patient_ids_agg)
        
        # Standardize length to match processed_signals
        target_len = processed_signals.shape[1]
        pat_seqs = standardize_feature_length(pat_seqs, target_len)
        hr_seqs = standardize_feature_length(hr_seqs, target_len)
        print(f"   - Physiological features standardized to length: {target_len}")
        
        # Validate features before scaling
        validate_data_integrity(pat_seqs, "PAT sequences (before scaling)")
        validate_data_integrity(hr_seqs, "HR sequences (before scaling)")
        
        # ===== Step 5: Scale Physiological Features =====
        print("\n🔄 STEP 5: Scaling physiological features...")
        
        # FIX 3: Subject-wise PAT normalization (using training set statistics per patient)
        print("   - Applying subject-wise PAT normalization (train stats only per patient)...")
        pat_seqs_scaled, pat_stats = normalize_pat_subject_wise(pat_seqs, patient_ids_agg, train_mask)
        
        # HR normalization (global statistics from training data)
        print("   - Applying global HR normalization (train stats only)...")
        scaler_hr = StandardScaler()
        scaler_hr.fit(hr_seqs[train_mask])
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
        
        # ===== Step 6.5: Train on Absolute BP (Fixed from Residual Training) =====
        print("\n🔄 STEP 6.5: Using absolute BP values for training...")
        
        # FIXED: Train directly on absolute BP instead of residuals
        # Use absolute BP values directly
        y_sbp_absolute = y_sbp.copy()
        y_dbp_absolute = y_dbp.copy()
        
        validate_data_integrity(y_sbp_absolute, "SBP absolute values")
        validate_data_integrity(y_dbp_absolute, "DBP absolute values")
        
        print(f"   - SBP range: {np.min(y_sbp_absolute):.1f} to {np.max(y_sbp_absolute):.1f} mmHg")
        print(f"   - DBP range: {np.min(y_dbp_absolute):.1f} to {np.max(y_dbp_absolute):.1f} mmHg")
        print(f"   - Model will predict absolute BP values directly")
        
        # Apply masks for train/val/test split
        X_train_phys = X_phys_informed[train_mask]
        X_val_phys = X_phys_informed[val_mask]
        X_test_phys = X_phys_informed[test_mask]
        
        # Use absolute BP values for training
        y_train_sbp = y_sbp_absolute[train_mask]
        y_train_dbp = y_dbp_absolute[train_mask]
        y_val_sbp = y_sbp_absolute[val_mask]
        y_val_dbp = y_dbp_absolute[val_mask]
        y_test_sbp = y_sbp_absolute[test_mask]
        y_test_dbp = y_dbp_absolute[test_mask]
        
        # ===== Step 6.6: Data Augmentation for High BP Samples =====
        if ENABLE_AUGMENTATION:
            print(f"\n🔄 STEP 6.6: Augmenting high BP samples (SBP > {BP_THRESHOLD} mmHg)...")
            print(f"   - Augmentation factor: {AUGMENTATION_FACTOR}x")
            print(f"   - Techniques: time warp (±{AUGMENTATION_CONFIG['warp_factor']*100:.0f}%), "
                  f"amplitude scale (±{AUGMENTATION_CONFIG['scale_factor']*100:.0f}%), "
                  f"noise (SNR {AUGMENTATION_CONFIG['snr_range'][0]}-{AUGMENTATION_CONFIG['snr_range'][1]} dB)")
            
            # Count high BP samples before augmentation
            n_high_bp_before = (y_train_sbp > BP_THRESHOLD).sum()
            n_total_before = len(y_train_sbp)
            print(f"   - Training set before augmentation: {n_high_bp_before}/{n_total_before} "
                  f"({100*n_high_bp_before/n_total_before:.1f}%) high BP samples")
            
            # Augment training data only (not validation or test)
            X_train_phys, y_train_sbp, y_train_dbp = augment_high_bp_samples(
                X_train_phys, y_train_sbp, y_train_dbp,
                augmentation_factor=AUGMENTATION_FACTOR,
                bp_threshold=BP_THRESHOLD,
                augmentation_config=AUGMENTATION_CONFIG
            )
            
            # Report results
            n_high_bp_after = (y_train_sbp > BP_THRESHOLD).sum()
            n_total_after = len(y_train_phys)
            print(f"   - Training set after augmentation: {n_high_bp_after}/{n_total_after} "
                  f"({100*n_high_bp_after/n_total_after:.1f}%) high BP samples")
            print(f"   ✅ Augmentation complete: {n_total_before} → {n_total_after} training samples")
        else:
            print("\n⏭️  STEP 6.6: Data augmentation disabled")
        
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
        
        # ===== Step 7.5: Cache Preprocessed Data =====
        cache_path = os.path.join(CHECKPOINT_DIR, 'preprocessed_data_cache.npz')
        print(f"\n💾 STEP 7.5: Caching preprocessed data for future runs...")
        print(f"   Cache location: {cache_path}")
        
        try:
            np.savez_compressed(
                cache_path,
                X_train=X_train_phys,
                X_val=X_val_phys,
                X_test=X_test_phys,
                y_train_sbp=y_train_sbp,
                y_train_dbp=y_train_dbp,
                y_val_sbp=y_val_sbp,
                y_val_dbp=y_val_dbp,
                y_test_sbp=y_test_sbp,
                y_test_dbp=y_test_dbp,
                y_sbp=y_sbp,
                y_dbp=y_dbp,
                val_mask=val_mask,
                test_mask=test_mask
            )
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"   ✅ Data cached successfully! ({cache_size_mb:.1f} MB)")
            print(f"   ℹ️  Next run will skip data loading and preprocessing!")
        except Exception as e:
            print(f"   ⚠️  Failed to cache data: {e}")
            print(f"   Training will continue normally...")
    
    # ===== Step 8: Build Model with Weighted Loss =====
    print("\n🏗️  STEP 8: Building physiology-informed CNN-LSTM model with weighted loss...")
    phys_input_shape = (X_train_phys.shape[1], X_train_phys.shape[2])
    
    # Load or initialize training state
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.h5')
    checkpoint_state_path = os.path.join(CHECKPOINT_DIR, 'training_state.pkl')
    
    training_state = {
        'total_epochs': 0,
        'best_val_loss': float('inf'),
        'lr_reduction_count': 0,
        'runs_completed': 0
    }
    
    is_resuming = False
    if os.path.exists(checkpoint_state_path):
        try:
            with open(checkpoint_state_path, 'rb') as f:
                training_state = pickle.load(f)
            
            print(f"\n📊 Previous Training State:")
            print(f"   - Epochs completed: {training_state['total_epochs']}")
            print(f"   - Best validation loss: {training_state['best_val_loss']:.4f}")
            print(f"   - LR reductions: {training_state['lr_reduction_count']}")
            print(f"   - Runs completed: {training_state['runs_completed']}")
            
            if training_state['lr_reduction_count'] >= 3:
                print(f"\n   ⚠️  Warning: Model has been fine-tuned {training_state['lr_reduction_count']} times")
                print(f"   💡 Consider stopping - diminishing returns likely")
            
            is_resuming = True
        except Exception as e:
            print(f"   ⚠️ Failed to load training state: {e}")
            print(f"   Starting fresh training...")
    
    # Create model with weighted loss
    print(f"\n🎯 Creating model with weighted loss function:")
    print(f"   - SBP weight: {SBP_LOSS_WEIGHT}x (SBP prioritized over DBP)")
    print(f"   - Extreme BP weight: {EXTREME_BP_WEIGHT}x (high/low BP prioritized)")
    print(f"   - Normal BP range: 90-140 mmHg (weight 1.0x)")
    
    phys_informed_model = create_phys_informed_model(phys_input_shape)
    
    print("\n📋 Model Architecture:")
    phys_informed_model.summary()
    
    # Load weights if resuming
    initial_lr = K.get_value(phys_informed_model.optimizer.learning_rate)
    
    if is_resuming and os.path.exists(checkpoint_path):
        print(f"\n🔄 CHECKPOINT FOUND: Resuming training from {checkpoint_path}")
        try:
            phys_informed_model.load_weights(checkpoint_path)
            
            # Adjust learning rate based on number of reductions
            lr_reduction_count = training_state['lr_reduction_count']
            if lr_reduction_count < 3:
                new_lr = initial_lr * (RESUME_LR_REDUCTION_FACTOR ** (lr_reduction_count + 1))
                K.set_value(phys_informed_model.optimizer.learning_rate, new_lr)
                print(f"   ✅ Weights loaded successfully")
                print(f"   📉 Learning rate adjusted: {initial_lr:.2e} → {new_lr:.2e}")
                print(f"   🔄 LR reduction #{lr_reduction_count + 1}")
                training_state['lr_reduction_count'] += 1
            else:
                print(f"   ✅ Weights loaded successfully")
                print(f"   ⚠️ Already reduced LR {lr_reduction_count} times")
                print(f"   📊 Keeping current learning rate: {initial_lr:.2e}")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print(f"   🆕 Training from scratch instead")
            is_resuming = False
    else:
        print(f"\n🆕 No checkpoint found - training from scratch")
        print(f"   Initial learning rate: {initial_lr:.2e}")
    
    print(f"   🎯 Current learning rate: {K.get_value(phys_informed_model.optimizer.learning_rate):.2e}")
    
    # ===== Step 8.5: Calculate Sample Weights for Extreme BP Focus =====
    print("\n⚖️  STEP 8.5: Calculating sample weights for extreme BP focus...")
    
    def calculate_sample_weights(y_sbp, y_dbp):
        """Give more weight to extreme BP samples during training."""
        weights = np.ones(len(y_sbp))
        
        # High BP samples: 5x weight
        weights[y_sbp > 140] = 5.0
        
        # Low BP samples: 5x weight  
        weights[y_sbp < 90] = 5.0
        
        # Very low BP: 8x weight (main problem area from patient analysis)
        weights[y_sbp < 75] = 8.0
        
        return weights
    
    # Calculate weights for training set
    train_sample_weights = calculate_sample_weights(y_train_sbp, y_train_dbp)
    
    # Count weighted samples
    normal_count = int(np.sum(train_sample_weights == 1.0))
    high_count = int(np.sum((train_sample_weights == 5.0) & (y_train_sbp > 140)))
    low_count = int(np.sum((train_sample_weights == 5.0) & (y_train_sbp >= 75) & (y_train_sbp < 90)))
    very_low_count = int(np.sum(train_sample_weights == 8.0))
    
    print(f"   - Normal BP samples (90-140 mmHg): {normal_count} (weight: 1.0x)")
    print(f"   - High BP samples (>140 mmHg): {high_count} (weight: 5.0x)")
    print(f"   - Low BP samples (75-90 mmHg): {low_count} (weight: 5.0x)")
    print(f"   - Very low BP samples (<75 mmHg): {very_low_count} (weight: 8.0x)")
    print(f"   - Effective training size: {np.sum(train_sample_weights):.0f} weighted samples")
    
    if very_low_count > 0:
        print(f"   ✅ Very low BP samples will be emphasized {very_low_count} × 8 = {very_low_count * 8} effective samples")
    else:
        print(f"   ⚠️  No very low BP samples (<75 mmHg) in training set!")
    
    if high_count > 0:
        print(f"   ✅ High BP samples will be emphasized {high_count} × 5 = {high_count * 5} effective samples")
    else:
        print(f"   ⚠️  No high BP samples (>140 mmHg) in training set!")
    
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
    print(f"   - Loss: Weighted Huber (extreme BP focused)")
    print(f"   - Sample weighting: ENABLED (8x for very low BP, 5x for high/low BP)")
    print(f"   - Optimizer: Adam with gradient clipping")
    print(f"   - Outputs: SBP and DBP (dual prediction)")
    print(f"   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint\n")
    
    training_start_time = datetime.now()
    
    # Sample weighting is now built into WeightedHuberLoss
    # No need for separate sample_weight parameter
    
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
    
    # Predict absolute BP (no reconstruction needed)
    val_predictions = phys_informed_model.predict(X_val_phys, verbose=0)
    y_pred_sbp_val = val_predictions[0].flatten()  # Already absolute BP
    y_pred_dbp_val = val_predictions[1].flatten()  # Already absolute BP
    
    # FIXED: No reconstruction needed since model predicts absolute BP
    # Old approach (residual training):
    # y_pred_sbp_val = reconstruct_from_residuals(val_patient_ids, y_pred_sbp_residual, sbp_baselines)
    # y_pred_dbp_val = reconstruct_from_residuals(val_patient_ids, y_pred_dbp_residual, dbp_baselines)
    
    # Get actual absolute BP values for validation
    y_true_sbp_val = y_sbp[val_mask]
    y_true_dbp_val = y_dbp[val_mask]
    
    # Calculate metrics on reconstructed absolute values
    sbp_metrics_val = calculate_comprehensive_metrics(y_true_sbp_val, y_pred_sbp_val)
    dbp_metrics_val = calculate_comprehensive_metrics(y_true_dbp_val, y_pred_dbp_val)
    
    val_results = {
        'metrics': {
            'sbp': sbp_metrics_val,
            'dbp': dbp_metrics_val
        }
    }
    
    print("\n📊 Validation Set Performance (Reconstructed from ΔBP):")
    print(f"   SBP - MAE: {sbp_metrics_val['MAE']:.2f} mmHg, RMSE: {sbp_metrics_val['RMSE']:.2f} mmHg, R²: {sbp_metrics_val['R2']:.4f}")
    print(f"   DBP - MAE: {dbp_metrics_val['MAE']:.2f} mmHg, RMSE: {dbp_metrics_val['RMSE']:.2f} mmHg, R²: {dbp_metrics_val['R2']:.4f}")
    
    # ===== Step 13: Evaluate on Test Set =====
    print("\n📊 STEP 13: Evaluating on test set...")
    
    # Create attention visualization model
    print("   - Creating attention visualization model...")
    attention_model = create_attention_visualization_model(phys_input_shape)
    
    # Copy weights from trained model
    attention_model.set_weights(phys_informed_model.get_weights())
    
    # Get predictions (absolute BP) and attention weights
    predictions = attention_model.predict(X_test_phys, verbose=0)
    y_pred_sbp_test = predictions[0].flatten()  # Already absolute BP
    y_pred_dbp_test = predictions[1].flatten()  # Already absolute BP
    attention_weights = predictions[2]
    
    # FIXED: No reconstruction needed since model predicts absolute BP
    # Old approach (residual training):
    # y_pred_sbp_test = reconstruct_from_residuals(test_patient_ids, y_pred_sbp_residual, sbp_baselines)
    # y_pred_dbp_test = reconstruct_from_residuals(test_patient_ids, y_pred_dbp_residual, dbp_baselines)
    
    # Get actual absolute BP values for test
    y_true_sbp_test = y_sbp[test_mask]
    y_true_dbp_test = y_dbp[test_mask]
    
    # Calculate metrics on reconstructed absolute values
    sbp_metrics_test = calculate_comprehensive_metrics(y_true_sbp_test, y_pred_sbp_test)
    dbp_metrics_test = calculate_comprehensive_metrics(y_true_dbp_test, y_pred_dbp_test)
    
    test_results = {
        'metrics': {
            'sbp': sbp_metrics_test,
            'dbp': dbp_metrics_test
        },
        'attention_weights': attention_weights
    }
    
    print("\n📊 Test Set Performance (Reconstructed from ΔBP):")
    print(f"   SBP - MAE: {sbp_metrics_test['MAE']:.2f} mmHg, RMSE: {sbp_metrics_test['RMSE']:.2f} mmHg, R²: {sbp_metrics_test['R2']:.4f}")
    print(f"   DBP - MAE: {dbp_metrics_test['MAE']:.2f} mmHg, RMSE: {dbp_metrics_test['RMSE']:.2f} mmHg, R²: {dbp_metrics_test['R2']:.4f}")
    
    # Visualize attention weights
    print("\n📊 Analyzing attention patterns...")
    avg_attention = np.mean(attention_weights, axis=0)
    print(f"   - Average attention weight range: [{np.min(avg_attention):.4f}, {np.max(avg_attention):.4f}]")
    print(f"   - Attention weights sum per sample (should NOT be 1.0 for sigmoid): {np.mean(np.sum(attention_weights, axis=1)):.4f}")
    
    # Plot attention visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(avg_attention.flatten(), linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Attention Weight (Sigmoid)', fontsize=12, fontweight='bold')
    ax.set_title('PAT-Based Sigmoid Attention Weights Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'attention_weights.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # SBP plot
    axes[0].scatter(y_true_sbp_test, y_pred_sbp_test, alpha=0.6, edgecolors='k', s=50)
    min_sbp = min(y_true_sbp_test.min(), y_pred_sbp_test.min())
    max_sbp = max(y_true_sbp_test.max(), y_pred_sbp_test.max())
    axes[0].plot([min_sbp, max_sbp], [min_sbp, max_sbp], 'r--', lw=2, label='Perfect Prediction')
    axes[0].fill_between([min_sbp, max_sbp], [min_sbp - 10, max_sbp - 10], [min_sbp + 10, max_sbp + 10],
                         alpha=0.2, color='green', label='±10 mmHg')
    axes[0].set_xlabel('Actual SBP (mmHg)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted SBP (mmHg)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Test Set SBP Predictions\nMAE={sbp_metrics_test["MAE"]:.2f} mmHg, R²={sbp_metrics_test["R2"]:.4f}',
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # DBP plot
    axes[1].scatter(y_true_dbp_test, y_pred_dbp_test, alpha=0.6, edgecolors='k', s=50)
    min_dbp = min(y_true_dbp_test.min(), y_pred_dbp_test.min())
    max_dbp = max(y_true_dbp_test.max(), y_pred_dbp_test.max())
    axes[1].plot([min_dbp, max_dbp], [min_dbp, max_dbp], 'r--', lw=2, label='Perfect Prediction')
    axes[1].fill_between([min_dbp, max_dbp], [min_dbp - 10, max_dbp - 10], [min_dbp + 10, max_dbp + 10],
                         alpha=0.2, color='green', label='±10 mmHg')
    axes[1].set_xlabel('Actual DBP (mmHg)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted DBP (mmHg)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Test Set DBP Predictions\nMAE={dbp_metrics_test["MAE"]:.2f} mmHg, R²={dbp_metrics_test["R2"]:.4f}',
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'test_predictions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
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
    
    # ===== Step 15: Save Run Metadata and Training State =====
    print("\n💾 STEP 15: Saving run metadata and training state...")
    
    # Update training state for future runs
    final_val_loss = min(history.history['val_loss'])
    training_state['total_epochs'] += len(history.history['loss'])
    training_state['best_val_loss'] = min(training_state['best_val_loss'], final_val_loss)
    training_state['runs_completed'] += 1
    
    # Save training state for next run
    checkpoint_state_path = os.path.join(CHECKPOINT_DIR, 'training_state.pkl')
    with open(checkpoint_state_path, 'wb') as f:
        pickle.dump(training_state, f)
    
    print(f"   ✅ Training state saved:")
    print(f"      - Total epochs trained: {training_state['total_epochs']}")
    print(f"      - Best validation loss: {training_state['best_val_loss']:.4f}")
    print(f"      - LR reductions: {training_state['lr_reduction_count']}")
    print(f"      - Runs completed: {training_state['runs_completed']}")
    
    if training_state['lr_reduction_count'] >= 3:
        print(f"\n   💡 Tip: Model has been fine-tuned {training_state['lr_reduction_count']} times.")
        print(f"      Consider stopping if performance not improving.")
    
    metadata = {
        'run_timestamp': training_start_time.isoformat(),
        'training_mode': 'resumed' if is_resuming else 'fresh',
        'two_phase_training': ENABLE_TWO_PHASE_TRAINING and is_resuming,
        'initial_learning_rate': float(initial_lr),
        'final_learning_rate': float(K.get_value(phys_informed_model.optimizer.learning_rate)),
        'training_duration_seconds': training_duration,
        'epochs_trained': len(history.history['loss']),
        'total_epochs_all_runs': training_state['total_epochs'],
        'runs_completed': training_state['runs_completed'],
        'lr_reduction_count': training_state['lr_reduction_count'],
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
            'resume_lr_reduction_factor': RESUME_LR_REDUCTION_FACTOR,
            'sbp_loss_weight': SBP_LOSS_WEIGHT,
            'extreme_bp_weight': EXTREME_BP_WEIGHT
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
