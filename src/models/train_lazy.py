"""
Memory-efficient training using data generators.
Use this for large datasets (200+ patients) that don't fit in RAM.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime
import json
import pickle

# Memory optimization
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

from config import EPOCHS, BATCH_SIZE, PROCESSED_DATA_DIR, CHECKPOINT_DIR
from data_generator import LazyBPDataGenerator, split_patients_for_generators
from model_builder_attention import create_phys_informed_model

print("="*70)
print("  🚀 MEMORY-EFFICIENT TRAINING MODE")
print("  Using lazy loading generators for large datasets")
print("="*70)

# Setup
CACHE_DIR = os.path.join(CHECKPOINT_DIR, 'patient_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Split patients
print("\n📂 STEP 1: Splitting patients into train/val/test sets...")
train_files, val_files, test_files = split_patients_for_generators(
    PROCESSED_DATA_DIR,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
)

# Create generators
print("\n🔄 STEP 2: Creating data generators...")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Cache directory: {CACHE_DIR}")

train_generator = LazyBPDataGenerator(
    train_files,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    is_training=True
)

val_generator = LazyBPDataGenerator(
    val_files,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    is_training=False
)

test_generator = LazyBPDataGenerator(
    test_files,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    is_training=False
)

print(f"\n✅ Generators created:")
print(f"   Training: {len(train_generator)} batches (~{train_generator.total_samples} samples)")
print(f"   Validation: {len(val_generator)} batches (~{val_generator.total_samples} samples)")
print(f"   Test: {len(test_generator)} batches (~{test_generator.total_samples} samples)")

# Build model
print("\n🏗️  STEP 3: Building model...")
input_shape = (875, 4)  # (timesteps, channels)
model = create_phys_informed_model(input_shape)

print("\n📋 Model Architecture:")
model.summary()

# Load checkpoint if exists
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_lazy.keras')
checkpoint_state_path = os.path.join(CHECKPOINT_DIR, 'training_state_lazy.pkl')

training_state = {
    'total_epochs': 0,
    'best_val_loss': float('inf'),
    'runs_completed': 0
}

if os.path.exists(checkpoint_state_path):
    try:
        with open(checkpoint_state_path, 'rb') as f:
            training_state = pickle.load(f)
        
        print(f"\n📊 Previous Training State:")
        print(f"   Epochs completed: {training_state['total_epochs']}")
        print(f"   Best val loss: {training_state['best_val_loss']:.4f}")
        
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
            print(f"   ✅ Checkpoint loaded!")
    except Exception as e:
        print(f"   ⚠️  Failed to load checkpoint: {e}")

# Setup callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n🏃‍♂️ STEP 4: Training model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Starting from epoch: {training_state['total_epochs']}")
print(f"   Memory-efficient: Only {BATCH_SIZE} samples in RAM at a time")
print()

start_time = datetime.now()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    initial_epoch=training_state['total_epochs'],
    callbacks=callbacks,
    verbose=1
)

duration = (datetime.now() - start_time).total_seconds()

print(f"\n✅ Training completed in {duration/60:.1f} minutes")

# Update training state
training_state['total_epochs'] += len(history.history['loss'])
training_state['best_val_loss'] = min(history.history['val_loss'])
training_state['runs_completed'] += 1

with open(checkpoint_state_path, 'wb') as f:
    pickle.dump(training_state, f)

# Evaluate on test set
print("\n📊 STEP 5: Evaluating on test set...")
print("   Loading test data into memory...")
X_test, y_test_sbp, y_test_dbp = test_generator.get_all_data()

predictions = model.predict(X_test, batch_size=32, verbose=0)
y_pred_sbp = predictions[0].flatten()
y_pred_dbp = predictions[1].flatten()

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sbp_mae = mean_absolute_error(y_test_sbp, y_pred_sbp)
sbp_rmse = np.sqrt(mean_squared_error(y_test_sbp, y_pred_sbp))
sbp_r2 = r2_score(y_test_sbp, y_pred_sbp)

dbp_mae = mean_absolute_error(y_test_dbp, y_pred_dbp)
dbp_rmse = np.sqrt(mean_squared_error(y_test_dbp, y_pred_dbp))
dbp_r2 = r2_score(y_test_dbp, y_pred_dbp)

print("\n" + "="*70)
print("  📊 FINAL TEST RESULTS")
print("="*70)
print(f"SBP - MAE: {sbp_mae:.2f} mmHg, RMSE: {sbp_rmse:.2f} mmHg, R²: {sbp_r2:.4f}")
print(f"DBP - MAE: {dbp_mae:.2f} mmHg, RMSE: {dbp_rmse:.2f} mmHg, R²: {dbp_r2:.4f}")

if sbp_mae < 10 and dbp_mae < 10:
    print("\n🎉 SUCCESS! Both targets achieved!")
else:
    print(f"\n⚠️  Targets not yet met. Continue training or add more data.")

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'mode': 'lazy_loading',
    'total_patients': len(train_files) + len(val_files) + len(test_files),
    'train_patients': len(train_files),
    'val_patients': len(val_files),
    'test_patients': len(test_files),
    'epochs_trained': training_state['total_epochs'],
    'test_metrics': {
        'sbp_mae': float(sbp_mae),
        'sbp_rmse': float(sbp_rmse),
        'sbp_r2': float(sbp_r2),
        'dbp_mae': float(dbp_mae),
        'dbp_rmse': float(dbp_rmse),
        'dbp_r2': float(dbp_r2)
    }
}

with open(os.path.join(CHECKPOINT_DIR, 'lazy_training_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Results saved to: {CHECKPOINT_DIR}")
print("="*70)
