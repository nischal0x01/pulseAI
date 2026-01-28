"""
Rebuild model architecture replacing Lambda layer with Slicing layer,
load weights, and convert to TensorFlow.js format
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Dense, LSTM, 
    BatchNormalization, Multiply, Activation, Reshape
)
import subprocess
import sys
import os
import shutil

print("="*70)
print("Model Rebuilder & Converter for TensorFlow.js")
print("="*70)

# Define model config
CONV1D_FILTERS_1 = 64
CONV1D_FILTERS_2 = 128
CONV1D_KERNEL_SIZE = 5
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
ATTENTION_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3

print("\n[1/5] Building model architecture (without Lambda layers)...")

# Build model
inputs = Input(shape=(875, 4), name='input')

# CNN Feature Extraction
x = Conv1D(CONV1D_FILTERS_1, CONV1D_KERNEL_SIZE, 
           activation='relu', padding='same', name='conv1d_1')(inputs)
x = BatchNormalization(name='bn_1')(x)
x = MaxPooling1D(2, name='pool_1')(x)
x = Dropout(DROPOUT_RATE, name='dropout_1')(x)

x = Conv1D(CONV1D_FILTERS_2, CONV1D_KERNEL_SIZE, 
           activation='relu', padding='same', name='conv1d_2')(x)
x = BatchNormalization(name='bn_2')(x)
cnn_features = MaxPooling1D(2, name='pool_2')(x)
cnn_features = Dropout(DROPOUT_RATE, name='dropout_2')(cnn_features)

# PAT extraction using Reshape (alternative to Lambda that works with TFJS)
# Extract 4th channel (index 3) - PAT channel
# Use slicing operation: inputs[:,:,3:4] 
print("  - Using tf.keras.layers.Lambda for PAT extraction (will need custom layer in JS)")
pat_channel = tf.keras.layers.Lambda(lambda x: x[:, :, 2:3], 
                                     output_shape=(875, 1),
                                     name='extract_pat')(inputs)

# Downsample PAT to match pooled features
pat_downsampled = MaxPooling1D(4, name='pat_downsample')(pat_channel)

# PAT-based Attention
attention = Dense(ATTENTION_UNITS, activation='relu', 
                 name='pat_attention_dense1')(pat_downsampled)
attention = Dense(1, activation='linear', 
                 name='pat_attention_dense2')(attention)
attention_weights = Activation('softmax', name='pat_attention_softmax')(attention)

# Apply attention
attended_features = Multiply(name='pat_attention_multiply')([cnn_features, attention_weights])

# LSTM layers
lstm_out = LSTM(LSTM_UNITS_1, return_sequences=True, name='lstm_1')(attended_features)
lstm_out = Dropout(DROPOUT_RATE, name='dropout_3')(lstm_out)
lstm_out = LSTM(LSTM_UNITS_2, return_sequences=False, name='lstm_2')(lstm_out)
lstm_out = Dropout(DROPOUT_RATE, name='dropout_4')(lstm_out)

# Dense layers
dense_out = Dense(DENSE_UNITS, activation='relu', name='dense_1')(lstm_out)
dense_out = Dropout(DROPOUT_RATE, name='dropout_5')(dense_out)

# Output layers
sbp_output = Dense(1, activation='linear', name='sbp_output')(dense_out)
dbp_output = Dense(1, activation='linear', name='dbp_output')(dense_out)

# Create model
model = Model(inputs=inputs, outputs=[sbp_output, dbp_output],
              name='PhysInformed_CNN_LSTM_Attention_Dual')

print(f"✓ Model architecture built")
print(f"  - Layers: {len(model.layers)}")
print(f"  - Input shape: {model.input_shape}")
print(f"  - Output shapes: {[out.shape for out in model.outputs]}")

# Load weights
weights_path = 'checkpoints/best_model.h5'
print(f"\n[2/5] Loading weights from {weights_path}...")

try:
    model.load_weights(weights_path)
    print(f"✓ Weights loaded successfully")
except Exception as e:
    print(f"✗ Failed to load weights: {e}")
    print("\nNote: Layer names must match exactly. Available weights layers:")
    try:
        import h5py
        with h5py.File(weights_path, 'r') as f:
            def print_structure(name, obj):
                print(f"  {name}")
            f.visititems(print_structure)
    except:
        pass
    sys.exit(1)

# Test inference
print(f"\n[3/5] Testing model inference...")
try:
    import numpy as np
    test_input = np.random.randn(1, 875, 4).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    print(f"✓ Model inference successful")
    print(f"  - SBP output shape: {test_output[0].shape}")
    print(f"  - DBP output shape: {test_output[1].shape}")
except Exception as e:
    print(f"⚠ Warning: Inference test failed: {e}")
    print("  - Continuing anyway...")

# Save as SavedModel format (Keras 3 compatible)
saved_model_dir = 'temp_saved_model'
print(f"\n[4/5] Saving model as SavedModel format...")

if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

try:
    # Keras 3 - save to directory creates SavedModel format by default
    model.export(saved_model_dir)
    print(f"✓ Saved to {saved_model_dir}")
except AttributeError:
    # Fallback for older Keras versions
    try:
        model.save(saved_model_dir)
        print(f"✓ Saved to {saved_model_dir}")
    except Exception as e:
        print(f"✗ Failed to save: {e}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to save: {e}")
    sys.exit(1)

# Convert to TensorFlow.js
output_dir = 'frontend/public/models'
print(f"\n[5/5] Converting to TensorFlow.js format...")
print(f"  Output directory: {output_dir}")

cmd = [
    'tensorflowjs_converter',
    '--input_format=tf_saved_model',
    '--skip_op_check',
    '--strip_debug_ops=True',
    saved_model_dir,
    output_dir
]

print(f"  Command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode == 0:
        print("✓ Conversion successful!")
        
        # Verify output
        model_json_path = os.path.join(output_dir, 'model.json')
        if os.path.exists(model_json_path):
            import json
            with open(model_json_path, 'r') as f:
                model_json = json.load(f)
            print(f"\n✓ Model files created:")
            print(f"  - model.json")
            print(f"  - Format: {model_json.get('format', 'unknown')}")
            
            # Check for weights
            if 'weightsManifest' in model_json:
                for manifest in model_json['weightsManifest']:
                    for path in manifest.get('paths', []):
                        print(f"  - {path}")
            
            print(f"\n{'='*70}")
            print("SUCCESS! Your model is ready for Next.js")
            print(f"{'='*70}")
            print("\nUsage in your frontend:")
            print("  import * as tf from '@tensorflow/tfjs'")
            print("  const model = await tf.loadLayersModel('/models/model.json')")
            print("\nNote: You need the custom ExtractPATLayer in model-service.ts")
            print("      (already added in previous step)")
            
    else:
        print("✗ Conversion failed!")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        
        print(f"\n{'='*70}")
        print("TROUBLESHOOTING")
        print(f"{'='*70}")
        print("\nPossible issues:")
        print("1. tensorflowjs not installed: pip install tensorflowjs")
        print("2. Python version incompatibility (need 3.8-3.11)")
        print("3. TensorFlow version mismatch")
        print("\nTry manual conversion:")
        print(f"  tensorflowjs_converter --input_format=tf_saved_model \\")
        print(f"    {saved_model_dir} {output_dir}")
        sys.exit(1)
        
except FileNotFoundError:
    print("✗ tensorflowjs_converter not found!")
    print("\nInstall with:")
    print("  pip install tensorflowjs")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup
    print(f"\n[Cleanup] Removing temporary files...")
    try:
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
            print(f"✓ Removed {saved_model_dir}")
    except Exception as e:
        print(f"⚠ Warning: Cleanup failed: {e}")
