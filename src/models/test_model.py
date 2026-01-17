"""
Quick test script to verify model architecture and data flow.
Run this before full training to catch issues early.
"""

import numpy as np
import sys

print("Testing CNN-LSTM Attention Model Architecture...\n")

# Test imports
print("1. Testing imports...")
try:
    from config import *
    print("   ✅ config.py imported")
    
    from model_builder_attention import (
        create_phys_informed_model,
        create_attention_visualization_model
    )
    print("   ✅ model_builder_attention.py imported")
    
    from evaluation import calculate_comprehensive_metrics
    print("   ✅ evaluation.py imported")
    
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test model creation
print("\n2. Testing model creation...")
try:
    input_shape = (875, 4)  # (timesteps, channels)
    model = create_phys_informed_model(input_shape)
    print(f"   ✅ Model created with input shape: {input_shape}")
    print(f"   ✅ Model has {model.count_params():,} parameters")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# Test attention visualization model
print("\n3. Testing attention visualization model...")
try:
    attention_model = create_attention_visualization_model(input_shape)
    print(f"   ✅ Attention model created")
    print(f"   ✅ Attention model outputs: {len(attention_model.outputs)}")
except Exception as e:
    print(f"   ❌ Attention model creation failed: {e}")
    sys.exit(1)

# Test forward pass with dummy data
print("\n4. Testing forward pass with dummy data...")
try:
    # Create dummy 4-channel input
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 875, 4).astype(np.float32)
    
    # Ensure no NaN/Inf in dummy data
    assert not np.any(np.isnan(dummy_input))
    assert not np.any(np.isinf(dummy_input))
    
    # Test prediction
    predictions = model.predict(dummy_input, verbose=0)
    print(f"   ✅ Forward pass successful")
    print(f"   ✅ Prediction shape: {predictions.shape}")
    print(f"   ✅ Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Test attention model
    predictions_att, attention_weights = attention_model.predict(dummy_input, verbose=0)
    print(f"   ✅ Attention forward pass successful")
    print(f"   ✅ Attention weights shape: {attention_weights.shape}")
    print(f"   ✅ Attention weights sum (should be ~1.0): {attention_weights[0].sum():.4f}")
    
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    sys.exit(1)

# Test metrics calculation
print("\n5. Testing metrics calculation...")
try:
    # Dummy true and predicted values
    y_true = np.random.uniform(100, 160, size=(20, 1))
    y_pred = y_true + np.random.normal(0, 5, size=(20, 1))
    
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    print(f"   ✅ Metrics calculated")
    print(f"   ✅ MAE: {metrics['MAE']:.2f}")
    print(f"   ✅ RMSE: {metrics['RMSE']:.2f}")
    print(f"   ✅ R²: {metrics['R2']:.4f}")
    
except Exception as e:
    print(f"   ❌ Metrics calculation failed: {e}")
    sys.exit(1)

# Verify 4-channel architecture
print("\n6. Verifying 4-channel architecture...")
try:
    # Get first layer
    first_layer = model.layers[0]
    input_shape_check = first_layer.input_shape
    
    assert input_shape_check[-1] == 4, f"Expected 4 channels, got {input_shape_check[-1]}"
    print(f"   ✅ Confirmed 4-channel input: [ECG, PPG, PAT, HR]")
    
except Exception as e:
    print(f"   ❌ Channel verification failed: {e}")
    sys.exit(1)

# Model summary
print("\n7. Model Architecture Summary:")
print("="*60)
model.summary()

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nModel is ready for training. Run:")
print("  python src/models/model.py")
print("or:")
print("  python src/models/train_attention.py")
print("="*60 + "\n")
