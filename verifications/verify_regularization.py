"""
Verify and test regularization techniques in the BP model.
"""

import numpy as np
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

try:
    import tensorflow as tf
    from model_builder_attention import create_phys_informed_model
except ImportError as e:
    print(f"❌ Cannot import model: {e}")
    sys.exit(1)

print("="*70)
print("  REGULARIZATION VERIFICATION")
print("="*70)

# First, check the function signature
print("\n🔍 Checking function signature...")
sig = inspect.signature(create_phys_informed_model)
print(f"Function signature: {sig}")
print("\nParameters:")
for param_name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else "no default"
    print(f"   - {param_name}: {default}")

# Create a test model with correct parameters
print("\n🔬 Creating test model...")
try:
    # Try to create model with just input_shape
    model = create_phys_informed_model(input_shape=(875, 4))
    print("✅ Model created successfully with input_shape only")
except Exception as e:
    print(f"❌ Error creating model: {e}")
    sys.exit(1)

print("\n📊 Analyzing Model Architecture:")
print("-" * 70)

# Check for Dropout layers
dropout_layers = [layer for layer in model.layers if 'dropout' in layer.name.lower()]
print(f"\n🔍 Dropout Layers: {len(dropout_layers)}")
if len(dropout_layers) > 0:
    for layer in dropout_layers:
        if hasattr(layer, 'rate'):
            print(f"   ✓ {layer.name}: rate={layer.rate}")
        else:
            print(f"   ✓ {layer.name}")
    print("   ✅ Dropout is used")
else:
    print("   ❌ WARNING: No dropout layers found!")
    print("   → Need to add Dropout layers to model architecture")

# Check for L2 Regularization
print(f"\n🔍 L2 Regularization:")
has_l2 = False
l2_layers = []
for layer in model.layers:
    if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
        has_l2 = True
        l2_layers.append(layer.name)

if has_l2:
    print(f"   ✓ Found L2 in {len(l2_layers)} layers:")
    for name in l2_layers[:5]:  # Show first 5
        print(f"      - {name}")
    if len(l2_layers) > 5:
        print(f"      ... and {len(l2_layers) - 5} more")
    print("   ✅ L2 regularization is used")
else:
    print("   ❌ WARNING: No L2 regularization found!")
    print("   → Need to add kernel_regularizer=l2(0.001) to layers")

# Check for Batch Normalization
bn_layers = [layer for layer in model.layers if 'batch' in layer.name.lower() or isinstance(layer, tf.keras.layers.BatchNormalization)]
print(f"\n🔍 Batch Normalization: {len(bn_layers)} layers")
if len(bn_layers) > 0:
    for layer in bn_layers[:5]:  # Show first 5
        print(f"   ✓ {layer.name}")
    if len(bn_layers) > 5:
        print(f"      ... and {len(bn_layers) - 5} more")
    print("   ✅ Batch normalization is used")
else:
    print("   ⚠️  No batch normalization found (optional but recommended)")

# Check layer types
print(f"\n🔍 Model Architecture Overview:")
conv_layers = [l for l in model.layers if 'conv' in l.name.lower()]
dense_layers = [l for l in model.layers if 'dense' in l.name.lower()]
attention_layers = [l for l in model.layers if 'attention' in l.name.lower()]

print(f"   - Conv1D layers: {len(conv_layers)}")
print(f"   - Dense layers: {len(dense_layers)}")
print(f"   - Attention layers: {len(attention_layers)}")

# Model summary statistics
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

print("\n" + "="*70)
print("  MODEL SUMMARY")
print("="*70)
print(f"  Total Parameters:     {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Conv1D Layers:        {len(conv_layers)}")
print(f"  Dense Layers:         {len(dense_layers)}")
print(f"  Dropout Layers:       {len(dropout_layers)} {'✅' if len(dropout_layers) > 0 else '❌'}")
print(f"  L2 Regularization:    {'✅ Present' if has_l2 else '❌ Missing'}")
print(f"  Batch Normalization:  {len(bn_layers)} layers")
print("="*70)

# Detailed analysis
print("\n💡 DETAILED ANALYSIS:")
print("-" * 70)

issues_found = False

if len(dropout_layers) == 0:
    print("\n❌ CRITICAL: No Dropout layers found")
    print("   Without dropout, your model will likely overfit!")
    print("   ")
    print("   Fix: Add Dropout layers after Conv1D and Dense layers")
    print("   Example:")
    print("     x = layers.Conv1D(...)(x)")
    print("     x = layers.Dropout(0.3)(x)  # Add this")
    issues_found = True
else:
    print(f"\n✅ Dropout is present ({len(dropout_layers)} layers)")
    avg_rate = np.mean([l.rate for l in dropout_layers if hasattr(l, 'rate')])
    print(f"   Average dropout rate: {avg_rate:.2f}")
    if avg_rate < 0.2:
        print(f"   ⚠️  Dropout rate is low. Consider 0.3-0.5 for better regularization")
    
if not has_l2:
    print("\n❌ CRITICAL: No L2 regularization found")
    print("   L2 regularization helps prevent overfitting by penalizing large weights")
    print("   ")
    print("   Fix: Add kernel_regularizer to Conv1D and Dense layers")
    print("   Example:")
    print("     from tensorflow.keras.regularizers import l2")
    print("     x = layers.Conv1D(..., kernel_regularizer=l2(0.001))(x)")
    issues_found = True
else:
    print(f"\n✅ L2 regularization is present ({len(l2_layers)} layers)")

if len(bn_layers) == 0:
    print("\n⚠️  No Batch Normalization found")
    print("   BatchNorm helps with training stability and can act as regularization")
    print("   Consider adding after Conv1D layers")
else:
    print(f"\n✅ Batch Normalization is present ({len(bn_layers)} layers)")

print("\n📋 TRAINING CHECKLIST:")
print("-" * 70)
print(f"  {'✅' if len(dropout_layers) > 0 else '❌'} Dropout in model architecture")
print(f"  {'✅' if has_l2 else '❌'} L2 regularization in Conv1D/Dense layers")
print(f"  {'✅' if len(bn_layers) > 0 else '⚠️ '} Batch Normalization (optional)")
print("  ☐ EarlyStopping callback (patience=15-20)")
print("  ☐ ReduceLROnPlateau callback")
print("  ☐ ModelCheckpoint to save best weights")
print("=" * 70)

if issues_found:
    print("\n⚠️  REGULARIZATION ISSUES DETECTED!")
    print("   Your negative R² scores may be due to:")
    print("   1. Missing regularization causing overfitting")
    print("   2. Baseline reconstruction issues (check previous fix)")
    print("   3. Both problems combined")
    print("\n   Next steps:")
    print("   1. Update model_builder_attention.py to add missing regularization")
    print("   2. Verify training script has callbacks (EarlyStopping, etc.)")
    print("   3. Consider training on absolute BP instead of residuals")
else:
    print("\n✅ MODEL REGULARIZATION LOOKS GOOD!")
    print("   Your negative R² is likely due to baseline reconstruction")
    print("   Consider switching to absolute BP training (see previous recommendation)")

print("="*70 + "\n")

# Show first few layers for debugging
print("📝 First 10 layers of model:")
print("-" * 70)
for i, layer in enumerate(model.layers[:10]):
    config = layer.get_config() if hasattr(layer, 'get_config') else {}
    print(f"{i+1}. {layer.__class__.__name__:20s} | {layer.name:25s}")
    if 'dropout' in layer.name.lower() and hasattr(layer, 'rate'):
        print(f"   └─ rate: {layer.rate}")
    if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
        print(f"   └─ L2 regularizer present")

print("="*70 + "\n")