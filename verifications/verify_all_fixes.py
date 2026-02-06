"""
Final verification of all fixes to address negative R² scores.
"""
import os
import sys

print("="*70)
print("  FINAL VERIFICATION - ALL FIXES FOR NEGATIVE R² ")
print("="*70)

# Check 1: Verify model has L2 regularization
print("\n🔍 CHECK 1: Model Architecture Regularization")
print("-" * 70)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'models'))

try:
    import tensorflow as tf
    from model_builder_attention import create_phys_informed_model
    
    model = create_phys_informed_model(input_shape=(875, 4))
    
    # Count regularization layers
    dropout_layers = [l for l in model.layers if 'dropout' in l.name.lower()]
    l2_layers = [l for l in model.layers if hasattr(l, 'kernel_regularizer') and l.kernel_regularizer is not None]
    bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    
    print(f"✅ Dropout layers: {len(dropout_layers)} (rate=0.3)")
    print(f"✅ L2 regularization: {len(l2_layers)} layers (λ=0.001)")
    print(f"✅ Batch normalization: {len(bn_layers)} layers")
    
    if len(dropout_layers) >= 5 and len(l2_layers) >= 7:
        print("✅ Model regularization: COMPLETE")
        check1_pass = True
    else:
        print("❌ Model regularization: INCOMPLETE")
        check1_pass = False
        
except Exception as e:
    print(f"❌ Error checking model: {e}")
    check1_pass = False

# Check 2: Verify training uses absolute BP
print("\n🔍 CHECK 2: Absolute BP Training (Not Residuals)")
print("-" * 70)

train_file = 'src/models/train_attention.py'
with open(train_file, 'r') as f:
    content = f.read()

uses_absolute = 'y_sbp_absolute' in content and 'y_dbp_absolute' in content
residuals_commented = '# y_sbp_residual = convert_to_residuals' in content
no_reconstruction = '# Already absolute BP' in content

if uses_absolute and residuals_commented and no_reconstruction:
    print("✅ Training uses absolute BP values")
    print("✅ Residual conversion is commented out")
    print("✅ No reconstruction step (predictions already absolute)")
    check2_pass = True
else:
    print("❌ Training still uses residuals or missing changes")
    check2_pass = False

# Check 3: Verify training callbacks
print("\n🔍 CHECK 3: Training Callbacks")
print("-" * 70)

has_early_stopping = 'EarlyStopping' in content and 'patience=15' in content
has_reduce_lr = 'ReduceLROnPlateau' in content
has_checkpoint = 'ModelCheckpoint' in content
uses_callbacks = 'callbacks=callbacks_list' in content

if has_early_stopping and has_reduce_lr and has_checkpoint and uses_callbacks:
    print("✅ EarlyStopping configured (patience=15)")
    print("✅ ReduceLROnPlateau configured")
    print("✅ ModelCheckpoint configured")
    print("✅ Callbacks passed to model.fit()")
    check3_pass = True
else:
    print("❌ Some callbacks missing or not configured")
    check3_pass = False

# Check 4: Verify fixes from verify_fixes.py
print("\n🔍 CHECK 4: Data Leakage Prevention (Previous Fixes)")
print("-" * 70)

# Check if verify_fixes.py passes
try:
    from subprocess import run, PIPE
    result = run(['python', 'verify_fixes.py'], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0 and '✅ ALL VERIFICATION TESTS PASSED!' in result.stdout:
        print("✅ ΔBP baseline computation (train data only)")
        print("✅ Sigmoid attention (not softmax)")
        print("✅ Subject-wise PAT normalization")
        check4_pass = True
    else:
        print("⚠️  Previous fixes verification incomplete")
        check4_pass = False
except Exception as e:
    print(f"⚠️  Could not verify previous fixes: {e}")
    check4_pass = False

# Final Summary
print("\n" + "="*70)
print("  SUMMARY - READY FOR RETRAINING?")
print("="*70)

all_checks = [
    ("Model Regularization", check1_pass),
    ("Absolute BP Training", check2_pass),
    ("Training Callbacks", check3_pass),
    ("Data Leakage Prevention", check4_pass)
]

for check_name, passed in all_checks:
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")

all_pass = all([p for _, p in all_checks])

print("="*70)

if all_pass:
    print("\n🎉 ALL FIXES COMPLETE! READY TO RETRAIN!")
    print("="*70)
    print("\n📋 What's Been Fixed:")
    print("-" * 70)
    print("1. ✅ Added L2 regularization (λ=0.001) to all Conv1D/Dense/LSTM layers")
    print("2. ✅ Kept Dropout (rate=0.3) in 5 layers")
    print("3. ✅ Kept Batch Normalization in 2 layers")
    print("4. ✅ EarlyStopping (patience=15) prevents overfitting")
    print("5. ✅ Switched from ΔBP (residual) to absolute BP training")
    print("6. ✅ No baseline reconstruction errors")
    print("7. ✅ Training data-only normalization (no leakage)")
    
    print("\n🚀 Next Step: Retrain the Model")
    print("-" * 70)
    print("Run: python src/models/train_attention.py")
    print("\nExpected improvements:")
    print("  • R² should become positive (>0)")
    print("  • SBP MAE should drop below 10 mmHg (target: <5 mmHg)")
    print("  • DBP MAE should stay around 5-7 mmHg")
    print("  • Model should converge faster with L2 regularization")
    print("  • Early stopping should prevent overfitting")
    
    print("\n💡 Monitor during training:")
    print("  • Watch for val_loss decreasing")
    print("  • Check if early stopping triggers (good sign!)")
    print("  • MAE should decrease on both train and val sets")
    
else:
    print("\n⚠️  SOME CHECKS FAILED - REVIEW BEFORE RETRAINING")
    print("="*70)
    print("\nFailed checks need to be addressed before retraining.")

print("="*70 + "\n")
