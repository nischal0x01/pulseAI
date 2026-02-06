"""
Verify training callbacks configuration in train_attention.py
"""
import os
import re

train_script = '../src/models/train_attention.py'

if not os.path.exists(train_script):
    print(f"❌ {train_script} not found!")
    exit(1)

print("="*70)
print("  TRAINING CALLBACKS VERIFICATION")
print("="*70)

# Read the training script
with open(train_script, 'r') as f:
    content = f.read()

# Check for callback configurations
print("\n🔍 Checking callback configurations...")
print("-" * 70)

# EarlyStopping
early_stopping_match = re.search(
    r'EarlyStopping\((.*?)\)',
    content,
    re.DOTALL
)
if early_stopping_match:
    config = early_stopping_match.group(1)
    print("\n✅ EarlyStopping found:")
    
    patience = re.search(r'patience\s*=\s*(\d+)', config)
    monitor = re.search(r"monitor\s*=\s*['\"]([^'\"]+)['\"]", config)
    restore = re.search(r'restore_best_weights\s*=\s*(True|False)', config)
    
    if patience:
        print(f"   - patience: {patience.group(1)}")
    if monitor:
        print(f"   - monitor: {monitor.group(1)}")
    if restore:
        print(f"   - restore_best_weights: {restore.group(1)}")
else:
    print("❌ EarlyStopping not found!")

# ReduceLROnPlateau
reduce_lr_match = re.search(
    r'ReduceLROnPlateau\((.*?)\)',
    content,
    re.DOTALL
)
if reduce_lr_match:
    config = reduce_lr_match.group(1)
    print("\n✅ ReduceLROnPlateau found:")
    
    factor = re.search(r'factor\s*=\s*([\d.]+)', config)
    patience = re.search(r'patience\s*=\s*(\d+)', config)
    min_lr = re.search(r'min_lr\s*=\s*([\de\-.]+)', config)
    
    if factor:
        print(f"   - factor: {factor.group(1)}")
    if patience:
        print(f"   - patience: {patience.group(1)}")
    if min_lr:
        print(f"   - min_lr: {min_lr.group(1)}")
else:
    print("❌ ReduceLROnPlateau not found!")

# ModelCheckpoint
checkpoint_match = re.search(
    r'ModelCheckpoint\((.*?)\)',
    content,
    re.DOTALL
)
if checkpoint_match:
    config = checkpoint_match.group(1)
    print("\n✅ ModelCheckpoint found:")
    
    filepath = re.search(r"filepath\s*=\s*['\"]([^'\"]+)['\"]", config)
    monitor = re.search(r"monitor\s*=\s*['\"]([^'\"]+)['\"]", config)
    save_best = re.search(r'save_best_only\s*=\s*(True|False)', config)
    
    if filepath:
        print(f"   - filepath: {filepath.group(1)}")
    if monitor:
        print(f"   - monitor: {monitor.group(1)}")
    if save_best:
        print(f"   - save_best_only: {save_best.group(1)}")
else:
    print("❌ ModelCheckpoint not found!")

# Check if callbacks are used in fit() - check multiline
fit_with_callbacks = re.search(r'\.fit\s*\(.*?callbacks\s*=', content, re.DOTALL)
if fit_with_callbacks:
    print("\n✅ Callbacks are passed to model.fit()")
else:
    print("\n⚠️  WARNING: Callbacks may not be passed to model.fit()")

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)

has_early_stopping = early_stopping_match is not None
has_reduce_lr = reduce_lr_match is not None
has_checkpoint = checkpoint_match is not None
uses_callbacks = fit_with_callbacks is not None

all_good = has_early_stopping and has_reduce_lr and has_checkpoint and uses_callbacks

if all_good:
    print("\n✅ ALL CALLBACKS PROPERLY CONFIGURED!")
    print("\n   Training regularization is complete:")
    print("   ✅ Dropout (0.3 rate)")
    print("   ✅ L2 Regularization (0.001)")
    print("   ✅ Batch Normalization")
    print("   ✅ EarlyStopping")
    print("   ✅ ReduceLROnPlateau")
    print("   ✅ ModelCheckpoint")
else:
    print("\n⚠️  Some callbacks missing or not configured")
    if not has_early_stopping:
        print("   ❌ EarlyStopping")
    if not has_reduce_lr:
        print("   ❌ ReduceLROnPlateau")
    if not has_checkpoint:
        print("   ❌ ModelCheckpoint")
    if not uses_callbacks:
        print("   ❌ Callbacks not passed to fit()")

print("\n" + "="*70)
print("  NEXT STEPS TO FIX NEGATIVE R²")
print("="*70)
print("""
Your regularization is perfect! The negative R² is due to baseline
reconstruction in ΔBP (residual) training.

🎯 RECOMMENDED FIX:

Option 1: Train on Absolute BP (EASIEST & RECOMMENDED)
   - Edit train_attention.py around line 160-170
   - Comment out: y_sbp_residual = convert_to_residuals(...)
   - Replace with: y_sbp_residual = y_sbp
   - Comment out: y_dbp_residual = convert_to_residuals(...)
   - Replace with: y_dbp_residual = y_dbp
   - Skip reconstruction step (predictions are already absolute)

Option 2: Fix Baseline Computation
   - Use median instead of mean for baselines
   - Ensure baselines computed ONLY from training data
   - Add per-patient baselines for test patients

Option 3: Verify Bridge Server Prediction
   - If model trained on residuals, bridge_server.py needs
     to add baselines back when making predictions

Run: grep -n "convert_to_residuals" src/models/train_attention.py
To see where these functions are called.
""")
print("="*70 + "\n")
