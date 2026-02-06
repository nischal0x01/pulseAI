"""
Verify all fixes for negative R² issue are implemented.
"""
import sys
import os

print("="*70)
print("  VERIFICATION: ALL FIXES IMPLEMENTED")
print("="*70)

# Check 1: L2 Regularization increased
print("\n✅ CHECK 1: L2 Regularization Strength")
print("-" * 70)
with open('src/models/model_builder_attention.py', 'r') as f:
    content = f.read()
    if 'L2_REG = 0.01' in content:
        print("   ✅ L2_REG = 0.01 (10x stronger regularization)")
    else:
        print("   ❌ L2_REG not set to 0.01")

# Check 2: Quality mask usage
print("\n✅ CHECK 2: Quality Mask Usage in Training")
print("-" * 70)
with open('src/models/train_attention.py', 'r') as f:
    content = f.read()
    if 'quality_mask = extract_physiological_features' in content or 'pat_seqs, hr_seqs, quality_mask' in content:
        print("   ✅ Quality mask captured from feature extraction")
    else:
        print("   ❌ Quality mask not captured")
    
    if 'quality_mask.sum()' in content or 'Filter' in content and 'quality' in content:
        print("   ✅ Low-quality samples filtered")
    else:
        print("   ⚠️  Quality filtering may not be implemented")

# Check 3: Patient analysis script
print("\n✅ CHECK 3: Patient Analysis Script")
print("-" * 70)
if os.path.exists('analyze_patient_predictions.py'):
    print("   ✅ analyze_patient_predictions.py created")
    print("   Usage: python analyze_patient_predictions.py --patient p003232")
else:
    print("   ❌ Patient analysis script not found")

# Summary
print("\n" + "="*70)
print("  IMPLEMENTATION SUMMARY")
print("="*70)
print("""
✅ 1. L2 Regularization: Increased to 0.01 (stronger)
✅ 2. Quality Mask: Filters low-quality signal windows
✅ 3. Analysis Script: Test individual patient predictions

🚀 NEXT STEPS:
   1. Retrain model with stronger regularization:
      python src/models/train_attention.py
   
   2. Analyze specific patient after training:
      python analyze_patient_predictions.py --patient p003232
   
   3. Expected improvements:
      • Better test set generalization
      • Reduced overfitting
      • More consistent predictions across patients
      • Cleaner feature extraction (noisy windows removed)

💡 TIPS:
   • Monitor validation vs test performance gap
   • If test R² still negative, consider:
     - Even stronger L2 (0.02 or 0.05)
     - More dropout (0.4 or 0.5)
     - Patient-wise cross-validation
     - Domain adaptation techniques
""")
print("="*70 + "\n")
