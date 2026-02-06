"""
Diagnose if more data would help improve model performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Fix path to import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
from data_loader import load_aggregate_data
from preprocessing import create_subject_wise_splits

print("="*70)
print("  DATA SIZE ANALYSIS - Would More Data Help?")
print("="*70)

# Load data (path relative to project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data', 'processed')
print(f"\n📁 Loading data from: {data_path}")

result = load_aggregate_data(processed_dir=data_path)
if len(result) == 5:
    signals_agg, y_sbp, y_dbp, demographics_agg, patient_ids_agg = result
else:
    signals_agg, y_sbp, y_dbp, patient_ids_agg = result

# Create splits
train_mask, val_mask, test_mask = create_subject_wise_splits(
    patient_ids_agg, test_size=0.15, val_size=0.15
)

# 1. Check BP distribution
print("\n📊 1. BP RANGE COVERAGE")
print("-" * 70)

sbp_train = y_sbp[train_mask]
print(f"Training SBP Distribution:")
print(f"   Mean: {np.mean(sbp_train):.1f} ± {np.std(sbp_train):.1f} mmHg")
print(f"   Range: {np.min(sbp_train):.1f} - {np.max(sbp_train):.1f} mmHg")

# Define BP categories
bp_ranges = [
    ("Hypotensive", 0, 90),
    ("Normal", 90, 120),
    ("Elevated", 120, 130),
    ("Stage 1 HTN", 130, 140),
    ("Stage 2 HTN", 140, 180),
    ("Crisis", 180, 300)
]

print("\n   SBP Category Distribution (Training):")
for category, low, high in bp_ranges:
    count = np.sum((sbp_train >= low) & (sbp_train < high))
    pct = 100 * count / len(sbp_train)
    print(f"   {category:15s}: {count:5d} ({pct:5.1f}%) - Range: {low}-{high} mmHg")

# Check if model is seeing enough variety
low_bp_count = np.sum(sbp_train < 100)
high_bp_count = np.sum(sbp_train > 140)
normal_bp_count = np.sum((sbp_train >= 100) & (sbp_train <= 140))

print(f"\n   ⚠️  IMBALANCE CHECK:")
print(f"   Low BP (<100):    {low_bp_count:5d} ({100*low_bp_count/len(sbp_train):.1f}%)")
print(f"   Normal BP (100-140): {normal_bp_count:5d} ({100*normal_bp_count/len(sbp_train):.1f}%)")
print(f"   High BP (>140):   {high_bp_count:5d} ({100*high_bp_count/len(sbp_train):.1f}%)")

if high_bp_count < len(sbp_train) * 0.1:
    print("\n   ❌ PROBLEM: Very few high BP samples (<10%)!")
    print("   → More high BP data would significantly help")
if low_bp_count < len(sbp_train) * 0.1:
    print("\n   ❌ PROBLEM: Very few low BP samples (<10%)!")
    print("   → More low BP data would significantly help")

# 2. Check samples per patient
print("\n📊 2. DATA PER PATIENT")
print("-" * 70)

train_patients = np.unique(patient_ids_agg[train_mask])
samples_per_patient = []
for pid in train_patients:
    count = np.sum((patient_ids_agg == pid) & train_mask)
    samples_per_patient.append(count)

samples_per_patient = np.array(samples_per_patient)
print(f"   Patients in training: {len(train_patients)}")
print(f"   Avg samples/patient: {np.mean(samples_per_patient):.1f}")
print(f"   Min samples/patient: {np.min(samples_per_patient)}")
print(f"   Max samples/patient: {np.max(samples_per_patient)}")

if np.min(samples_per_patient) < 50:
    print(f"\n   ⚠️  Some patients have <50 samples")
    print(f"   → More data per patient would help personalization")

# 3. Learning curve simulation
print("\n📊 3. LEARNING CURVE ANALYSIS")
print("-" * 70)

# Simulate what happens with different data sizes
data_fractions = [0.25, 0.5, 0.75, 1.0]
print(f"\n   Current training samples: {len(sbp_train)}")
print(f"   Current patients: {len(train_patients)}")
print(f"\n   Estimated samples needed for different scenarios:")

for frac in data_fractions:
    n_samples = int(len(sbp_train) / frac)
    n_patients = int(len(train_patients) / frac)
    if frac == 1.0:
        status = "✅ CURRENT"
    elif frac > 1.0:
        status = f"🎯 TARGET (+{int((1/frac - 1)*100)}% more)"
    else:
        status = f"   ({int(frac*100)}% of current)"
    print(f"   {status:20s}: {n_samples:6d} samples, {n_patients:3d} patients")

# 4. Test set difficulty
print("\n📊 4. TEST SET DIFFICULTY")
print("-" * 70)

sbp_test = y_sbp[test_mask]
print(f"Test SBP: {np.mean(sbp_test):.1f} ± {np.std(sbp_test):.1f} mmHg")
print(f"Train SBP: {np.mean(sbp_train):.1f} ± {np.std(sbp_train):.1f} mmHg")

# Check if test has more extreme values
test_extreme = np.sum((sbp_test < 90) | (sbp_test > 140))
train_extreme = np.sum((sbp_train < 90) | (sbp_train > 140))

print(f"\n   Extreme BP values (<90 or >140):")
print(f"   Train: {train_extreme} ({100*train_extreme/len(sbp_train):.1f}%)")
print(f"   Test:  {test_extreme} ({100*test_extreme/len(sbp_test):.1f}%)")

if test_extreme/len(sbp_test) > train_extreme/len(sbp_train) * 1.5:
    print(f"\n   ❌ PROBLEM: Test set has MORE extreme values than training!")
    print(f"   → Need more training data with extreme BP values")

# 5. Recommendations
print("\n" + "="*70)
print("  RECOMMENDATIONS")
print("="*70)

recommendations = []

# Check data size
if len(train_patients) < 50:
    recommendations.append(("HIGH", f"Add more patients (current: {len(train_patients)}, target: 50-100)"))

# Check BP coverage
if high_bp_count < len(sbp_train) * 0.15:
    recommendations.append(("HIGH", f"Add more high BP samples (current: {100*high_bp_count/len(sbp_train):.1f}%, target: >15%)"))

if low_bp_count < len(sbp_train) * 0.10:
    recommendations.append(("MEDIUM", f"Add more low BP samples (current: {100*low_bp_count/len(sbp_train):.1f}%, target: >10%)"))

# Check samples per patient
if np.mean(samples_per_patient) < 100:
    recommendations.append(("LOW", f"More samples per patient would help (current avg: {np.mean(samples_per_patient):.0f}, target: 100+)"))

if len(recommendations) == 0:
    print("\n✅ DATA SIZE IS REASONABLE!")
    print("\n   Your negative R² is likely due to:")
    print("   1. Model architecture (not data size)")
    print("   2. Feature quality (PAT/HR may not capture full BP range)")
    print("   3. Training strategy (loss function, regularization)")
    print("\n   💡 Try these instead of more data:")
    print("   • Stronger regularization (already done)")
    print("   • Different loss function (Huber loss)")
    print("   • Weighted loss (penalize extreme BP errors more)")
    print("   • Ensemble model")
else:
    print("\n📋 Priority Order:")
    for priority, rec in sorted(recommendations, key=lambda x: x[0], reverse=True):
        print(f"\n   [{priority:6s}] {rec}")
    
    print("\n💡 HOW TO GET MORE DATA:")
    print("   1. Collect more recordings from existing patients")
    print("   2. Add new patients with diverse BP ranges")
    print("   3. Focus on patients with:")
    print("      • Hypertension (SBP > 140)")
    print("      • Hypotension (SBP < 90)")
    print("      • Large BP variability")
    print("\n   4. Consider data augmentation:")
    print("      • Time warping")
    print("      • Amplitude scaling")
    print("      • Adding synthetic noise")

print("="*70)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SBP distribution comparison
ax = axes[0, 0]
bins = np.arange(60, 200, 10)
ax.hist(sbp_train, bins=bins, alpha=0.6, label='Train', edgecolor='black')
ax.hist(sbp_test, bins=bins, alpha=0.6, label='Test', edgecolor='black')
ax.axvline(np.mean(sbp_train), color='blue', linestyle='--', linewidth=2, label=f'Train Mean: {np.mean(sbp_train):.0f}')
ax.axvline(np.mean(sbp_test), color='orange', linestyle='--', linewidth=2, label=f'Test Mean: {np.mean(sbp_test):.0f}')
ax.set_xlabel('SBP (mmHg)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('SBP Distribution: Train vs Test', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Samples per patient
ax = axes[0, 1]
ax.hist(samples_per_patient, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(samples_per_patient), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(samples_per_patient):.0f}')
ax.set_xlabel('Samples per Patient', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Data Distribution Across Patients', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: BP categories
ax = axes[1, 0]
categories = [cat for cat, _, _ in bp_ranges]
counts = [np.sum((sbp_train >= low) & (sbp_train < high)) for _, low, high in bp_ranges]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(categories)))
bars = ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Training Data Coverage by BP Category', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({100*count/len(sbp_train):.1f}%)',
            ha='center', va='bottom', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Cumulative data
ax = axes[1, 1]
sorted_patients = np.argsort(samples_per_patient)[::-1]
cumsum = np.cumsum(samples_per_patient[sorted_patients])
ax.plot(range(1, len(cumsum)+1), cumsum, linewidth=2, marker='o', markersize=4)
ax.axhline(len(sbp_train), color='red', linestyle='--', label=f'Total: {len(sbp_train)} samples')
ax.axhline(len(sbp_train)*0.8, color='orange', linestyle='--', label='80% of data')
ax.set_xlabel('Number of Patients (sorted by sample count)', fontsize=12)
ax.set_ylabel('Cumulative Samples', fontsize=12)
ax.set_title('Cumulative Data Contribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
output_path = os.path.join(project_root, 'checkpoints', 'data_size_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n📊 Visualization saved to: {output_path}")
plt.show()

print("\n✅ Analysis complete!")