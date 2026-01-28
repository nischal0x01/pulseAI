"""
Convert Keras H5 model to TensorFlow.js format
Workaround for Python 3.13 compatibility issues
"""

import tensorflow as tf
import subprocess
import sys

print("="*60)
print("TFJS Model Converter - Python 3.13 Compatibility Mode")
print("="*60)

# Load H5 model
print("\n[1/3] Loading model from checkpoints/best_model.h5...")
try:
    model = tf.keras.models.load_model('checkpoints/best_model.h5', safe_mode=False)
    print(f"✓ Model loaded successfully")
    print(f"  - Input shape: {model.input_shape}")
    print(f"  - Output shape: {model.output_shape}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Save in Keras native format (.keras)
keras_path = 'temp_model.keras'
print(f"\n[2/3] Saving as Keras native format to {keras_path}...")
try:
    model.save(keras_path)
    print(f"✓ Saved successfully")
except Exception as e:
    print(f"✗ Failed to save: {e}")
    sys.exit(1)

# Try to convert using tensorflowjs_converter
output_path = 'frontend/public/models'
print(f"\n[3/3] Converting to TensorFlow.js format...")
print(f"  Output: {output_path}")

cmd = [
    'tensorflowjs_converter',
    '--input_format=keras',
    keras_path,
    output_path
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Conversion successful!")
        print(f"\nYour model is ready at: {output_path}")
        print("\nUsage in Next.js:")
        print("  const model = await tf.loadLayersModel('/models/model.json');")
    else:
        print("✗ Conversion failed!")
        print("\nERROR OUTPUT:")
        print(result.stderr)
        print("\n" + "="*60)
        print("WORKAROUND OPTIONS:")
        print("="*60)
        print("\n1. Use Python 3.10 or 3.11 (recommended):")
        print("   conda create -n tfjs python=3.11")
        print("   conda activate tfjs")
        print("   pip install tensorflowjs tensorflow")
        print(f"   tensorflowjs_converter --input_format=keras {keras_path} {output_path}")
        print("\n2. Use Docker:")
        print("   docker run -v $(pwd):/workspace tensorflow/tensorflow:latest")
        print("   pip install tensorflowjs")
        print(f"   tensorflowjs_converter --input_format=keras {keras_path} {output_path}")
except FileNotFoundError:
    print("✗ tensorflowjs_converter not found!")
    print("\nInstall it with: pip install tensorflowjs")
    print("Note: May require Python 3.10 or 3.11 due to compatibility issues")