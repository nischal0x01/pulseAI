#!/bin/bash
# Convert Keras model to TensorFlow.js using Docker
# This avoids Python version compatibility issues

echo "Converting model using Docker..."
echo "================================"

docker run --rm -v "$(pwd)":/workspace -w /workspace \
  tensorflow/tensorflow:latest-py3 \
  bash -c "pip install -q tensorflowjs && pin install -q 'numpy==1.23.5'\
           tensorflowjs_converter \
             --input_format=keras \
             temp_model.keras \
             frontend/public/models"c

if [ $? -eq 0 ]; then
  echo ""
  echo "✓ Conversion successful!"
  echo "Model files are in: frontend/public/models/"
  echo ""
  echo "Usage in Next.js:"
  echo "  const model = await tf.loadLayersModel('/models/model.json');"
else
  echo ""
  echo "✗ Conversion failed. Make sure Docker is installed and running."
fi
