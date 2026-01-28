#!/bin/bash
# Convert Keras model to TensorFlow.js using Docker
# Fast & robust: avoids reinstalling packages every run and works regardless of where the script is run

set -e

# 1️⃣ Detect project root (where temp_model.keras lives)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root detected: $PROJECT_ROOT"

# 2️⃣ Docker image name
DOCKER_IMAGE="tfjs_converter_image"

# 3️⃣ Check if the Docker image exists; if not, build it
if [[ "$(docker images -q $DOCKER_IMAGE 2> /dev/null)" == "" ]]; then
  echo "Docker image '$DOCKER_IMAGE' not found. Building it now..."
  docker build -t $DOCKER_IMAGE -f- "$PROJECT_ROOT" <<EOF
FROM tensorflow/tensorflow:2.15.0
RUN pip install --upgrade pip
RUN pip install tensorflowjs
EOF
else
  echo "Docker image '$DOCKER_IMAGE' found. Using cached version."
fi

echo "Converting model using Docker..."
echo "================================"

# 4️⃣ Run the conversion
docker run --rm \
  -v "$PROJECT_ROOT":/workspace \
  -w /workspace \
  $DOCKER_IMAGE \
  tensorflowjs_converter --input_format=keras temp_model.keras frontend/public/models

echo ""
echo "✓ Conversion successful!"
echo "Model files are in: frontend/public/models/"
echo ""
echo "Usage in Next.js:"
echo "  const model = await tf.loadLayersModel('/models/model.json');"
