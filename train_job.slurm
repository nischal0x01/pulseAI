#!/bin/bash
#SBATCH --job-name=bp_model_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

# ==============================================================================
# Slurm Job Script for Cuffless Blood Pressure Model Training
# ==============================================================================
# This script sets up the environment and runs the blood pressure prediction
# model training on an HPC cluster with GPU support.
# ==============================================================================

echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules
module purge
module load cuda/11.8
module load cudnn/8.6c
module load python/3.10

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install requirements
echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Display GPU information
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Display Python and TensorFlow versions
echo "Python version:"
python --version
echo "TensorFlow version:"
python -c "import tensorflow as tf; print(tf.__version__); print('GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
echo "=========================================="

# Run the training script
echo "Starting model training..."
python src/models/train.py

# Capture exit status
TRAIN_STATUS=$?

# Deactivate virtual environment
deactivate

echo "=========================================="
echo "Job finished on $(date)"
echo "Exit status: $TRAIN_STATUS"
echo "=========================================="

# Exit with the training script's status
exit $TRAIN_STATUS
