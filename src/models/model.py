"""
Blood Pressure Prediction Model - Main Entry Point

This script serves as the main entry point for training blood pressure prediction models.
The code has been refactored into modular components for better maintainability:

Modules:
    - config.py: Configuration and hyperparameters
    - data_loader.py: Data loading from MAT files
    - preprocessing.py: Signal preprocessing and data splitting
    - feature_engineering.py: Physiological feature extraction (PAT, HR)
    - model_builder.py: Neural network architecture definitions
    - utils.py: Utility functions for data validation
    - train.py: Main training pipeline orchestration

Usage:
    python src/models/model.py
    
    Or use the new train.py directly:
    python src/models/train.py
"""

# Simply import and run the main training pipeline
from train import main

if __name__ == "__main__":
    main()
