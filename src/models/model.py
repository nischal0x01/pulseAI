"""
Blood Pressure Prediction Model - Main Entry Point

This script implements a physiology-informed CNN-LSTM model with PAT-based attention
for cuffless blood pressure estimation.

Architecture:
- 4-channel input: [ECG, PPG, PAT, HR]
- CNN layers extract morphological features
- PAT-based attention mechanism weights temporal importance
- LSTM layers model cardiovascular dynamics
- Dense layers regress systolic blood pressure

Features:
- Rigorous data validation (no NaN/Inf)
- Patient-wise data splitting (no leakage)
- Huber loss with gradient clipping
- Comprehensive evaluation (MAE, RMSE, R², Pearson correlation)
- Attention visualization

Target: MAE < 10 mmHg

Usage:
    python src/models/model.py
"""

# Import and run the attention-based training pipeline
from train_attention import main

if __name__ == "__main__":
    main()
