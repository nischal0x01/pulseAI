# Quick Reference: CNN-LSTM Attention Model

## 🚀 Quick Start

```bash
# Test model (fast, no training)
python src/models/test_model.py

# Train model
python src/models/model.py
```

## 📋 Key Requirements Checklist

### ✅ Input
- [x] ECG waveform (bandpass filtered 0.5-40 Hz)
- [x] PPG waveform (bandpass filtered 0.5-8 Hz)
- [x] PAT sequence (R-peak to PPG peak delay)
- [x] HR sequence (from R-R intervals)
- [x] All z-score normalized
- [x] No NaN or Inf values (asserted)

### ✅ Model Architecture
- [x] 4-channel input: [ECG, PPG, PAT, HR]
- [x] CNN layers for morphology
- [x] PAT-based attention mechanism
- [x] LSTM layers for temporal modeling
- [x] Dense layers for BP regression

### ✅ Training
- [x] Huber loss (robust to outliers)
- [x] Adam optimizer with gradient clipping
- [x] ReduceLROnPlateau callback
- [x] Patient-wise splitting (no leakage)

### ✅ Evaluation
- [x] MAE (target: < 10 mmHg)
- [x] RMSE
- [x] R² score
- [x] Pearson correlation
- [x] Predicted vs actual plot
- [x] Attention visualization

## 🎯 Target Metric

**MAE < 10 mmHg** for systolic blood pressure

## 📁 Key Files

| File | Purpose |
|------|---------|
| `model.py` | Main entry point - run this |
| `train_attention.py` | Training pipeline |
| `model_builder_attention.py` | Model architecture with attention |
| `evaluation.py` | Metrics & visualization |
| `config.py` | All hyperparameters |
| `test_model.py` | Quick architecture test |

## ⚙️ Key Hyperparameters

```python
# config.py
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GRADIENT_CLIP_NORM = 1.0

LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
ATTENTION_UNITS = 64
DROPOUT_RATE = 0.3
```

## 🔍 Model Outputs

1. **Predictions**: Systolic BP (mmHg)
2. **Attention weights**: What model focuses on
3. **Metrics**: MAE, RMSE, R², Pearson
4. **Plots**: Saved to `checkpoints/`

## 📊 Data Flow

```
MAT files → Load → Filter → Extract PAT/HR → 
→ Create 4-channel input → Validate (no NaN/Inf) →
→ Train with attention → Evaluate → Visualize
```

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| NaN during training | Increase gradient clipping to 0.5 |
| Not converging | Reduce learning rate to 5e-5 |
| MAE > 10 mmHg | Increase model capacity, add data |
| Import errors | Check you're in correct directory |

## 📈 Expected Results

```
Test Set Evaluation Metrics
  📊 MAE:     7-12 mmHg  (target: <10)
  📊 RMSE:    10-15 mmHg
  📊 R²:      0.70-0.85
  📊 Pearson: 0.80-0.90
```

## 🎨 Visualizations

After training, find in `checkpoints/`:
- `training_history.png` - Loss and MAE curves
- `pred_vs_actual_test.png` - Scatter plot
- `error_dist_test.png` - Error distribution
- `attention_test.png` - Attention weights
- `best_model.h5` - Trained model

## 💡 Pro Tips

1. **Always run test first**: `python test_model.py`
2. **Check data validation**: Look for ✅ checkmarks
3. **Monitor attention**: Ensure it's learning patterns
4. **Patient-wise split**: Prevents data leakage
5. **Save checkpoints**: Best model saved automatically

## 🔗 Related Files

- `MODEL_DETAILS.md` - Comprehensive documentation
- `IMPLEMENTATION_SUMMARY.md` - What was built
- `ARCHITECTURE.md` - System diagrams
- `README.md` - Module overview

## 🎓 Key Concepts

**PAT (Pulse Arrival Time)**
- Time from ECG R-peak to PPG systolic peak
- Inversely related to blood pressure
- Used to generate attention weights

**Attention Mechanism**
- Weights temporal importance
- Based on PAT channel
- Shows what model focuses on
- Provides interpretability

**Patient-Wise Splitting**
- No patient appears in multiple sets
- Prevents data leakage
- More realistic evaluation

## ✨ What Makes This Special

1. **PAT-Based Attention** - Novel mechanism using physiological timing
2. **Rigorous Validation** - No NaN/Inf, proper splitting
3. **Interpretable** - Attention shows model reasoning
4. **Production-Ready** - Clean, modular, documented

## 📞 Need Help?

Check these documents in order:
1. This file (quick reference)
2. `IMPLEMENTATION_SUMMARY.md` (what was built)
3. `MODEL_DETAILS.md` (comprehensive guide)
4. Code comments (inline documentation)

---

**Version 2.0.0** | *Updated: January 17, 2026*
