# TensorFlow.js Integration Guide

## Overview

This guide explains how the blood pressure prediction model is integrated into the Next.js frontend using TensorFlow.js.

## Architecture

### Components

1. **Model Service** (`lib/model-service.ts`)
   - Singleton pattern for model management
   - Handles model loading, prediction, and cleanup
   - Automatic signal preprocessing and normalization

2. **BP Predictor** (`components/bp-predictor.tsx`)
   - Main UI component for displaying predictions
   - Auto-loads model on mount
   - Updates predictions every 2 seconds
   - Shows BP category, confidence, and status

3. **Model Debugger** (`components/model-debugger.tsx`)
   - Development tool for testing model loading
   - Displays model info and diagnostics
   - Remove before production

### Signal Flow

```
WebSocket → SignalBuffer → BP Predictor → Model Service → TensorFlow.js Model
                ↓                                              ↓
         875 samples (3.5s)                            [SBP, DBP] prediction
```

## Model Requirements

### Input Format
- **Shape**: `[1, 875, 4]`
- **Features**:
  1. ECG signal (normalized with z-score)
  2. PPG signal (normalized with z-score)
  3. PAT - Pulse Arrival Time (normalized)
  4. HR - Heart Rate (normalized to 0-1 range)

### Output Format
- **Dual outputs**: `[[SBP], [DBP]]`
- **Range**: SBP: 80-200 mmHg, DBP: 40-130 mmHg

## Setup Instructions

### 1. Convert Model

First, ensure your Keras model is converted to TensorFlow.js format:

```bash
# Option A: Using Docker (recommended for Python 3.13)
bash convert_with_docker.sh

# Option B: Using Python 3.11 environment
conda create -n tfjs python=3.11
conda activate tfjs
pip install tensorflowjs tensorflow
tensorflowjs_converter --input_format=keras temp_model.keras frontend/public/models
```

### 2. Verify Model Files

Check that these files exist:
```
frontend/public/models/
├── model.json
└── group1-shard1of1.bin (or multiple shards)
```

### 3. Test Model Loading

Add the `ModelDebugger` component temporarily to test:

```tsx
import { ModelDebugger } from "@/components/model-debugger"

// In your page
<ModelDebugger />
```

Click "Test Model Load" to verify the model loads correctly.

### 4. Use BP Predictor

The `BPPredictor` component is already integrated into the main page. It will:
- Auto-load the model on mount
- Wait for sufficient signal data (875 samples)
- Make predictions every 2 seconds
- Display results with BP category

## API Reference

### Model Service

```typescript
import { getModelInstance } from "@/lib/model-service"

const model = getModelInstance()

// Load model
await model.loadModel()

// Check if ready
if (model.isModelReady()) {
  // Make prediction
  const prediction = await model.predict(
    ecgSignal,  // number[] - 875 ECG samples
    ppgSignal,  // number[] - 875 PPG samples
    patSignal,  // number[] - optional PAT values
    hrSignal    // number[] - optional HR values
  )
  
  console.log(prediction)
  // { sbp: 120, dbp: 80, confidence: 0.85 }
}

// Clean up
model.dispose()
```

### BP Predictor Component

```tsx
import { BPPredictor } from "@/components/bp-predictor"

export default function Page() {
  return (
    <SignalProvider>
      <BPPredictor />
    </SignalProvider>
  )
}
```

## Data Preprocessing

The model service automatically handles:

1. **Signal Resampling**: Adjusts signals to exactly 875 samples
2. **Normalization**:
   - ECG/PPG: Z-score normalization
   - HR: Min-max scaling (40-200 bpm → 0-1)
   - PAT: Z-score normalization
3. **Missing Data**: Fills with reasonable defaults if PAT/HR unavailable

## Blood Pressure Categories

| Category | SBP (mmHg) | DBP (mmHg) | Color |
|----------|------------|------------|-------|
| Low | <90 | <60 | Blue |
| Normal | 90-129 | 60-79 | Green |
| Elevated | 130-139 | 80-89 | Yellow |
| High | 140-179 | 90-119 | Orange |
| Crisis | ≥180 | ≥120 | Red |

## Performance Optimization

### Memory Management
- Model is loaded once and reused (singleton pattern)
- Tensors are disposed after each prediction
- Use `model.dispose()` to free memory when done

### Prediction Frequency
- Default: Every 2 seconds
- Adjust in `bp-predictor.tsx` (line ~45)
- Balance real-time updates vs. performance

### Lazy Loading
Model loads on component mount, not on page load. This prevents blocking initial page render.

## Troubleshooting

### Model Won't Load
- **Check file path**: Model should be at `/public/models/model.json`
- **Check conversion**: Ensure `tensorflowjs_converter` completed successfully
- **Check browser console**: Look for detailed error messages

### Predictions Are Inaccurate
- **Verify signal quality**: Check that signals are clean and properly sampled
- **Check input shape**: Must be exactly 875 samples
- **Check normalization**: Ensure signals are in expected range

### Performance Issues
- **Reduce prediction frequency**: Change interval from 2s to 5s
- **Use WebAssembly backend**: `tf.setBackend('wasm')`
- **Check memory leaks**: Monitor with Chrome DevTools

## Production Checklist

- [ ] Model converted and tested
- [ ] Remove `ModelDebugger` component
- [ ] Test on target devices/browsers
- [ ] Add error tracking (Sentry, etc.)
- [ ] Implement medical disclaimer UI
- [ ] Add confidence thresholds for alerts
- [ ] Test with various signal qualities
- [ ] Performance profiling complete

## Medical Disclaimer

**Important**: This implementation is for research/educational purposes. The predictions should NOT be used for medical diagnosis or treatment without proper clinical validation and regulatory approval.

Always include appropriate disclaimers in your UI.
