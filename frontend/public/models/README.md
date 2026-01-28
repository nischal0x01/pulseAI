# Model Files

This directory will contain your TensorFlow.js model files after conversion.

## Required Files

After running the conversion script, you should have:
- `model.json` - Model architecture and metadata
- `group1-shard1of1.bin` (or multiple shards) - Model weights

## Conversion

Run the Docker conversion script from the project root:
```bash
bash convert_with_docker.sh
```

Or manually:
```bash
tensorflowjs_converter --input_format=keras temp_model.keras frontend/public/models
```

## Model Input

The model expects input shape: `[1, 875, 4]`
- 875 samples (3.5 seconds at 250 Hz)
- 4 features per sample:
  1. ECG signal (normalized)
  2. PPG signal (normalized)
  3. PAT (Pulse Arrival Time)
  4. HR (Heart Rate)

## Model Output

The model outputs: `[[SBP], [DBP]]`
- SBP: Systolic Blood Pressure (mmHg)
- DBP: Diastolic Blood Pressure (mmHg)
