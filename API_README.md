# Blood Pressure Prediction API

Python backend API for cuffless blood pressure estimation using CNN-LSTM-Attention model.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements/requirements.txt
```

### 2. Start the API Server

```bash
python api_server.py
```

The API will start on `http://localhost:8000`

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ecg": [0.1, 0.2, ...],  # 875 samples
    "ppg": [0.5, 0.6, ...]   # 875 samples
  }'
```

## API Endpoints

### GET `/`
Health check - returns API status

### GET `/health`
Detailed health check - returns model status and configuration

### POST `/predict`
Predict blood pressure from signal data

**Request Body:**
```json
{
  "ecg": [float array],     // Required: ECG signal
  "ppg": [float array],     // Required: PPG signal
  "pat": [float array],     // Optional: PAT signal
  "hr": [float array]       // Optional: Heart rate signal
}
```

**Response:**
```json
{
  "sbp": 120.5,             // Systolic BP in mmHg
  "dbp": 80.2,              // Diastolic BP in mmHg
  "confidence": 0.85,       // Prediction confidence
  "message": "Success"
}
```

### POST `/predict/batch`
Batch prediction for multiple signal sets

## Architecture

- **FastAPI**: Modern web framework for building APIs
- **TensorFlow/Keras**: Deep learning model inference
- **CORS**: Enabled for Next.js frontend communication
- **Preprocessing**: Automatic signal resampling and normalization

## Integration with Frontend

The Next.js frontend automatically connects to this API. Make sure:

1. API server is running on port 8000
2. Frontend has `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local`
3. Both services can communicate (check CORS settings)

## Model

The API loads the trained model from `checkpoints/best_model.h5`. This model expects:

- **Input**: 4 channels (ECG, PPG, PAT, HR) with 875 timesteps
- **Output**: 2 values (SBP, DBP) in mmHg

## Development

To run in development mode with auto-reload:

```bash
uvicorn api_server:app --reload --port 8000
```

## Production Deployment

For production, use:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use with Gunicorn:

```bash
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```
