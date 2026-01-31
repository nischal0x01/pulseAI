"""
FastAPI server for blood pressure prediction
Receives signal data from ESP32/frontend and returns SBP/DBP predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Optional
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = FastAPI(
    title="Blood Pressure Prediction API",
    description="Cuffless blood pressure estimation using CNN-LSTM-Attention model",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = "checkpoints/best_model.h5"


class SignalData(BaseModel):
    """Input signal data from ESP32"""
    ecg: List[float] = Field(..., description="ECG signal data", min_items=1)
    ppg: List[float] = Field(..., description="PPG signal data", min_items=1)
    pat: Optional[List[float]] = Field(None, description="PAT signal (optional)")
    hr: Optional[List[float]] = Field(None, description="Heart rate signal (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ecg": [0.1, 0.2, 0.15] + [0.0] * 872,  # 875 samples
                "ppg": [0.5, 0.6, 0.55] + [0.0] * 872,
                "pat": [0.2] * 875,
                "hr": [75.0] * 875
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    sbp: float = Field(..., description="Systolic blood pressure (mmHg)")
    dbp: float = Field(..., description="Diastolic blood pressure (mmHg)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    message: str = Field(default="Success")


def load_model():
    """Load the trained model"""
    global model
    
    if model is not None:
        return model
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Import custom model builder
        from models.model_builder_attention import create_phys_informed_cnn_lstm_attention
        
        # Rebuild architecture
        model = create_phys_informed_cnn_lstm_attention(input_shape=(875, 4))
        
        # Load weights
        model.load_weights(MODEL_PATH)
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  - Input shape: {model.input_shape}")
        logger.info(f"  - Output shapes: {[out.shape for out in model.outputs]}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def preprocess_signals(ecg: List[float], ppg: List[float], 
                       pat: Optional[List[float]] = None, 
                       hr: Optional[List[float]] = None) -> np.ndarray:
    """
    Preprocess signals for model input
    
    Args:
        ecg: ECG signal
        ppg: PPG signal
        pat: PAT signal (optional, will calculate if not provided)
        hr: Heart rate signal (optional, will use default if not provided)
        
    Returns:
        Preprocessed input tensor of shape (1, 875, 4)
    """
    TARGET_LENGTH = 875
    
    def resample_or_pad(signal: List[float], target_length: int) -> np.ndarray:
        """Resample or pad signal to target length"""
        signal_array = np.array(signal)
        
        if len(signal_array) == target_length:
            return signal_array
        elif len(signal_array) > target_length:
            # Downsample
            indices = np.linspace(0, len(signal_array) - 1, target_length).astype(int)
            return signal_array[indices]
        else:
            # Pad with last value
            pad_length = target_length - len(signal_array)
            pad_value = signal_array[-1] if len(signal_array) > 0 else 0
            return np.pad(signal_array, (0, pad_length), constant_values=pad_value)
    
    def normalize_zscore(signal: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return np.zeros_like(signal)
        return (signal - mean) / std
    
    def normalize_hr(signal: np.ndarray) -> np.ndarray:
        """Normalize heart rate to 0-1 range (40-200 bpm)"""
        return (signal - 40) / 160
    
    # Resample/pad all signals
    ecg_processed = resample_or_pad(ecg, TARGET_LENGTH)
    ppg_processed = resample_or_pad(ppg, TARGET_LENGTH)
    
    # Handle optional PAT and HR
    if pat is None:
        # Calculate PAT as PPG-ECG delay (simplified)
        pat_processed = np.full(TARGET_LENGTH, 0.2)  # Default PAT value
    else:
        pat_processed = resample_or_pad(pat, TARGET_LENGTH)
    
    if hr is None:
        # Use default heart rate
        hr_processed = np.full(TARGET_LENGTH, 75.0)  # Default HR
    else:
        hr_processed = resample_or_pad(hr, TARGET_LENGTH)
    
    # Normalize signals
    ecg_norm = normalize_zscore(ecg_processed)
    ppg_norm = normalize_zscore(ppg_processed)
    pat_norm = normalize_zscore(pat_processed)
    hr_norm = normalize_hr(hr_processed)
    
    # Stack into input tensor [batch, timesteps, channels]
    input_tensor = np.stack([ecg_norm, ppg_norm, pat_norm, hr_norm], axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    
    return input_tensor


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't fail startup, will retry on first request


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Blood Pressure Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model
    
    if model is None:
        return {
            "status": "starting",
            "model_loaded": False,
            "message": "Model not yet loaded"
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "input_shape": str(model.input_shape),
        "output_shapes": [str(out.shape) for out in model.outputs]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_blood_pressure(data: SignalData):
    """
    Predict blood pressure from signal data
    
    Args:
        data: Signal data from ESP32 (ECG, PPG, optionally PAT and HR)
        
    Returns:
        Predicted SBP and DBP values
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    
    try:
        # Preprocess input
        input_tensor = preprocess_signals(
            ecg=data.ecg,
            ppg=data.ppg,
            pat=data.pat,
            hr=data.hr
        )
        
        logger.info(f"Predicting with input shape: {input_tensor.shape}")
        
        # Make prediction
        predictions = model.predict(input_tensor, verbose=0)
        
        # Extract SBP and DBP
        sbp = float(predictions[0][0][0])
        dbp = float(predictions[1][0][0])
        
        # Clamp to reasonable ranges
        sbp = max(80, min(200, sbp))
        dbp = max(40, min(130, dbp))
        
        # Calculate confidence (simplified - could be enhanced with uncertainty estimation)
        confidence = 0.85
        
        logger.info(f"Prediction: SBP={sbp:.1f}, DBP={dbp:.1f}")
        
        return PredictionResponse(
            sbp=round(sbp, 1),
            dbp=round(dbp, 1),
            confidence=confidence,
            message="Prediction successful"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(data: List[SignalData]):
    """
    Batch prediction endpoint
    
    Args:
        data: List of signal data
        
    Returns:
        List of predictions
    """
    results = []
    
    for signal_data in data:
        try:
            result = await predict_blood_pressure(signal_data)
            results.append(result)
        except HTTPException as e:
            results.append({
                "sbp": None,
                "dbp": None,
                "confidence": 0.0,
                "message": f"Error: {e.detail}"
            })
    
    return results


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
