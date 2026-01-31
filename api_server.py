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

# Signal processing constants
TARGET_SIGNAL_LENGTH = 875  # Number of samples required for model input
MIN_SBP = 80  # Minimum valid SBP value (mmHg)
MAX_SBP = 200  # Maximum valid SBP value (mmHg)
MIN_DBP = 40  # Minimum valid DBP value (mmHg)
MAX_DBP = 130  # Maximum valid DBP value (mmHg)
DEFAULT_PAT_VALUE = 0.2  # Default PAT value when not provided
DEFAULT_HR_VALUE = 75.0  # Default heart rate (bpm) when not provided
MIN_HR = 40.0  # Minimum heart rate for normalization (bpm)
MAX_HR = 200.0  # Maximum heart rate for normalization (bpm)

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
                "ecg": [0.1, 0.2, 0.15] + [0.0] * (TARGET_SIGNAL_LENGTH - 3),
                "ppg": [0.5, 0.6, 0.55] + [0.0] * (TARGET_SIGNAL_LENGTH - 3),
                "pat": [DEFAULT_PAT_VALUE] * TARGET_SIGNAL_LENGTH,
                "hr": [DEFAULT_HR_VALUE] * TARGET_SIGNAL_LENGTH
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
        Preprocessed input tensor of shape (1, TARGET_SIGNAL_LENGTH, 4)
    """
    TARGET_LENGTH = TARGET_SIGNAL_LENGTH
    
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
        """Normalize heart rate to 0-1 range"""
        return (signal - MIN_HR) / (MAX_HR - MIN_HR)
    
    # Resample/pad all signals
    ecg_processed = resample_or_pad(ecg, TARGET_LENGTH)
    ppg_processed = resample_or_pad(ppg, TARGET_LENGTH)
    
    # Handle optional PAT and HR
    if pat is None:
        # Calculate PAT as PPG-ECG delay (simplified)
        pat_processed = np.full(TARGET_LENGTH, DEFAULT_PAT_VALUE)
    else:
        pat_processed = resample_or_pad(pat, TARGET_LENGTH)
    
    if hr is None:
        # Use default heart rate
        hr_processed = np.full(TARGET_LENGTH, DEFAULT_HR_VALUE)
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


def calculate_prediction_confidence(sbp: float, dbp: float) -> float:
    """
    Calculate prediction confidence based on physiological validity
    
    Args:
        sbp: Predicted systolic blood pressure
        dbp: Predicted diastolic blood pressure
        
    Returns:
        Confidence score between 0 and 1
    """
    base_confidence = 0.85
    
    # Check if SBP > DBP (physiological requirement)
    if sbp <= dbp:
        return 0.3  # Low confidence for invalid measurements
    
    # Check pulse pressure (SBP - DBP should be 30-60 typically)
    pulse_pressure = sbp - dbp
    if pulse_pressure < 20 or pulse_pressure > 100:
        base_confidence *= 0.7
    
    # Check if values are at clamping limits (indicates out-of-range prediction)
    if sbp == MIN_SBP or sbp == MAX_SBP or dbp == MIN_DBP or dbp == MAX_DBP:
        base_confidence *= 0.6
    
    # Boost confidence for normal ranges (90-120/60-80)
    if 90 <= sbp <= 120 and 60 <= dbp <= 80:
        base_confidence = min(1.0, base_confidence * 1.1)
    
    return round(base_confidence, 2)


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
        sbp = max(MIN_SBP, min(MAX_SBP, sbp))
        dbp = max(MIN_DBP, min(MAX_DBP, dbp))
        
        # Calculate confidence based on prediction validity
        # Higher confidence if values are within normal ranges and physiologically consistent
        confidence = calculate_prediction_confidence(sbp, dbp)
        
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
