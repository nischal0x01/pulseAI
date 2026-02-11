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
import glob
import pickle
from pathlib import Path

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


@app.get("/patients")
async def get_patients():
    """Get list of available patient IDs from processed data directory."""
    try:
        import glob
        from pathlib import Path
        
        data_dir = Path("data/processed")
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail="Data directory not found")
        
        # Find all patient .mat files
        patient_files = sorted(glob.glob(str(data_dir / "p*.mat")))
        patients = [Path(f).stem for f in patient_files]
        
        if not patients:
            return {"patients": [], "message": "No patient data found"}
        
        logger.info(f"Found {len(patients)} patients")
        return {
            "patients": patients,
            "count": len(patients)
        }
        
    except Exception as e:
        logger.error(f"Error fetching patients: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-patient")
async def analyze_patient(request: dict):
    """
    Analyze predictions for a specific patient.
    
    Args:
        request: {
            "patient_id": str,
            "n_samples": int (optional, default 10),
            "apply_calibration": bool (optional, default False)
        }
    
    Returns:
        Detailed analysis including metrics and sample predictions
    """
    try:
        patient_id = request.get("patient_id")
        n_display_samples = request.get("n_samples", 10)
        apply_calibration = request.get("apply_calibration", False)
        
        if not patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required")
        
        logger.info(f"Analyzing patient: {patient_id} (apply_calibration={apply_calibration})")
        
        # Import necessary functions
        from models.preprocessing import preprocess_signals
        from models.feature_engineering import (
            extract_physiological_features, 
            standardize_feature_length,
            normalize_pat_subject_wise,
            create_4_channel_input
        )
        from models.data_loader import _read_subj_wins
        import pickle
        
        # Load patient data
        patient_file = f"data/processed/{patient_id}.mat"
        if not os.path.exists(patient_file):
            raise HTTPException(status_code=404, detail=f"Patient file not found: {patient_id}")
        
        signals_agg, y_sbp, y_dbp, demographics = _read_subj_wins(patient_file)
        n_samples = len(y_sbp)
        
        logger.info(f"Loaded {n_samples} samples for patient {patient_id}")
        
        # Preprocess signals
        processed_signals = preprocess_signals(signals_agg)
        
        # Extract features
        ppg_raw = signals_agg[:, 0, :]
        ecg_raw = signals_agg[:, 1, :]
        pat_seqs, hr_seqs, quality_mask = extract_physiological_features(ecg_raw, ppg_raw)
        
        # Standardize feature lengths
        target_len = processed_signals.shape[1]
        pat_seqs = standardize_feature_length(pat_seqs, target_len)
        hr_seqs = standardize_feature_length(hr_seqs, target_len)
        
        # Normalize features
        patient_ids = np.array([patient_id] * n_samples)
        train_mask = np.ones(n_samples, dtype=bool)
        pat_seqs_scaled, _ = normalize_pat_subject_wise(pat_seqs, patient_ids, train_mask)
        
        hr_mean = np.mean(hr_seqs)
        hr_std = np.std(hr_seqs)
        hr_seqs_scaled = (hr_seqs - hr_mean) / (hr_std if hr_std > 0 else 1.0)
        
        # Create 4-channel input
        X_phys_informed = create_4_channel_input(processed_signals, pat_seqs_scaled, hr_seqs_scaled)
        
        # Make predictions
        global model
        if model is None:
            load_model()
        
        predictions = model.predict(X_phys_informed, verbose=0)
        y_pred_sbp = predictions[0].flatten()
        y_pred_dbp = predictions[1].flatten()
        
        # Check for calibration existence
        calibration_path = f"calibration/calibration_{patient_id}.pkl"
        calibration_exists = os.path.exists(calibration_path)
        calibrated = False
        
        # Only apply calibration if explicitly requested
        if apply_calibration and calibration_exists:
            with open(calibration_path, 'rb') as f:
                cal_params = pickle.load(f)
            
            # Apply calibration
            if cal_params['method'] == 'linear':
                y_pred_sbp = cal_params['sbp_slope'] * y_pred_sbp + cal_params['sbp_intercept']
                y_pred_dbp = cal_params['dbp_slope'] * y_pred_dbp + cal_params['dbp_intercept']
            else:  # offset
                y_pred_sbp += cal_params['sbp_intercept']
                y_pred_dbp += cal_params['dbp_intercept']
            
            calibrated = True
            logger.info(f"Applied calibration for patient {patient_id}")
        elif calibration_exists:
            logger.info(f"Calibration exists but not applied (apply_calibration={apply_calibration})")
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred):
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - (ss_res / ss_tot) if ss_tot != 0 else 0)
            pearson = float(np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0)
            mean_error = float(np.mean(y_pred - y_true))
            std_error = float(np.std(y_pred - y_true))
            
            return {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 3),
                "pearson": round(pearson, 3),
                "mean_error": round(mean_error, 2),
                "std_error": round(std_error, 2)
            }
        
        sbp_metrics = calculate_metrics(y_sbp, y_pred_sbp)
        dbp_metrics = calculate_metrics(y_dbp, y_pred_dbp)
        
        # Sample predictions for display
        display_indices = np.linspace(0, n_samples-1, min(n_display_samples, n_samples), dtype=int)
        sample_predictions = []
        for idx in display_indices:
            sample_predictions.append({
                "index": int(idx),
                "actual_sbp": round(float(y_sbp[idx]), 1),
                "pred_sbp": round(float(y_pred_sbp[idx]), 1),
                "sbp_error": round(float(y_pred_sbp[idx] - y_sbp[idx]), 1),
                "actual_dbp": round(float(y_dbp[idx]), 1),
                "pred_dbp": round(float(y_pred_dbp[idx]), 1),
                "dbp_error": round(float(y_pred_dbp[idx] - y_dbp[idx]), 1)
            })
        
        return {
            "patient_id": patient_id,
            "n_samples": n_samples,
            "sbp_metrics": sbp_metrics,
            "dbp_metrics": dbp_metrics,
            "sample_predictions": sample_predictions,
            "calibrated": calibrated,
            "summary_stats": {
                "actual_sbp_mean": round(float(np.mean(y_sbp)), 1),
                "actual_sbp_std": round(float(np.std(y_sbp)), 1),
                "pred_sbp_mean": round(float(np.mean(y_pred_sbp)), 1),
                "pred_sbp_std": round(float(np.std(y_pred_sbp)), 1),
                "actual_dbp_mean": round(float(np.mean(y_dbp)), 1),
                "actual_dbp_std": round(float(np.std(y_dbp)), 1),
                "pred_dbp_mean": round(float(np.mean(y_pred_dbp)), 1),
                "pred_dbp_std": round(float(np.std(y_pred_dbp)), 1)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing patient: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calibrate-patient")
async def calibrate_patient(request: dict):
    """
    Calibrate model for a specific patient using stratified sampling.
    
    Args:
        request: {
            "patient_id": str,
            "n_calibration_samples": int (default 50),
            "method": str ("linear" or "offset", default "offset"),
            "stratify": str ("sbp", "dbp", or "both", default "sbp")
        }
    
    Returns:
        Calibration parameters and metrics
    """
    try:
        patient_id = request.get("patient_id")
        n_calibration_samples = request.get("n_calibration_samples", 50)
        method = request.get("method", "offset")
        stratify = request.get("stratify", "sbp")
        
        if not patient_id:
            raise HTTPException(status_code=400, detail="patient_id is required")
        
        if method not in ["linear", "offset"]:
            raise HTTPException(status_code=400, detail="method must be 'linear' or 'offset'")
        
        if stratify not in ["sbp", "dbp", "both"]:
            raise HTTPException(status_code=400, detail="stratify must be 'sbp', 'dbp', or 'both'")
        
        logger.info(f"Calibrating patient {patient_id} with method={method}, stratify={stratify}")
        
        # Import necessary functions
        from models.preprocessing import preprocess_signals
        from models.feature_engineering import (
            extract_physiological_features,
            standardize_feature_length,
            normalize_pat_subject_wise,
            create_4_channel_input
        )
        from models.data_loader import _read_subj_wins
        import pickle
        
        # Load patient data
        patient_file = f"data/processed/{patient_id}.mat"
        if not os.path.exists(patient_file):
            raise HTTPException(status_code=404, detail=f"Patient file not found: {patient_id}")
        
        signals_agg, y_sbp, y_dbp, demographics = _read_subj_wins(patient_file)
        n_total = len(y_sbp)
        
        if n_total < n_calibration_samples:
            n_calibration_samples = n_total
            logger.warning(f"Only {n_total} samples available, using all for calibration")
        
        # Preprocess all data
        processed_signals = preprocess_signals(signals_agg)
        
        # Extract features
        ppg_raw = signals_agg[:, 0, :]
        ecg_raw = signals_agg[:, 1, :]
        pat_seqs, hr_seqs, quality_mask = extract_physiological_features(ecg_raw, ppg_raw)
        
        target_len = processed_signals.shape[1]
        pat_seqs = standardize_feature_length(pat_seqs, target_len)
        hr_seqs = standardize_feature_length(hr_seqs, target_len)
        
        # Normalize features
        patient_ids = np.array([patient_id] * n_total)
        train_mask = np.ones(n_total, dtype=bool)
        pat_seqs_scaled, _ = normalize_pat_subject_wise(pat_seqs, patient_ids, train_mask)
        
        hr_mean = np.mean(hr_seqs)
        hr_std = np.std(hr_seqs)
        hr_seqs_scaled = (hr_seqs - hr_mean) / (hr_std if hr_std > 0 else 1.0)
        
        X_phys = create_4_channel_input(processed_signals, pat_seqs_scaled, hr_seqs_scaled)
        
        # Stratified sampling
        def stratified_sample(y_values, n_samples, n_bins=5):
            bins = np.linspace(y_values.min(), y_values.max(), n_bins + 1)
            bin_indices = np.digitize(y_values, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            samples_per_bin = n_samples // n_bins
            extra_samples = n_samples % n_bins
            
            selected_indices = []
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                bin_sample_indices = np.where(bin_mask)[0]
                
                if len(bin_sample_indices) == 0:
                    continue
                
                n_bin_samples = samples_per_bin + (1 if bin_idx < extra_samples else 0)
                n_bin_samples = min(n_bin_samples, len(bin_sample_indices))
                
                selected = np.random.choice(bin_sample_indices, n_bin_samples, replace=False)
                selected_indices.extend(selected)
            
            return np.array(sorted(selected_indices))
        
        # Select calibration indices based on stratification strategy
        if stratify == "sbp":
            cal_indices = stratified_sample(y_sbp, n_calibration_samples)
        elif stratify == "dbp":
            cal_indices = stratified_sample(y_dbp, n_calibration_samples)
        else:  # both
            # Stratify on combined SBP+DBP
            combined_bp = (y_sbp + y_dbp) / 2
            cal_indices = stratified_sample(combined_bp, n_calibration_samples)
        
        test_indices = np.array([i for i in range(n_total) if i not in cal_indices])
        
        # Make predictions
        global model
        if model is None:
            load_model()
        
        predictions = model.predict(X_phys, verbose=0)
        y_pred_sbp = predictions[0].flatten()
        y_pred_dbp = predictions[1].flatten()
        
        # Calculate calibration parameters
        if method == "linear":
            # Linear regression: y_true = slope * y_pred + intercept
            from sklearn.linear_model import LinearRegression
            
            sbp_model = LinearRegression()
            sbp_model.fit(y_pred_sbp[cal_indices].reshape(-1, 1), y_sbp[cal_indices])
            sbp_slope = float(sbp_model.coef_[0])
            sbp_intercept = float(sbp_model.intercept_)
            
            dbp_model = LinearRegression()
            dbp_model.fit(y_pred_dbp[cal_indices].reshape(-1, 1), y_dbp[cal_indices])
            dbp_slope = float(dbp_model.coef_[0])
            dbp_intercept = float(dbp_model.intercept_)
            
            # Apply calibration
            y_pred_sbp_cal = sbp_slope * y_pred_sbp + sbp_intercept
            y_pred_dbp_cal = dbp_slope * y_pred_dbp + dbp_intercept
            
        else:  # offset
            sbp_slope = 1.0
            dbp_slope = 1.0
            sbp_intercept = float(np.mean(y_sbp[cal_indices] - y_pred_sbp[cal_indices]))
            dbp_intercept = float(np.mean(y_dbp[cal_indices] - y_pred_dbp[cal_indices]))
            
            # Apply calibration
            y_pred_sbp_cal = y_pred_sbp + sbp_intercept
            y_pred_dbp_cal = y_pred_dbp + dbp_intercept
        
        # Save calibration parameters
        cal_params = {
            "patient_id": patient_id,
            "method": method,
            "stratify": stratify,
            "n_calibration_samples": len(cal_indices),
            "sbp_slope": sbp_slope,
            "sbp_intercept": sbp_intercept,
            "dbp_slope": dbp_slope,
            "dbp_intercept": dbp_intercept,
            "calibration_indices": cal_indices.tolist(),
            "timestamp": str(np.datetime64('now'))
        }
        
        # Ensure calibration directory exists
        os.makedirs("calibration", exist_ok=True)
        
        cal_path = f"calibration/calibration_{patient_id}.pkl"
        with open(cal_path, 'wb') as f:
            pickle.dump(cal_params, f)
        
        logger.info(f"Calibration saved to {cal_path}")
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred):
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            return {"mae": round(mae, 2), "rmse": round(rmse, 2)}
        
        # Before calibration (on test set)
        test_sbp_before = calculate_metrics(y_sbp[test_indices], y_pred_sbp[test_indices])
        test_dbp_before = calculate_metrics(y_dbp[test_indices], y_pred_dbp[test_indices])
        
        # After calibration (on test set)
        test_sbp_after = calculate_metrics(y_sbp[test_indices], y_pred_sbp_cal[test_indices])
        test_dbp_after = calculate_metrics(y_dbp[test_indices], y_pred_dbp_cal[test_indices])
        
        return {
            "patient_id": patient_id,
            "method": method,
            "stratify": stratify,
            "n_calibration_samples": len(cal_indices),
            "n_test_samples": len(test_indices),
            "calibration_params": {
                "sbp_slope": round(sbp_slope, 4),
                "sbp_intercept": round(sbp_intercept, 2),
                "dbp_slope": round(dbp_slope, 4),
                "dbp_intercept": round(dbp_intercept, 2)
            },
            "test_metrics": {
                "before_calibration": {
                    "sbp": test_sbp_before,
                    "dbp": test_dbp_before
                },
                "after_calibration": {
                    "sbp": test_sbp_after,
                    "dbp": test_dbp_after
                }
            },
            "improvement": {
                "sbp_mae": round(test_sbp_before["mae"] - test_sbp_after["mae"], 2),
                "dbp_mae": round(test_dbp_before["mae"] - test_dbp_after["mae"], 2)
            },
            "message": "Calibration completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calibrating patient: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
