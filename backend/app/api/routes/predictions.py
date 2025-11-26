"""Prediction API endpoints"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile
from datetime import datetime

from app.schemas.requests import PredictionRequest, BatchPredictionRequest
from app.schemas.responses import PredictionResult, BatchPredictionResponse
from app.utils.files import validate_audio_file
from app.core.model_instance import model_manager

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])

@router.post("/single", response_model=PredictionResult)
async def predict_single(file: UploadFile = File(...)):
    """Predict talent for a single audio file"""
    try:
        print(f"Received file: {file.filename}")
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Temp file created: {tmp_path}")
        
        # Validate
        is_valid, msg = validate_audio_file(tmp_path)
        if not is_valid:
            os.unlink(tmp_path)
            print(f"Validation failed: {msg}")
            raise HTTPException(status_code=400, detail=msg)
        
        print(f"File validated successfully")
        
        # Predict
        if not model_manager.model_loaded:
            print("Model not loaded, attempting to load...")
            model_manager.load_model()
        
        if not model_manager.is_model_ready():
            print("Model is not ready")
            raise HTTPException(status_code=503, detail="Model not available")
        
        print("Making prediction...")
        result = model_manager.predict(tmp_path)
        os.unlink(tmp_path)
        
        print(f"Prediction result: {result}")
        
        return PredictionResult(
            filename=file.filename,
            label=result['label'],
            confidence=result['confidence'],
            probability_good=result['probability_good'],
            probability_bad=result['probability_bad'],
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        print(f"Error in predict_single: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict talent for multiple audio files"""
    results = []
    successful = 0
    failed = 0
    total_files = len(files)
    
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    if not model_manager.is_model_ready():
        raise HTTPException(status_code=503, detail="Model not available")
    
    for file in files:
        tmp_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Validate audio file
            is_valid, msg = validate_audio_file(tmp_path)
            if not is_valid:
                failed += 1
                continue
            
            # Make prediction
            result = model_manager.predict(tmp_path)
            
            results.append(PredictionResult(
                filename=file.filename or "unknown",
                label=result['label'],
                confidence=result['confidence'],
                probability_good=result['probability_good'],
                probability_bad=result['probability_bad'],
                timestamp=datetime.utcnow()
            ))
            successful += 1
        except Exception as e:
            failed += 1
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return BatchPredictionResponse(
        total=total_files,
        successful=successful,
        failed=failed,
        results=results,
        timestamp=datetime.utcnow()
    )
