"""Prediction API endpoints"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile
from datetime import datetime

from app.schemas.requests import PredictionRequest, BatchPredictionRequest
from app.schemas.responses import PredictionResult, BatchPredictionResponse
from app.utils.model import ModelManager
from app.utils.files import validate_audio_file

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])
model_manager = ModelManager()

@router.post("/single", response_model=PredictionResult)
async def predict_single(file: UploadFile = File(...)):
    """Predict talent for a single audio file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Validate
        is_valid, msg = validate_audio_file(tmp_path)
        if not is_valid:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=msg)
        
        # Predict
        if not model_manager.model_loaded:
            model_manager.load_model()
        
        if not model_manager.is_model_ready():
            raise HTTPException(status_code=503, detail="Model not available")
        
        result = model_manager.predict(tmp_path)
        os.unlink(tmp_path)
        
        return PredictionResult(
            filename=file.filename,
            label=result['label'],
            confidence=result['confidence'],
            probability_good=result['probability_good'],
            probability_bad=result['probability_bad'],
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list = File(...)):
    """Predict talent for multiple audio files"""
    results = []
    successful = 0
    failed = 0
    
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            is_valid, msg = validate_audio_file(tmp_path)
            if not is_valid:
                failed += 1
                os.unlink(tmp_path)
                continue
            
            result = model_manager.predict(tmp_path)
            os.unlink(tmp_path)
            
            results.append(PredictionResult(
                filename=file.filename,
                label=result['label'],
                confidence=result['confidence'],
                probability_good=result['probability_good'],
                probability_bad=result['probability_bad'],
                timestamp=datetime.utcnow()
            ))
            successful += 1
        except Exception as e:
            failed += 1
    
    return BatchPredictionResponse(
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        timestamp=datetime.utcnow()
    )
