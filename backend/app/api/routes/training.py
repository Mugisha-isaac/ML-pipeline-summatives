"""Model training and retraining endpoints"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import tempfile
import pandas as pd
from uuid import uuid4
from datetime import datetime
from typing import Dict

from app.schemas.requests import RetrainingRequest
from app.schemas.responses import TrainingResponse, TrainingStatus, DataUploadResponse, ModelMetrics
from app.utils.files import validate_audio_file, save_upload_file
from app.utils.audio import FeatureExtractor, AudioPreprocessor
from app.models_ml.trainer import ModelTrainer
from app.config.settings import UPLOAD_DIR, RETRAIN_MIN_SAMPLES
from app.core.model_instance import model_manager

router = APIRouter(prefix="/api/v1", tags=["Training"])

# Track training status per training_id
training_jobs: Dict[str, Dict] = {}

@router.post("/upload-data", response_model=DataUploadResponse)
async def upload_training_data(files: list = File(...)):
    """Upload audio files for retraining"""
    uploaded = 0
    failed = 0
    
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
            
            save_upload_file(tmp_path, str(UPLOAD_DIR))
            uploaded += 1
        except Exception as e:
            failed += 1
    
    total = len(files)
    message = f"Uploaded {uploaded} files successfully"
    if failed > 0:
        message += f", {failed} failed"
    
    return DataUploadResponse(
        total_files=total,
        uploaded_files=uploaded,
        failed_files=failed,
        message=message,
        timestamp=datetime.utcnow()
    )

@router.post("/retrain", response_model=TrainingResponse)
async def trigger_retraining(request: RetrainingRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    training_id = str(uuid4())
    
    # Extract features from uploaded files
    audio_files = []
    if os.path.exists(str(UPLOAD_DIR)):
        for f in os.listdir(str(UPLOAD_DIR)):
            if f.endswith('.wav'):
                audio_files.append(os.path.join(str(UPLOAD_DIR), f))
    
    if len(audio_files) < RETRAIN_MIN_SAMPLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least {RETRAIN_MIN_SAMPLES} samples, found {len(audio_files)}"
        )
    
    # Initialize training status for this job
    training_jobs[training_id] = {
        "status": "started",
        "progress": 0,
        "message": f"Retraining started with {len(audio_files)} samples"
    }
    
    # Start training in background
    background_tasks.add_task(
        _retrain_model,
        audio_files,
        request.epochs,
        request.batch_size,
        training_id
    )
    
    return TrainingResponse(
        status="started",
        message=f"Retraining started with {len(audio_files)} samples",
        training_id=training_id,
        timestamp=datetime.utcnow()
    )

async def _retrain_model(audio_files, epochs, batch_size, training_id):
    """Background task for model retraining"""
    try:
        training_jobs[training_id]["status"] = "training"
        training_jobs[training_id]["progress"] = 10
        training_jobs[training_id]["message"] = "Extracting features"
        
        # Extract features
        preprocessor = AudioPreprocessor()
        feature_extractor = FeatureExtractor()
        features_list = []
        
        for idx, audio_file in enumerate(audio_files):
            try:
                audio, sr = preprocessor.load_audio_file(audio_file)
                audio_clean = preprocessor.remove_silence(audio)
                features = feature_extractor.extract_features(audio_clean)
                features_flat = feature_extractor.flatten_features(features)
                features_flat['filename'] = os.path.basename(audio_file)
                features_flat['label'] = 'good' if 'good' in audio_file.lower() else 'bad'
                features_flat['augmentation'] = 'original'
                features_list.append(features_flat)
                
                progress = 10 + int((idx / len(audio_files)) * 40)
                training_jobs[training_id]["progress"] = progress
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        
        if not features_list:
            training_jobs[training_id]["status"] = "failed"
            training_jobs[training_id]["message"] = "No valid features extracted"
            return
        
        # Train model
        df = pd.DataFrame(features_list)
        trainer = ModelTrainer()
        
        training_jobs[training_id]["message"] = "Building model"
        feature_count = trainer.get_feature_count(df)
        trainer.build_model(feature_count)
        
        training_jobs[training_id]["message"] = "Preparing data"
        training_jobs[training_id]["progress"] = 60
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        training_jobs[training_id]["message"] = "Training model"
        training_jobs[training_id]["progress"] = 70
        trainer.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
        
        training_jobs[training_id]["message"] = "Evaluating model"
        training_jobs[training_id]["progress"] = 90
        metrics = trainer.evaluate(X_test, y_test)
        
        # Save model
        model_manager.model = trainer.model
        model_manager.scaler = trainer.scaler
        model_manager.label_encoder = trainer.label_encoder
        model_manager.model_loaded = True
        model_manager.save_model()
        
        training_jobs[training_id]["status"] = "completed"
        training_jobs[training_id]["progress"] = 100
        training_jobs[training_id]["message"] = f"Training completed. Accuracy: {metrics['accuracy']:.4f}"
    except Exception as e:
        print(f"Training error for {training_id}: {e}")
        import traceback
        traceback.print_exc()
        training_jobs[training_id]["status"] = "failed"
        training_jobs[training_id]["progress"] = 0
        training_jobs[training_id]["message"] = str(e)

@router.get("/train-status/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Get training status for a specific training job"""
    if training_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Training job {training_id} not found")
    
    job = training_jobs[training_id]
    return TrainingStatus(
        status=job.get("status", "idle"),
        progress=job.get("progress", 0),
        message=job.get("message", None),
        timestamp=datetime.utcnow()
    )

@router.get("/model-metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics"""
    if not model_manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Return zero metrics if model just loaded without training info
    return ModelMetrics(
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        confusion_matrix=None
    )
