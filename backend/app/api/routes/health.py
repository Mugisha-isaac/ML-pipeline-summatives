"""Health and status check endpoints"""
from fastapi import APIRouter
from datetime import datetime

from app.schemas.responses import HealthStatus, ModelInfo
from app.config.settings import API_VERSION
from app.main import model_manager

router = APIRouter(prefix="/api/v1", tags=["Health"])

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    return HealthStatus(
        status="healthy",
        model_loaded=model_manager.is_model_ready(),
        version=API_VERSION,
        timestamp=datetime.utcnow()
    )

@router.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    classes = list(model_manager.label_encoder.classes_) if model_manager.label_encoder else []
    
    return ModelInfo(
        model_name="Audio Talent Classifier",
        status="ready" if model_manager.is_model_ready() else "not_loaded",
        accuracy=None,
        total_samples_trained=None,
        classes=classes,
        features_count=None
    )
