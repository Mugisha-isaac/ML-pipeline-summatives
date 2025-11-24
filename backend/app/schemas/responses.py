"""Response schemas"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class PredictionResult(BaseModel):
    filename: str
    label: str
    confidence: float
    probability_good: float
    probability_bad: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchPredictionResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[PredictionResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelInfo(BaseModel):
    model_name: str
    status: str
    accuracy: Optional[float]
    total_samples_trained: Optional[int]
    classes: List[str]
    features_count: Optional[int]

class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str

class TrainingStatus(BaseModel):
    status: str
    progress: float
    epoch: Optional[int]
    total_epochs: Optional[int]
    current_loss: Optional[float]
    current_accuracy: Optional[float]
    message: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VisualizationData(BaseModel):
    feature_name: str
    data: Dict
    interpretation: str

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    class_labels: List[str]

class DataUploadResponse(BaseModel):
    total_files: int
    uploaded_files: int
    failed_files: int
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
