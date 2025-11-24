"""Request schemas"""
from pydantic import BaseModel, Field
from typing import Optional, List

class PredictionRequest(BaseModel):
    filename: str = Field(..., description="Name of the audio file")

class BatchPredictionRequest(BaseModel):
    filenames: List[str] = Field(..., description="List of audio filenames")

class RetrainingRequest(BaseModel):
    data_source: str = Field(default="uploads", description="Where to load training data from")
    epochs: Optional[int] = Field(default=100, description="Number of training epochs")
    batch_size: Optional[int] = Field(default=32, description="Batch size for training")
