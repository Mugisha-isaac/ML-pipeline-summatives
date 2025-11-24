"""API routes"""
from fastapi import APIRouter
from app.api.routes import predictions, health, training, visualizations

api_router = APIRouter()
api_router.include_router(predictions.router)
api_router.include_router(health.router)
api_router.include_router(training.router)
api_router.include_router(visualizations.router)
