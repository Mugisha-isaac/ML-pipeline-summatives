"""Global model manager instance - avoid circular imports"""
from app.utils.model import ModelManager

# Create single global instance
model_manager = ModelManager()
