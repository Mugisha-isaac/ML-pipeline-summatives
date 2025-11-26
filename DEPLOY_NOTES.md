# Deployment Fix - Model Manager Issue

## Problem
The API endpoints were getting 500 errors on predictions because each route was creating its own new instance of `ModelManager()`, and these instances didn't have the model loaded. Only the global instance in `main.py` had the model loaded during startup.

## Solution
Updated all route files to import the global `model_manager` instance from `app.main` instead of creating new instances:

### Files Changed:
1. **app/api/routes/predictions.py** - Changed to use global model_manager
2. **app/api/routes/health.py** - Changed to use global model_manager  
3. **app/api/routes/training.py** - Changed to use global model_manager

### Before:
```python
from app.utils.model import ModelManager
model_manager = ModelManager()  # NEW INSTANCE - no model loaded!
```

### After:
```python
from app.main import model_manager  # GLOBAL INSTANCE - model loaded at startup
```

## Why This Works
- `app.main.py` creates the global `model_manager` instance
- During app startup (lifespan), the model is loaded once
- All routes now use this same loaded instance
- Predictions work correctly because the model is available

## Testing
All endpoints should now work:
- GET /health - Should show model_loaded: true
- POST /api/v1/predictions/single - Should work with audio files
- GET /api/v1/visualizations/* - Already working

## Deployment
1. Commit and push these changes
2. Redeploy on Render
3. Test predictions on https://ml-pipeline-summatives.onrender.com/docs
