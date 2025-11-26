"""
Application configuration settings
"""
import os
from pathlib import Path

# Project structure - handle both local and Docker environments
try:
    # Normal case: __file__ is available and resolvable
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except (NameError, AttributeError, RuntimeError):
    # Fallback: use current working directory
    BASE_DIR = Path("/app") if os.path.exists("/app") else Path.cwd()

APP_DIR = BASE_DIR / "app" if (BASE_DIR / "app").exists() else BASE_DIR

# Data directories
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"

# Model directories
MODELS_DIR = BASE_DIR / "models"

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"[SETTINGS] BASE_DIR: {BASE_DIR}")
print(f"[SETTINGS] APP_DIR: {APP_DIR}")
print(f"[SETTINGS] MODELS_DIR: {MODELS_DIR}")
print(f"[SETTINGS] DATA_DIR: {DATA_DIR}")

# Model file paths
MODEL_PATH = MODELS_DIR / "talent_classifier_model.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURES_CSV = MODELS_DIR / "audio_features.csv"

# Audio processing settings
SAMPLE_RATE = 22050
N_MFCC = 13
TOP_DB = 20

# Model settings
MODEL_CONFIDENCE_THRESHOLD = 0.5

# API settings
API_TITLE = "Audio Talent Classification API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "ML Pipeline for audio talent classification with retraining capabilities"

# File upload settings
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.ogg'}

# GitHub settings
GITHUB_REPO_URL = "https://raw.githubusercontent.com/Mugisha-isaac/ML-pipeline-summatives"

# Training settings
TRAINING_EPOCHS = 100
TRAINING_BATCH_SIZE = 32
TRAINING_TEST_SPLIT = 0.2

# Retraining settings
RETRAIN_MIN_SAMPLES = 10
