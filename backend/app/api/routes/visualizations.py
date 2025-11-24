"""Data visualization endpoints"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import json
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

from app.schemas.responses import VisualizationData
from app.config.settings import FEATURES_CSV

matplotlib.use('Agg')
router = APIRouter(prefix="/api/v1", tags=["Visualizations"])

@router.get("/visualizations/mfcc")
async def get_mfcc_visualization():
    """Get MFCC feature distribution visualization"""
    try:
        # Create sample visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        features = np.random.rand(13)
        ax.bar(range(len(features)), features)
        ax.set_xlabel('MFCC Coefficient')
        ax.set_ylabel('Mean Value')
        ax.set_title('MFCC Feature Distribution')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
            plt.close()
            return FileResponse(tmp.name, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizations/spectral")
async def get_spectral_visualization():
    """Get spectral features distribution"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Spectral Centroid', 'Spectral Rolloff', 'ZCR', 'RMS']
        values = np.random.rand(len(features)) * 1000
        ax.barh(features, values)
        ax.set_xlabel('Feature Value')
        ax.set_title('Spectral Features Distribution')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
            plt.close()
            return FileResponse(tmp.name, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualizations/feature-info")
async def get_feature_info():
    """Get feature interpretations"""
    interpretations = {
        "mfcc": {
            "description": "Mel-Frequency Cepstral Coefficients (MFCCs)",
            "interpretation": "MFCCs capture the characteristics of human hearing. They represent the short-term power spectrum of a sound based on a nonlinear mel scale of frequencies. Good singers show consistent MFCC patterns while poor singers show erratic patterns.",
            "good_singers": "High consistency and smooth transitions between MFCCs",
            "bad_singers": "Inconsistent MFCC values with frequent jumps"
        },
        "spectral_centroid": {
            "description": "Spectral Centroid (Center of Mass of Spectrum)",
            "interpretation": "The spectral centroid indicates the 'center of mass' of the spectrum. It correlates with the perceived brightness of sound. Good singers maintain a consistent spectral centroid showing tonal stability.",
            "good_singers": "Stable spectral centroid with less variation",
            "bad_singers": "Highly variable spectral centroid indicating lack of tonal control"
        },
        "zero_crossing_rate": {
            "description": "Zero Crossing Rate (ZCR)",
            "interpretation": "ZCR is the rate at which the audio signal changes sign. It's useful for distinguishing between voiced and unvoiced frames. Good singers have more structured ZCR patterns.",
            "good_singers": "Clear separation between voiced and unvoiced sections",
            "bad_singers": "Chaotic ZCR patterns without clear structure"
        }
    }
    return interpretations
