"""Main FastAPI Application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager

from app.config.settings import API_TITLE, API_VERSION, API_DESCRIPTION
from app.api.routes import api_router
from app.utils.model import ModelManager

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Loading ML model...")
    model_manager.load_model()
    print("Model loaded successfully!")
    yield
    # Shutdown
    print("Application shutting down...")

# Create app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Audio Talent Classification API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/version")
async def get_version():
    """Get API version"""
    return {"version": API_VERSION}
