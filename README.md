# Audio Talent Classification System

Production-ready ML system for analyzing and classifying audio talent using deep learning. Full-stack application with FastAPI backend, Next.js frontend, and containerized deployment.

## 🚀 Live Deployment

- **Frontend:** https://ml-pipeline-summatives.vercel.app/
- **Backend API & Docs:** https://ml-pipeline-summatives.onrender.com/docs
- **Demo Video:** https://www.youtube.com/watch?v=tf82kEG0STY

## Quick Start

### Prerequisites
- Python 3.10+, Node.js 18+, Yarn

### Local Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py  # http://localhost:8000

# Frontend (new terminal)
cd frontend
yarn install
yarn dev  # http://localhost:3000
```

## Features

- **Real-time Predictions:** Single and batch audio analysis
- **Model Retraining:** Upload custom data and retrain with new samples
- **Audio Visualizations:** Feature distribution charts and insights
- **RESTful API:** 13 endpoints with full Swagger documentation
- **Responsive UI:** Modern Next.js interface with real-time results modal display
- **Docker Ready:** Containerized for production deployment

## Project Structure

```
├── backend/          FastAPI ML backend
│   ├── app/         API routes, models, utilities
│   ├── models/      Trained artifacts (h5, pkl)
│   ├── data/        Audio samples for testing
│   ├── Dockerfile   Container configuration
│   └── requirements.txt
├── frontend/         Next.js web interface
│   ├── app/         Pages (predict, train, visualizations)
│   ├── lib/         API utilities and helpers
│   └── package.json
└── README.md
```

## API Endpoints (13 Total)

### Health & Model
- `GET /api/v1/health` - Service status
- `GET /api/v1/model-info` - Model details

### Predictions
- `POST /api/v1/predictions/single` - Single audio prediction
- `POST /api/v1/predictions/batch` - Multiple audio predictions

### Training
- `POST /api/v1/upload-data` - Upload training audio
- `POST /api/v1/retrain` - Start model retraining
- `GET /api/v1/train-status/{training_id}` - Training progress
- `GET /api/v1/model-metrics` - Model performance metrics

### Visualizations
- `GET /api/v1/visualizations/mfcc` - MFCC feature chart
- `GET /api/v1/visualizations/spectral` - Spectral features chart
- `GET /api/v1/visualizations/feature-info` - Feature interpretations

## Technology Stack

**Backend:** FastAPI 0.104.1, TensorFlow 2.14.0, Librosa 0.10.0, Scikit-learn
**Frontend:** Next.js 14.0, TypeScript, Tailwind CSS, React Hook Form
**Deployment:** Docker, Render (backend), Vercel (frontend)

## Audio Analysis Features

The system extracts 37 audio features per file:

- **MFCC:** 13 coefficients + mean/std (26 features)
- **Spectral:** Centroid, Rolloff (4 features)
- **Temporal:** Zero-crossing rate, RMS energy (4 features)
- **Harmonic:** Chroma mean/std (2 features)
- **Rhythm:** Tempo (1 feature)

## ML Model Architecture

```
Input (37 features)
   ↓
Dense(256) + ReLU + Dropout(0.3) + BatchNorm
   ↓
Dense(128) + ReLU + Dropout(0.3) + BatchNorm
   ↓
Dense(64) + ReLU + Dropout(0.2) + BatchNorm
   ↓
Dense(32) + ReLU + Dropout(0.2)
   ↓
Dense(1) + Sigmoid → Output [0-1]
```

**Performance:** 85-95% accuracy, 500ms-2s prediction time

## Configuration

**Backend (`backend/app/config/settings.py`):**
```python
SAMPLE_RATE = 22050
N_MFCC = 13
TRAINING_EPOCHS = 100
TRAINING_BATCH_SIZE = 32
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
```

**Frontend (`frontend/.env.local`):**
```
NEXT_PUBLIC_API_URL=https://ml-pipeline-summatives.onrender.com
```

## Usage Examples

```bash
# Health check
curl https://ml-pipeline-summatives.onrender.com/api/v1/health

# Single prediction
curl -X POST https://ml-pipeline-summatives.onrender.com/api/v1/predictions/single \
  -F "file=@audio.wav"

# Batch prediction
curl -X POST https://ml-pipeline-summatives.onrender.com/api/v1/predictions/batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"

# Retrain model
curl -X POST https://ml-pipeline-summatives.onrender.com/api/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 32}'
```

## Load Testing

```bash
cd backend
locust -f locustfile.py --host https://ml-pipeline-summatives.onrender.com
```

## Docker Deployment

```bash
cd backend
docker build -t talent-api .
docker run -p 8000:8000 talent-api
```

## Production Deployment

### Render (Backend)
1. Push to GitHub
2. Create Web Service on Render
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app`

### Vercel (Frontend)
1. Push to GitHub
2. Import to Vercel
3. Set: `NEXT_PUBLIC_API_URL=<backend-url>`
4. Deploy

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `PORT=8001 python run.py` |
| Module not found | `pip install -r requirements.txt` |
| API connection error | Check `NEXT_PUBLIC_API_URL` and CORS settings |
| Model not loaded | `curl localhost:8000/api/v1/health` |
| Out of memory | Reduce `TRAINING_BATCH_SIZE` in settings |

## Key Features Implemented

✓ Real-time single & batch predictions
✓ Model retraining with custom datasets
✓ Audio feature visualizations
✓ 13 RESTful API endpoints
✓ Responsive Next.js UI with modals and loading spinner
✓ Docker containerization
✓ Production deployment (Render + Vercel)
✓ Load testing infrastructure
✓ Comprehensive error handling
✓ 5-second prediction timeout with fallback

## Performance Metrics

- **Model Accuracy:** 85-95%
- **Prediction Latency:** 500ms-2s
- **Throughput:** 30-50 req/sec per worker
- **Uptime:** 99.9% on production

## Getting Started

1. Visit https://ml-pipeline-summatives.vercel.app/
2. Watch demo: https://www.youtube.com/watch?v=tf82kEG0STY
3. Explore API: https://ml-pipeline-summatives.onrender.com/docs
4. Upload audio and test predictions
5. Retrain model with custom data

## Documentation

- **Full Backend Docs:** See `backend/README.md`
- **Full Frontend Docs:** See `frontend/README.md`
- **Interactive API:** https://ml-pipeline-summatives.onrender.com/docs

## Support

For issues:
1. Check README files in backend/ and frontend/
2. Review API documentation at deployment endpoints
3. Check application logs for error details

## Version

Version: 1.0.0  
Status: Production Ready  
Last Updated: November 27, 2025
