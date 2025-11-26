FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies - only essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for better compatibility
RUN pip install --upgrade pip

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from backend folder to /app
# This includes: app/, models/, data/, run.py, etc.
COPY . .

# Verify models and data are copied
RUN echo "Checking copied files:" && \
    ls -la /app/models/ 2>/dev/null || echo "WARNING: models directory not found" && \
    ls -la /app/data/ 2>/dev/null || echo "WARNING: data directory not found" && \
    ls -la /app/app/ 2>/dev/null || echo "WARNING: app directory not found"

# Create necessary directories for runtime operations
RUN mkdir -p /app/data/uploads /app/logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check - ensure endpoint returns 200 status
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health 2>/dev/null | grep -q "healthy" || exit 1

# Run the application
CMD ["python", "run.py"]
