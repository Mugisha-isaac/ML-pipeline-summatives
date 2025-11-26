# Production Deployment Guide - Render

This guide explains how to deploy the Audio Talent Classification application on Render.

## Prerequisites

- GitHub account with the repository pushed
- Render account (https://render.com)
- Docker and necessary build tools (for local testing)

## Deployment Steps

### 1. Backend API Deployment

#### Option A: Using Docker (Recommended)

1. **Connect to Render**
   - Go to https://render.com/dashboard
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Select the repository branch (main)

2. **Configure the Service**
   - **Service Name**: `audio-talent-api`
   - **Runtime**: `Docker`
   - **Build Command**: (leave empty - uses Dockerfile)
   - **Start Command**: (leave empty - uses CMD from Dockerfile)
   - **Instance Type**: `Starter` (free tier) or higher for production

3. **Set Environment Variables**
   ```
   PYTHONUNBUFFERED=1
   PYTHONDONTWRITEBYTECODE=1
   DISABLE_LOAD_TESTING=true
   TF_CPP_MIN_LOG_LEVEL=2
   WORKERS=4
   HOST=0.0.0.0
   PORT=8000
   ```

4. **Network Configuration**
   - **Port**: `8000`
   - **Health Check Path**: `/health`
   - **Health Check Timeout**: `10` seconds
   - **Health Check Interval**: `30` seconds

5. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete (5-15 minutes)
   - Your API will be available at: `https://audio-talent-api.onrender.com`

#### Option B: Using render.yaml

1. Place the `render.yaml` file in your repository root
2. Connect to Render and select "Infrastructure as Code"
3. Render will automatically deploy according to the configuration

### 2. Frontend Deployment

#### Option A: Static Site (Recommended for Next.js)

1. Create a new Web Service on Render
2. **Configure**:
   - **Build Command**: `cd frontend && npm install && npm run build && npm run export`
   - **Publish Directory**: `frontend/out`
   - **Environment Variables**: 
     ```
     NEXT_PUBLIC_API_URL=https://audio-talent-api.onrender.com
     ```

3. **Deploy**
   - Render will automatically deploy on push to main

#### Option B: Docker

1. Create a Dockerfile in the `frontend/` directory:
   ```dockerfile
   FROM node:18-alpine AS builder
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   RUN npm run build

   FROM node:18-alpine
   WORKDIR /app
   COPY --from=builder /app/.next ./.next
   COPY --from=builder /app/public ./public
   COPY --from=builder /app/package*.json ./
   RUN npm ci --only=production
   EXPOSE 3000
   CMD ["npm", "start"]
   ```

2. Deploy as Docker Web Service

### 3. Environment Variables Configuration

#### Backend Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONUNBUFFERED` | `1` | Real-time log output |
| `PYTHONDONTWRITEBYTECODE` | `1` | Don't create .pyc files |
| `DISABLE_LOAD_TESTING` | `true` | Disable Locust on production |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Reduce TensorFlow verbosity |
| `WORKERS` | `4` | Number of gunicorn workers |
| `HOST` | `0.0.0.0` | Listen on all interfaces |
| `PORT` | `8000` | Service port (set by Render) |

#### Frontend Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NEXT_PUBLIC_API_URL` | `https://audio-talent-api.onrender.com` | Backend API endpoint |

### 4. Health Checks

The application includes a built-in health check endpoint at `/health` that verifies:
- Database connectivity (if applicable)
- Model file availability
- Service status

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-26T10:30:00Z"
}
```

### 5. Database Configuration (if needed)

If using PostgreSQL or MySQL:

1. Create a database on Render
2. Add connection string to environment variables:
   ```
   DATABASE_URL=postgresql://user:password@host:port/dbname
   ```
3. Update application to use the connection string

### 6. Monitoring and Debugging

#### View Logs
- Go to your service dashboard
- Click "Logs" to see real-time output
- Check for deployment errors or runtime issues

#### Performance Monitoring
- Use Render's built-in metrics
- Monitor CPU usage, memory, and request rates
- Set up alerts for service interruptions

#### Common Issues

1. **Build Fails**: Check Docker build output in logs
   - Ensure all dependencies are in `requirements.txt`
   - Verify Dockerfile syntax

2. **Out of Memory**: Increase instance plan
   - TensorFlow models can be memory-intensive
   - Consider using smaller model or optimizations

3. **Slow Startup**: 
   - Model loading time increases startup duration
   - Set appropriate health check delays

4. **High CPU Usage**:
   - Reduce number of workers if constrained
   - Optimize model inference

### 7. Performance Optimization Tips

1. **Model Optimization**
   - Use TensorFlow Lite for smaller models
   - Implement model caching
   - Consider quantization

2. **Docker Image Optimization**
   - Multi-stage builds reduce image size
   - Remove unnecessary dependencies
   - Use `.dockerignore` effectively

3. **Application Configuration**
   - Disable load testing in production (`DISABLE_LOAD_TESTING=true`)
   - Use appropriate worker count based on plan
   - Implement request timeouts

4. **Free Tier Considerations**
   - Auto-spindown after 15 minutes of inactivity
   - Limited memory (512 MB)
   - Keep-alive cron job recommended
   - Consider upgrading for production use

### 8. Maintenance and Updates

1. **Code Updates**
   - Push changes to main branch
   - Render automatically redeploys
   - Monitor logs during deployment

2. **Dependency Updates**
   - Update `requirements.txt` for Python packages
   - Test locally before pushing
   - Monitor for security updates

3. **Model Updates**
   - Store models in `/app/models` directory
   - Rebuild Docker image after model changes
   - Version your models for rollback capability

### 9. Production Checklist

- [ ] Environment variables configured
- [ ] Health check endpoint verified
- [ ] Logs accessible and monitored
- [ ] Database connection tested (if applicable)
- [ ] API endpoints responding correctly
- [ ] Frontend connects to backend API
- [ ] Model files properly loaded
- [ ] Error handling in place
- [ ] Performance acceptable
- [ ] Security best practices followed

### 10. Support and Troubleshooting

For issues or questions:
1. Check Render documentation: https://render.com/docs
2. Review application logs in Render dashboard
3. Test locally with Docker: `docker build -t app . && docker run -p 8000:8000 app`
4. Verify all environment variables are set
5. Ensure model files are included in Docker build

## Rollback Procedure

If a deployment causes issues:

1. Go to Render dashboard
2. Select your service
3. Click "Deployments"
4. Select the previous successful deployment
5. Click "Redeploy"

The service will be restored to the previous working version.
