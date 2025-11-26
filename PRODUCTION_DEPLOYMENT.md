# Production Deployment Configuration

This document describes the production-ready Docker setup and Render deployment.

## Docker Image Improvements

### Multi-Stage Build
The Dockerfile uses a two-stage build process to optimize image size:
- **Builder Stage**: Installs all build dependencies and compiles Python packages
- **Runtime Stage**: Contains only runtime dependencies, significantly reducing image size

Benefits:
- Smaller final image (lighter deployment)
- Faster deployment
- Better security (fewer tools for potential exploits)

### Security Enhancements

1. **Non-Root User**: Application runs as `appuser` instead of root
   - Prevents privilege escalation
   - Limits damage if container is compromised

2. **Production Server**: Uses Gunicorn with Uvicorn workers
   - Multiple workers for concurrent request handling
   - Better resource management
   - Industry-standard for production Python apps

3. **Health Checks**: Built-in Docker health check
   - Automatically monitors application health
   - Renders can restart unhealthy containers
   - Verifies dependencies are loaded correctly

### Performance Optimization

1. **Virtual Environment**: Isolated Python packages
   - Clean dependency management
   - Reduced conflicts

2. **Environment Variables**:
   - `TF_CPP_MIN_LOG_LEVEL=2`: Reduces TensorFlow verbosity
   - `PYTHONDONTWRITEBYTECODE=1`: Prevents .pyc file generation
   - Optimizes startup time and resource usage

3. **Worker Configuration**:
   - 4 workers for concurrent request handling
   - 300-second timeout for long-running requests
   - Suitable for Render's container specifications

## Build Configuration

### Requirements
- All dependencies in `requirements.txt`
- Gunicorn installed for production server
- Uvicorn workers for async support

### Build Steps
1. Updates package manager and installs build dependencies
2. Installs Python packages in virtual environment
3. Copies only necessary application files
4. Sets up security and monitoring
5. Configures production server

### Image Size
- Expected size: ~1.2-1.5 GB
- Multi-stage build reduces size by ~30-40% compared to single stage
- Large size due to:
  - TensorFlow 2.17 (heavy ML library)
  - NumPy, SciPy, Pandas (scientific computing)
  - Librosa (audio processing)

## Render Deployment

### Service Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Runtime | Docker | Full control over environment |
| Instance Type | Starter (Free) | Entry-level, upgradable |
| Port | 8000 | Backend API port |
| Health Check Interval | 30s | Frequent monitoring |
| Health Check Timeout | 10s | Quick failure detection |
| Start Period | 40s | Allows model loading |

### Environment Variables

```
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
DISABLE_LOAD_TESTING=true
TF_CPP_MIN_LOG_LEVEL=2
WORKERS=4
HOST=0.0.0.0
PORT=8000
```

### Performance Considerations

#### Startup Time
- Expected: 30-60 seconds
- Model loading adds 20-40 seconds
- Health check needs 40-second grace period

#### Memory Usage
- Expected: 400-600 MB under normal load
- TensorFlow pre-allocates memory
- Upgrade instance if: 
  - Out of memory errors appear
  - Multiple concurrent requests timeout
  - Health checks fail

#### CPU Usage
- 4 workers distribute load
- Reduce if: CPU constantly at 100%
- Increase if: Requests queue up

## Testing Before Production

### Local Docker Testing

```bash
# Build and run locally
./run-docker.sh

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs

# Test predictions (with audio file)
curl -X POST http://localhost:8000/predict \
  -F "file=@audio.wav"
```

### Production Simulation

```bash
# Run with production settings
docker run \
  -e DISABLE_LOAD_TESTING=true \
  -e PYTHONUNBUFFERED=1 \
  -p 8000:8000 \
  audio-talent-api:latest
```

## Troubleshooting

### Build Fails
- Check logs for dependency conflicts
- Verify all files exist (models, data directories)
- Ensure requirements.txt is up-to-date

### Container Exits
- Review startup logs
- Check if port is in use
- Verify environment variables are set

### Health Check Fails
- Increase start-period if model loading is slow
- Check if `/health` endpoint is working
- Verify dependencies are installed correctly

### Out of Memory
- Reduce number of workers
- Upgrade Render instance
- Optimize model or preprocessing

### Slow Requests
- Check CPU usage
- Review model inference time
- Optimize audio processing
- Consider model quantization

## Scaling Considerations

### Vertical Scaling (Upgrade Instance)
- Free → Starter: $7/month
- Starter → Professional: $12/month
- Recommended for: High memory needs, CPU-intensive operations

### Horizontal Scaling (Multiple Instances)
- Use Render's load balancer
- Requires Starter plan or higher
- Share database for state management

### Optimization Steps
1. Profile application locally
2. Identify bottlenecks
3. Optimize code or model
4. Test with production-like data
5. Monitor after deployment

## Monitoring and Maintenance

### Daily Tasks
- Check service status
- Review error logs
- Monitor memory/CPU usage

### Weekly Tasks
- Analyze request patterns
- Review slow requests
- Check for dependency updates

### Monthly Tasks
- Update dependencies
- Review security updates
- Performance optimization review
- Backup important data

## Security Checklist

- [x] Non-root user in container
- [x] Environment variables in Render dashboard (not in code)
- [x] Health check endpoint working
- [x] Only necessary files copied to image
- [x] Production logging enabled
- [x] Request timeouts configured
- [x] Load testing disabled in production
- [ ] CORS configured for frontend domain
- [ ] API authentication (if needed)
- [ ] HTTPS enforced (Render provides automatically)

## Cost Estimation

### Free Tier (Render)
- Service: Free
- Costs: $0/month
- Limitations:
  - Auto-spins down after 15 minutes inactivity
  - Shared resources
  - 512 MB memory limit
  - Keep-alive cron job needed

### Starter Tier
- Service: $12/month per service
- Benefits:
  - Always running
  - 1 GB memory
- Good for: Production testing, small users

### Recommended for Production
- Professional: $42/month per service
- Production: $99/month per service
- Benefits: More memory, guaranteed resources, priority support

## Next Steps

1. Test Docker build locally: `./run-docker.sh`
2. Create Render account
3. Connect GitHub repository
4. Deploy using render.yaml or web interface
5. Set environment variables in Render
6. Monitor logs and health checks
7. Perform load testing
8. Implement monitoring/alerting
9. Set up backup strategy
10. Document runbooks for operations team
