# Quick Start: Deploy on Render in 5 Minutes

This is a simplified guide to get your app running on Render quickly.

## Prerequisites

- GitHub account with your code pushed
- Render account (free at https://render.com)

## Step 1: Create a Web Service

1. Go to https://render.com/dashboard
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect GitHub Account"** (if not already connected)
4. Select your repository
5. Click **"Connect"**

## Step 2: Configure the Service

Fill in these fields:

| Field | Value |
|-------|-------|
| **Name** | `audio-talent-api` |
| **Runtime** | `Docker` |
| **Build Command** | (leave empty) |
| **Start Command** | (leave empty) |
| **Instance Type** | `Free` (or `Starter` for production) |

## Step 3: Set Environment Variables

Click **"Environment Variables"** and add these:

```
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
DISABLE_LOAD_TESTING=true
TF_CPP_MIN_LOG_LEVEL=2
```

## Step 4: Configure Health Check

Scroll down and set:

| Field | Value |
|-------|-------|
| **Health Check Path** | `/health` |
| **Health Check Timeout** | `10` |
| **Health Check Interval** | `30` |

## Step 5: Deploy

Click **"Create Web Service"**

⏱️ Wait 5-15 minutes for deployment to complete

## Step 6: Access Your API

Your API will be available at:
```
https://audio-talent-api.onrender.com
```

### Useful Endpoints

- **API Docs**: https://audio-talent-api.onrender.com/docs
- **Health Check**: https://audio-talent-api.onrender.com/health
- **Predictions**: https://audio-talent-api.onrender.com/predict

## Troubleshooting

### Deployment Fails
1. Click **"Logs"** to see errors
2. Check if all model files are in the repository
3. Verify `requirements.txt` is complete
4. Ensure `Dockerfile` is in `backend/` directory

### App Crashes After Deployment
1. Check health check logs
2. Look for "Out of memory" errors
3. Verify model file path is correct
4. Increase "Start Period" to 60 seconds

### Slow First Request
- Model loading takes 30-60 seconds on first startup
- This is normal for ML applications

## Free Tier Limitations

- Service auto-stops after 15 minutes of inactivity
- 512 MB memory limit
- Limited compute power

### Keep Service Alive (Optional)

Add a cron job to Render dashboard:

1. Go to service settings
2. Click **"Cron Jobs"**
3. Add new cron job:
   - **Schedule**: `0 */6 * * *` (every 6 hours)
   - **Command**: `curl https://audio-talent-api.onrender.com/health`

## Next Steps

1. ✅ **Test the API**: Visit `/docs` endpoint
2. **Deploy Frontend** (if needed): Create another service for Next.js app
3. **Update Frontend**: Point to your new API URL
4. **Monitor**: Check logs regularly in Render dashboard
5. **Upgrade** (optional): Move to Starter plan for production

## Common Issues & Solutions

### "Docker build failed"
- Ensure `Dockerfile` exists in `backend/` directory
- Verify all dependencies in `requirements.txt`
- Check that `.dockerignore` isn't excluding needed files

### "Health check failed"
- Click **Settings** → Increase "Start Period" to 60s
- Ensure `/health` endpoint is working
- Check app logs in Render

### "Out of memory errors"
- Upgrade from Free to Starter tier
- Reduce concurrent workers (advanced)
- Optimize model size

### "Requests timing out"
- Check CPU usage in logs
- Model inference might be slow
- Increase instance tier

## Support

- **Render Docs**: https://render.com/docs
- **Check App Logs**: Dashboard → Your Service → Logs
- **GitHub Issues**: Create issue if bugs found

---

**Congratulations!** Your Audio Talent Classification app is now live on Render! 🎉
