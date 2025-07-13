# Railway Deployment Troubleshooting Guide

## Common Issues and Solutions

### 1. Port Configuration Issues

**Problem**: App not starting or port conflicts
**Solution**:

- Railway automatically sets the `PORT` environment variable
- Use `$PORT` in your Procfile: `web: gunicorn --bind 0.0.0.0:$PORT app:app`
- Don't hardcode port numbers

### 2. Model File Not Found

**Problem**: `model_reverse.pth` not loading
**Solution**:

- The app now handles missing model files gracefully
- It will use an untrained model if the file is not found
- Make sure the model file is committed to your repository

### 3. Build Failures

**Problem**: Build process failing
**Solution**:

- Check that all dependencies are in `requirements.txt`
- Ensure Python version is compatible (3.9.18)
- Use the `.dockerignore` file to exclude unnecessary files

### 4. Memory Issues

**Problem**: App running out of memory
**Solution**:

- Railway has memory limits
- Consider using a smaller model or fewer workers
- Monitor memory usage in Railway dashboard

### 5. Health Check Failures

**Problem**: Health checks failing
**Solution**:

- The `/health` endpoint is now available
- Railway will use this for health checks
- Make sure the endpoint returns quickly

## Deployment Steps

1. **Install Railway CLI**:

   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**:

   ```bash
   railway login
   ```

3. **Link your project**:

   ```bash
   railway link
   ```

4. **Deploy**:

   ```bash
   railway up
   ```

## Configuration Files

### railway.toml

- Configures build and deployment settings
- Sets environment variables
- Defines health check path

### Procfile

- Tells Railway how to start your app
- Uses `$PORT` environment variable
- Specifies gunicorn with 2 workers

### .dockerignore

- Excludes unnecessary files from build
- Reduces build time and size
- Improves deployment reliability

## Environment Variables

Railway automatically sets:

- `PORT`: The port your app should listen on
- `RAILWAY_STATIC_URL`: For static file serving

You can set custom variables in Railway dashboard:

- `FLASK_ENV`: Set to "production"
- `PYTHONPATH`: Set to "/app"

## Monitoring

1. **Check logs**: `railway logs`
2. **Monitor resources**: Railway dashboard
3. **Health check**: Visit `/health` endpoint
4. **Test endpoints**: Use `/`, `/predict`, `/tasks`

## Common Commands

```bash
# Deploy to Railway
railway up

# View logs
railway logs

# Open in browser
railway open

# Check status
railway status

# Set environment variables
railway variables set KEY=VALUE
```

## Support

If you're still having issues:

1. Check Railway documentation
2. Review build logs in Railway dashboard
3. Test locally with `python app.py`
4. Verify all files are committed to git
