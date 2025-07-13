# 🚀 Deployment Guide

This guide will help you deploy the Transformer project to various platforms.

## 📋 Prerequisites

- Git repository with your code
- Node.js (for frontend)
- Python 3.9+ (for backend)
- Account on deployment platforms (Vercel, Heroku, Railway, etc.)

## 🎯 Quick Deployment Options

### Option 1: Automated Deployment (Recommended)

Run the automated deployment script:

```bash
./deploy.sh
```

This script will guide you through deploying both frontend and backend.

### Option 2: Manual Deployment

## 🌐 Frontend Deployment (Vercel)

### Step 1: Prepare Frontend

```bash
cd transformer-web
npm install
npm run build
```

### Step 2: Deploy to Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

### Step 3: Configure Environment Variables

In your Vercel dashboard, add:

- `BACKEND_URL`: Your backend API URL

## 🔧 Backend Deployment

### Option A: Heroku

#### Step 1: Install Heroku CLI

```bash
# macOS
brew install heroku/brew/heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

#### Step 2: Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Add Python buildpack
heroku buildpacks:add heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### Step 3: Check Deployment

```bash
heroku logs --tail
heroku open
```

### Option B: Railway

#### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

#### Step 2: Deploy

```bash
# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

### Option C: Docker

#### Step 1: Build and Run

```bash
# Build image
docker build -t transformer-backend .

# Run container
docker run -p 5000:5000 transformer-backend
```

#### Step 2: Using Docker Compose

```bash
# Run both frontend and backend
docker-compose up --build
```

## 🔗 Connect Frontend to Backend

### Step 1: Get Backend URL

After deploying your backend, note the URL:

- Heroku: `https://your-app-name.herokuapp.com`
- Railway: `https://your-app-name.railway.app`
- Docker: `http://localhost:5000`

### Step 2: Update Frontend Environment

In your Vercel dashboard or local `.env.local`:

```env
BACKEND_URL=https://your-backend-url.com
```

### Step 3: Redeploy Frontend

```bash
cd transformer-web
vercel --prod
```

## 🧪 Testing Deployment

### Test Backend API

```bash
# Health check
curl https://your-backend-url.com/health

# Test prediction
curl -X POST https://your-backend-url.com/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "hello", "task": "reverse"}'
```

### Test Frontend

1. Open your frontend URL
2. Try different tasks (copy, reverse, sort, shift)
3. Check if predictions work correctly

## 🔧 Troubleshooting

### Common Issues

#### 1. Backend Not Loading Model

**Error**: "No pre-trained model found"

**Solution**:

- Ensure `model_reverse.pth` is in your repository
- Or train a new model first:

```bash
python tasks/reverse/train_reverse.py
```

#### 2. CORS Errors

**Error**: "Access to fetch at '...' from origin '...' has been blocked by CORS policy"

**Solution**:

- Backend already includes CORS configuration
- Check if `BACKEND_URL` is correct in frontend

#### 3. Memory Issues on Heroku

**Error**: "Application error" or timeout

**Solution**:

- Upgrade to paid Heroku dyno
- Or use Railway/Render which have better free tiers

#### 4. Build Failures

**Error**: "Build failed"

**Solution**:

- Check `requirements.txt` for correct versions
- Ensure all dependencies are listed
- Check Python version compatibility

### Debug Commands

```bash
# Check backend logs
heroku logs --tail

# Check frontend build
cd transformer-web
npm run build

# Test backend locally
python app.py

# Test frontend locally
cd transformer-web
npm run dev
```

## 📊 Monitoring

### Health Checks

- Backend: `GET /health`
- Frontend: Check if page loads

### Performance Monitoring

- Vercel Analytics (frontend)
- Heroku Metrics (backend)
- Railway Metrics (backend)

## 🔄 Continuous Deployment

### GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Heroku
        uses: akhileshns/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: "your-app-name"
          heroku_email: "your-email@example.com"

  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
```

## 🎉 Success

Once deployed, you'll have:

- **Frontend**: Interactive web interface at `https://your-app.vercel.app`
- **Backend**: API at `https://your-backend.herokuapp.com`
- **Features**: Copy, reverse, sort, and shift sequence tasks

Share your deployed URLs and let others try your transformer model!

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review platform-specific documentation
3. Check logs for error messages
4. Ensure all environment variables are set correctly
