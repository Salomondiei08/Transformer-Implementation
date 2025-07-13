#!/bin/bash

# Railway Deployment Script
echo "🚂 Railway Deployment Script"
echo "============================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Link to project (if not already linked)
echo "🔗 Linking to Railway project..."
railway link

# Deploy
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at the URL provided by Railway" 