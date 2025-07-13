#!/bin/bash

echo "🚀 Transformer Project Deployment Script"
echo "========================================"
``
# Check if we're in the right directory
if [ ! -f "transformer.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to deploy frontend to Vercel
deploy_frontend() {
    echo "📦 Deploying Frontend to Vercel..."
    
    cd transformer-web
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        echo "📥 Installing Vercel CLI..."
        npm install -g vercel
    fi
    
    # Build the project
    echo "🔨 Building Next.js project..."
    npm run build
    
    # Deploy to Vercel
    echo "🚀 Deploying to Vercel..."
    vercel --prod
    
    cd ..
    echo "✅ Frontend deployed successfully!"
}

# Function to deploy backend to Railway
deploy_backend() {
    echo "📦 Deploying Backend to Railway..."
    
    # Check if Railway CLI is installed
    if ! command -v railway &> /dev/null; then
        echo "📥 Installing Railway CLI..."
        npm install -g @railway/cli
    fi
    
    # Login to Railway
    echo "🔐 Logging into Railway..."
    railway login
    
    # Initialize Railway project
    echo "🚂 Initializing Railway project..."
    railway init
    
    # Deploy
    echo "🚀 Deploying to Railway..."
    railway up
    
    echo "✅ Backend deployed successfully!"
}

# Function to deploy backend to Heroku
deploy_backend_heroku() {
    echo "📦 Deploying Backend to Heroku..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        echo "📥 Installing Heroku CLI..."
        curl https://cli-assets.heroku.com/install.sh | sh
    fi
    
    # Login to Heroku
    echo "🔐 Logging into Heroku..."
    heroku login
    
    # Create Heroku app
    echo "🏗️ Creating Heroku app..."
    heroku create transformer-playground-backend
    
    # Add buildpacks
    echo "📦 Adding buildpacks..."
    heroku buildpacks:add heroku/python
    
    # Deploy
    echo "🚀 Deploying to Heroku..."
    git add .
    git commit -m "Deploy to Heroku"
    git push heroku main
    
    echo "✅ Backend deployed successfully!"
}

# Main deployment menu
echo "Choose deployment option:"
echo "1. Deploy Frontend to Vercel"
echo "2. Deploy Backend to Railway"
echo "3. Deploy Backend to Heroku"
echo "4. Deploy Both (Frontend + Backend)"
echo "5. Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        deploy_frontend
        ;;
    2)
        deploy_backend
        ;;
    3)
        deploy_backend_heroku
        ;;
    4)
        echo "🔄 Deploying both frontend and backend..."
        deploy_backend_heroku
        deploy_frontend
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo "🎉 Deployment complete!"
echo "📝 Next steps:"
echo "   1. Update the BACKEND_URL environment variable in your frontend deployment"
echo "   2. Test the deployed application"
echo "   3. Share the URLs with others!" 