#!/bin/bash

# Run Flask backend
printf '\n\033[1;34m[INFO]\033[0m Starting Flask backend (app.py) on http://localhost:5000 ...\n'
export FLASK_APP=app.py
export FLASK_ENV=development
python3 app.py &
BACKEND_PID=$!

# Wait a bit to ensure backend starts
sleep 3

# Run Next.js frontend
printf '\n\033[1;34m[INFO]\033[0m Starting Next.js frontend (transformer-web) on http://localhost:3000 ...\n'
cd transformer-web
npm install
npm run dev &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID
wait $FRONTEND_PID 