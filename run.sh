#!/bin/bash

if [ "$1" == "-g" ] && [ "$2" != "" ]; then
    cd frontend
    npx prettier --write .
    cd ..
    git add .
    git commit -m "$2"
    git push
    echo "Changes pushed to remote repository"
elif [ "$1" == "-g" ]; then
    cd frontend
    npx prettier --write .
    cd ..
    git add .
    git commit -m "Update"
    git push
    echo "Changes pushed to remote repository"
elif [ "$1" == "-f" ]; then
    cd frontend
    npm run dev
    cd ..
elif [ "$1" == "-fire" ]; then
    cd frontend
    npm run build
    firebase deploy
elif [ "$1" == "-b" ]; then
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8080
    cd ..
elif [ "$1" == "-lint" ]; then
    cd frontend
    npx prettier --write .
    cd ..
else
    echo "Usage: ./run.sh [-f] [-g] [commit message]"
    echo "  -f: Run frontend"
    echo "  -b: Run backend"
    echo "  -fire: Deploy backend to firebase"
    echo "  -lint: Run prettier on frontend"
    echo "  -g: Commit and push changes to remote repository (optional commit message)"
    echo "  -lint: Run prettier on frontend"
fi