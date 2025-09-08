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
    # Check if status is already prod
    if grep -q "prod" api.jsx; then
        echo "Status is already set to production"
    else
        # Update the status in the api.jsx file to the production URL
        sed -i 's|dev|prod|g' api.jsx
        echo "Status updated to production"
    fi
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
elif [ "$1" == "-build-deploy" ]; then
    cd backend
    gcloud builds submit . \
        --config=cloudbuild.yml \
        --verbosity=debug
    cd ..
    gcloud run deploy backend-app \
        --region=us-central1 \
        --image us-central1-docker.pkg.dev/$(gcloud config get-value project)/cloud-run-source-deploy/backend-app:latest \
        --allow-unauthenticated
elif [ "$1" == "-build" ]; then
    cd backend
    echo $(gcloud config get-value project)
    gcloud builds submit . \
        --config=cloudbuild.yml \
        --verbosity=debug
    cd ..
elif [ "$1" == "-deploy" ]; then
    # Get the first line of the full output and extract the digest
    LATEST_DIGEST=$(gcloud artifacts docker images list us-central1-docker.pkg.dev/$(gcloud config get-value project)/cloud-run-source-deploy/backend-app --limit=1 --sort-by="~createTime" | tail -n +2 | head -n 1 | awk '{print $2}')
    
    if [ -z "$LATEST_DIGEST" ]; then
        echo "No images found in registry. Please run -build first."
        exit 1
    fi
    
    echo "Deploying image with digest: $LATEST_DIGEST"
    gcloud run deploy backend-app \
        --region=us-central1 \
        --image us-central1-docker.pkg.dev/$(gcloud config get-value project)/cloud-run-source-deploy/backend-app@$LATEST_DIGEST \
        --allow-unauthenticated
else
    echo "Usage: ./run.sh [-f] [-g] [commit message]"
    echo "  -g: Commit and push changes to remote repository (optional commit message)"
    echo "  -f: Run frontend"
    echo "  -fire: Deploy backend to firebase"
    echo "  -b: Run backend"
    echo "  -lint: Run prettier on frontend"
    echo "  -deploy: Deploy backend to cloud run"
    echo "  -build: Build on cloud build"
    echo "  -build-deploy: Build and deploy backend to cloud run"
fi