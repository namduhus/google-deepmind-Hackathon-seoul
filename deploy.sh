#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="sim-to-action-vla"
REGION="${REGION:-us-central1}"

echo "========================================================"
echo "🚀 Google Cloud Run Deployment Script (${SERVICE_NAME})"
echo "========================================================"

# 1. Check gcloud CLI installation
if ! command -v gcloud &> /dev/null; then
  echo "Installing Google Cloud SDK via Homebrew..."
  brew install --cask google-cloud-sdk
fi

# 2. Load GEMINI_API_KEY and GEMINI_MODEL from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "Error: GEMINI_API_KEY is not set. Please check your .env file."
  exit 1
fi

# 3. Verify active GCP Project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
  echo "No active Google Cloud Project configured."
  echo "Please authenticate and set your project:"
  echo "  gcloud auth login"
  echo "  gcloud config set project YOUR_GCP_PROJECT_ID"
  exit 1
fi

echo "Target GCP Project: ${PROJECT_ID}"
echo "Target Region:      ${REGION}"

# 4. Enable required Google Cloud APIs
echo "Enabling Cloud Run, Cloud Build, and Artifact Registry APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"

# 5. Build and deploy to Google Cloud Run from source
echo "Building and deploying container to Google Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --port 8080 \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-pro}"

echo "========================================================"
echo "Deployment complete! Open the Service URL above."
echo "========================================================"
