#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

SERVICE_NAME="sim-to-action-vla"
PROJECT_ID="${PROJECT_ID:-h2gdm-hack26icn-2645}"
REGION="${REGION:-us-central1}"

echo "========================================================"
echo "Google Cloud Run Deployment (${SERVICE_NAME})"
echo "Project: ${PROJECT_ID} | Region: ${REGION}"
echo "========================================================"

# Load GEMINI_API_KEY and GEMINI_MODEL from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

gcloud config set project "${PROJECT_ID}"

echo "Enabling required Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"

echo "Building and deploying container to Google Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --port 8080 \
  --quiet \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY:-},GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-pro}"

echo "========================================================"
echo "Deployment complete!"
echo "========================================================"
