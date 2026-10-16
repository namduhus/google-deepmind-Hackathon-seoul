#!/usr/bin/env bash
set -euo pipefail

# Ensure standard paths are available
export PATH="/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

SERVICE_NAME="${SERVICE_NAME:-sim-to-action-vla}"
REGION="${REGION:-us-central1}"

# Load environment variables from .env if present
if [ -f .env ]; then
  set -a
  eval "$(grep -E -v '^(#|[[:space:]]*$)' .env)" 2>/dev/null || true
  set +a
fi

# Detect PROJECT_ID from env, gcloud config, or default
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo "h2gdm-hack26icn-2645")}"

echo "========================================================"
echo "Google Cloud Run Deployment: ${SERVICE_NAME}"
echo "Project: ${PROJECT_ID} | Region: ${REGION}"
echo "========================================================"

gcloud config set project "${PROJECT_ID}" --quiet

echo "1. Enabling required Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

echo "2. Configuring Cloud Build storage permissions (if required)..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)" 2>/dev/null || echo "")
if [ -n "${PROJECT_NUMBER}" ]; then
  COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
  gcloud storage buckets add-iam-policy-binding "gs://run-sources-${PROJECT_ID}-${REGION}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/storage.admin" \
    --quiet >/dev/null 2>&1 || true
fi

# Detect API key from either GEMINI_API_KEY or GOOGLE_API_KEY
API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"

echo "3. Building and deploying container to Google Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --port 8080 \
  --quiet \
  --set-env-vars="GEMINI_API_KEY=${API_KEY},GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash}"

echo "========================================================"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format="value(status.url)" 2>/dev/null || echo "")
echo "Deployment successfully finished!"
if [ -n "${SERVICE_URL}" ]; then
  echo "Live Service URL: ${SERVICE_URL}"
fi
echo "========================================================"
