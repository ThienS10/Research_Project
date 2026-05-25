#!/usr/bin/env bash
set -euo pipefail

# Build a lightweight Lambda deployment bundle containing only the Bedrock
# classifier and minimal handler. The output directory defaults to
# dist/lambda-minimal and a matching ZIP is created for upload.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/dist/lambda-minimal"}"
ZIP_PATH="${OUT_DIR}.zip"

echo "Building minimal Lambda bundle in: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Copy handler
cp "$ROOT_DIR/lambda_handler.py" "$OUT_DIR/"

# Copy only the lightweight Bedrock client artefacts (avoid rsync for portability)
cp -R "$ROOT_DIR/aws_bedrock_agent" "$OUT_DIR/"
# Strip caches and unused files
find "$OUT_DIR/aws_bedrock_agent" -name "__pycache__" -prune -exec rm -rf {} +
find "$OUT_DIR/aws_bedrock_agent" -name "*.pyc" -delete
find "$OUT_DIR/aws_bedrock_agent" -name ".aws-sam" -prune -exec rm -rf {} +

# Remove optional heavy assets not needed for classification
rm -f "$OUT_DIR/aws_bedrock_agent"/{data_pipeline.py,pipeline_runner.py,app.py,agent_invoke.py,agentcore_setup.py,requirements.txt,template.yaml}

# Zip the bundle with files at the root (Lambda expects handler at /var/task)
rm -f "$ZIP_PATH"
(
  cd "$OUT_DIR"
  zip -r "$ZIP_PATH" . >/dev/null
)

echo "Bundle ready: $ZIP_PATH"
echo "Size:"
du -h "$ZIP_PATH"
