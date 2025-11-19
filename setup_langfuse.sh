#!/bin/bash

# Setup Langfuse Environment Variables
# Source this file to set the environment variables in your current shell:
# source setup_langfuse.sh

# Load environment variables from .env file if it exists
if [[ -f .env ]]; then
    echo "📄 Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop automatically exporting
fi

echo "Setting up Langfuse environment variables..."

# Use .env values if available, otherwise use defaults
# NOTE: API keys should be set in .env file, not hardcoded here
export LANGFUSE_HOST="${LANGFUSE_HOST:-https://cloud.langfuse.com}"
if [[ -z "$LANGFUSE_PUBLIC_KEY" ]]; then
    echo "⚠️  Warning: LANGFUSE_PUBLIC_KEY not set. Please set it in .env file."
fi
if [[ -z "$LANGFUSE_SECRET_KEY" ]]; then
    echo "⚠️  Warning: LANGFUSE_SECRET_KEY not set. Please set it in .env file."
fi

echo "✓ Langfuse environment variables set!"
echo ""
echo "  LANGFUSE_HOST: $LANGFUSE_HOST"
echo "  LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "  LANGFUSE_SECRET_KEY: [hidden]"
echo ""
echo "Now you can run: python real_time_summarizer.py"

