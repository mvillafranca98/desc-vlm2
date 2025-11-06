#!/bin/bash

# Setup Langfuse Environment Variables
# Source this file to set the environment variables in your current shell:
# source setup_langfuse.sh

echo "Setting up Langfuse environment variables..."

export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-61ba616e-de46-4d72-a848-753fd9a5b3fb"
export LANGFUSE_SECRET_KEY="sk-lf-e8c7bb1b-554d-4396-95ad-95c30593d6c8"

echo "✓ Langfuse environment variables set!"
echo ""
echo "  LANGFUSE_HOST: $LANGFUSE_HOST"
echo "  LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "  LANGFUSE_SECRET_KEY: [hidden]"
echo ""
echo "Now you can run: python real_time_summarizer.py"

