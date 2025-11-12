#!/bin/bash

# Setup Langfuse Environment Variables
# Source this file to set the environment variables in your current shell:
# source setup_langfuse.sh

echo "Setting up Langfuse environment variables..."

export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-9d51189b-1ff3-4a6c-a899-335651915d99"
export LANGFUSE_SECRET_KEY="sk-lf-b9e367e9-8fd6-44ca-811f-13e7efe87a6e"

echo "✓ Langfuse environment variables set!"
echo ""
echo "  LANGFUSE_HOST: $LANGFUSE_HOST"
echo "  LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "  LANGFUSE_SECRET_KEY: [hidden]"
echo ""
echo "Now you can run: python real_time_summarizer.py"

