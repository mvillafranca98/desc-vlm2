#!/bin/bash

echo "================================================"
echo "Real-Time Video Summarization System"
echo "================================================"
echo ""

# Set Langfuse environment variables
export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-61ba616e-de46-4d72-a848-753fd9a5b3fb"
export LANGFUSE_SECRET_KEY="sk-lf-e8c7bb1b-554d-4396-95ad-95c30593d6c8"

# Set threading variables to prevent segfaults on macOS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "✓ Langfuse credentials set"
echo "  Host: $LANGFUSE_HOST"
echo "  Public Key: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "✓ Threading optimizations set (prevents segfaults)"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Run the application
echo "Starting application..."
echo "Press ESC or Q to exit"
echo ""
python real_time_summarizer.py "$@"

