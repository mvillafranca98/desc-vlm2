#!/bin/bash

echo "================================================"
echo "Real-Time Video Summarization System"
echo "================================================"
echo ""

# Set Langfuse environment variables (allow overrides)
export LANGFUSE_HOST="${LANGFUSE_HOST:-https://cloud.langfuse.com}"
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-pk-lf-9d51189b-1ff3-4a6c-a899-335651915d99}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-sk-lf-b9e367e9-8fd6-44ca-811f-13e7efe87a6e}"

# Set threading variables to prevent segfaults on macOS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# LanceDB credentials (allow overrides, provide defaults)
export LANCEDB_PROJECT_SLUG="${LANCEDB_PROJECT_SLUG:-descvlm2-lnh0lv}"
export LANCEDB_API_KEY="${LANCEDB_API_KEY:-sk_SJLWUL2G2JDQFLZ5WKKM5DC4KJKIVAV73IECRDHTRSBUAEMY2DSQ====}"
export LANCEDB_REGION="${LANCEDB_REGION:-us-east-1}"
export SCENE_INDEX_TABLE="${SCENE_INDEX_TABLE:-scene_embeddings}"

echo "✓ Langfuse credentials set"
echo "  Host: $LANGFUSE_HOST"
echo "  Public Key: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "✓ LanceDB credentials prepared"
echo "  Project Slug: $LANCEDB_PROJECT_SLUG"
echo "  Region: ${LANCEDB_REGION}"
echo "✓ Threading optimizations set (prevents segfaults)"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Determine camera selection if not provided via arguments
CAMERA_ARG_PROVIDED=false
for arg in "$@"; do
    if [[ "$arg" == "--camera" ]]; then
        CAMERA_ARG_PROVIDED=true
        break
    fi
done

if [[ "$CAMERA_ARG_PROVIDED" == false ]]; then
    echo "Select camera source:"
    echo "  1) Default system camera (index 0)"
    echo "  2) OBS Virtual Camera (index 1)"
    echo "  3) Enter custom camera index"
    read -rp "Choice [1-3] (default: 1): " CAMERA_CHOICE

    case "$CAMERA_CHOICE" in
        2)
            CAMERA_INDEX=1
            ;;
        3)
            read -rp "Enter camera index (integer): " CAMERA_INDEX
            if ! [[ "$CAMERA_INDEX" =~ ^-?[0-9]+$ ]]; then
                echo "Invalid input. Falling back to default camera (index 0)."
                CAMERA_INDEX=0
            fi
            ;;
        *)
            CAMERA_INDEX=0
            ;;
    esac

    set -- "$@" --camera "$CAMERA_INDEX"
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Run the application
echo "Starting application..."
echo "Press ESC or Q to exit"
echo ""
python real_time_summarizer.py "$@"

