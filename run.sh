#!/bin/bash

# Load environment variables from .env file if it exists
# Note: We'll allow SCENE_INDEX_MODEL to be overridden by user selection
ENV_SCENE_INDEX_MODEL=""
if [[ -f .env ]]; then
    echo "📄 Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop automatically exporting
    # Save the .env value but allow user to override
    ENV_SCENE_INDEX_MODEL="$SCENE_INDEX_MODEL"
    # Temporarily unset so menu shows
    unset SCENE_INDEX_MODEL
fi

echo "================================================"
echo "Real-Time Video Summarization System"
echo "================================================"
echo ""

# Set Langfuse environment variables (allow overrides, use .env if available)
export LANGFUSE_HOST="${LANGFUSE_HOST:-https://cloud.langfuse.com}"
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-pk-lf-9d51189b-1ff3-4a6c-a899-335651915d99}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-sk-lf-b9e367e9-8fd6-44ca-811f-13e7efe87a6e}"

# Set threading variables to prevent segfaults on macOS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# LanceDB credentials (allow overrides, use .env if available)
export LANCEDB_PROJECT_SLUG="${LANCEDB_PROJECT_SLUG:-descvlm2-lnh0lv}"
export LANCEDB_API_KEY="${LANCEDB_API_KEY:-sk_SJLWUL2G2JDQFLZ5WKKM5DC4KJKIVAV73IECRDHTRSBUAEMY2DSQ====}"
export LANCEDB_REGION="${LANCEDB_REGION:-us-east-1}"

# OpenAI API key (allow overrides, use .env if available)
export OPENAI_API_KEY="${OPENAI_API_KEY:-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}"

echo "✓ Langfuse credentials set"
echo "  Host: $LANGFUSE_HOST"
echo "  Public Key: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "✓ LanceDB credentials prepared"
echo "  Project Slug: $LANCEDB_PROJECT_SLUG"
echo "  Region: ${LANCEDB_REGION}"
echo "✓ Threading optimizations set (prevents segfaults)"
echo ""

# Embedding model selection
# Always show menu to let user choose which model to use for this run
echo "Select embedding model:"
if [[ -n "$ENV_SCENE_INDEX_MODEL" ]]; then
    echo "  (Note: .env has SCENE_INDEX_MODEL=$ENV_SCENE_INDEX_MODEL, but you can override it here)"
fi
echo "  1) OpenAI text-embedding-3-small (1536 dimensions, recommended)"
echo "  2) BAAI/bge-base-en (768 dimensions)"
echo "  3) intfloat/e5-large-v2 (1024 dimensions)"
echo "  4) sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)"
read -rp "Choice [1-4] (default: 1): " MODEL_CHOICE
    
case "$MODEL_CHOICE" in
    2)
        export SCENE_INDEX_MODEL="BAAI/bge-base-en"
        ;;
    3)
        export SCENE_INDEX_MODEL="intfloat/e5-large-v2"
        ;;
    4)
        export SCENE_INDEX_MODEL="sentence-transformers/all-MiniLM-L6-v2"
        ;;
    *)
        # Default to OpenAI text-embedding-3-small
        export SCENE_INDEX_MODEL="text-embedding-3-small"
        # Use default API key if set, otherwise prompt
        if [[ -z "$OPENAI_API_KEY" ]]; then
            echo ""
            echo "⚠️  OpenAI API key required"
            echo "Get your API key from: https://platform.openai.com/api-keys"
            read -rsp "Enter your OpenAI API key: " OPENAI_API_KEY
            echo ""
            export OPENAI_API_KEY
        fi
        echo "✓ Using OpenAI API key: ${OPENAI_API_KEY:0:20}..."
        ;;
esac

# Check for OpenAI API key if OpenAI model is selected
if [[ "$SCENE_INDEX_MODEL" == *"text-embedding"* ]] || [[ "$SCENE_INDEX_MODEL" == *"openai"* ]]; then
    if [[ -z "$OPENAI_API_KEY" ]]; then
        echo "⚠️  OpenAI API key required for $SCENE_INDEX_MODEL"
        echo "Get your API key from: https://platform.openai.com/api-keys"
        read -rsp "Enter your OpenAI API key: " OPENAI_API_KEY
        echo ""
        export OPENAI_API_KEY
    fi
    if [[ -n "$OPENAI_API_KEY" ]]; then
        echo "✓ Using OpenAI API key: ${OPENAI_API_KEY:0:20}..."
    fi
fi

# Derive LanceDB table name based on embedding model
derive_table_name() {
    local model="$1"
    local short="${model##*/}"
    short="${short//-/_}"
    # Use tr for lowercase conversion (more compatible than ${var,,})
    local lower_short=$(echo "$short" | tr '[:upper:]' '[:lower:]')
    local lower_model=$(echo "$model" | tr '[:upper:]' '[:lower:]')

    if [[ "$lower_short" == *"minilm"* ]]; then
        echo "scene_embeddings"
    elif [[ "$lower_short" == *"e5"* ]]; then
        echo "scene_embeddings_e5"
    elif [[ "$lower_short" == *"bge"* ]]; then
        echo "scene_embeddings_bge"
    elif [[ "$lower_model" == *"text-embedding-3-small"* ]] || [[ "$lower_model" == *"text-embedding-3"* ]]; then
        echo "scene_embeddings_openai_small"
    elif [[ "$lower_model" == *"openai"* ]]; then
        echo "scene_embeddings_openai"
    else
        echo "scene_embeddings_${short}"
    fi
}

# Always auto-generate table name based on selected model to ensure dimension match
EXPECTED_TABLE="$(derive_table_name "$SCENE_INDEX_MODEL")"

# Check if .env has a mismatched table name and warn/override
if [[ -n "$SCENE_INDEX_TABLE" ]] && [[ "$SCENE_INDEX_TABLE" != "$EXPECTED_TABLE" ]]; then
    echo "⚠️  Warning: .env has SCENE_INDEX_TABLE=$SCENE_INDEX_TABLE"
    echo "   But selected model $SCENE_INDEX_MODEL requires table: $EXPECTED_TABLE"
    echo "   Auto-correcting to use: $EXPECTED_TABLE"
    echo ""
fi

# Always use the model-specific table name (override .env if needed)
export SCENE_INDEX_TABLE="$EXPECTED_TABLE"
echo "✓ LanceDB table (auto-generated for $SCENE_INDEX_MODEL): $SCENE_INDEX_TABLE"

echo "✓ Embedding model: $SCENE_INDEX_MODEL"
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

# Run the application with embedding model and table args
echo "Starting application..."
echo "Press ESC or Q to exit"
echo ""
echo "📊 Configuration:"
echo "   Embedding Model: $SCENE_INDEX_MODEL"
echo "   LanceDB Table: $SCENE_INDEX_TABLE"
echo ""
python real_time_summarizer.py \
    --embedding-model "$SCENE_INDEX_MODEL" \
    --scene-table "$SCENE_INDEX_TABLE" \
    "$@"
