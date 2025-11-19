#!/bin/bash

# Load environment variables from .env file if it exists
ENV_SCENE_INDEX_TABLE=""
if [[ -f .env ]]; then
    echo "📄 Loading environment variables from .env file..."
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop automatically exporting
    # Save the .env table value to check for mismatches
    ENV_SCENE_INDEX_TABLE="$SCENE_INDEX_TABLE"
fi

echo "================================================"
echo "Amber - Video Scene Search Assistant"
echo "================================================"
echo ""

# Set LanceDB environment variables (allow overrides, use .env if available)
# NOTE: API keys should be set in .env file, not hardcoded here
export LANCEDB_PROJECT_SLUG="${LANCEDB_PROJECT_SLUG:-descvlm2-lnh0lv}"
if [[ -z "$LANCEDB_API_KEY" ]]; then
    echo "⚠️  Warning: LANCEDB_API_KEY not set. Please set it in .env file."
fi
export LANCEDB_REGION="${LANCEDB_REGION:-us-east-1}"

# Get embedding model (default to BAAI/bge-base-en if not set)
export SCENE_INDEX_MODEL="${SCENE_INDEX_MODEL:-BAAI/bge-base-en}"

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

# Always auto-generate table name based on model to ensure dimension match
EXPECTED_TABLE="$(derive_table_name "$SCENE_INDEX_MODEL")"

# Check if .env has a mismatched table name and warn/override
if [[ -n "$ENV_SCENE_INDEX_TABLE" ]] && [[ "$ENV_SCENE_INDEX_TABLE" != "$EXPECTED_TABLE" ]]; then
    echo "⚠️  Warning: .env has SCENE_INDEX_TABLE=$ENV_SCENE_INDEX_TABLE"
    echo "   But model $SCENE_INDEX_MODEL requires table: $EXPECTED_TABLE"
    echo "   Auto-correcting to use: $EXPECTED_TABLE"
    echo ""
fi

# Always use the model-specific table name (override .env if needed)
export SCENE_INDEX_TABLE="$EXPECTED_TABLE"

# Set Langfuse environment variables (allow overrides, use .env if available)
# NOTE: API keys should be set in .env file, not hardcoded here
export LANGFUSE_HOST="${LANGFUSE_HOST:-https://cloud.langfuse.com}"
if [[ -z "$LANGFUSE_PUBLIC_KEY" ]]; then
    echo "⚠️  Warning: LANGFUSE_PUBLIC_KEY not set. Please set it in .env file."
fi
if [[ -z "$LANGFUSE_SECRET_KEY" ]]; then
    echo "⚠️  Warning: LANGFUSE_SECRET_KEY not set. Please set it in .env file."
fi

echo "✓ LanceDB Integration:"
echo "  Project Slug: $LANCEDB_PROJECT_SLUG"
echo "  Region: ${LANCEDB_REGION}"
echo "  Table: $SCENE_INDEX_TABLE"
echo "  Connection: LanceDB Cloud"
echo "  Vector Search: Enabled (cosine similarity)"
echo "  Embedding Model: $SCENE_INDEX_MODEL"
echo ""

echo "✓ Langfuse Integration:"
echo "  Host: $LANGFUSE_HOST"
echo "  Public Key: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "  Status: Configured for telemetry"
echo "  Data Export: Traces, spans, generations, TTFT, tool calls, scores"
echo "  Note: Used in real_time_summarizer.py for observability"
echo ""

echo "✓ Models Configuration:"
echo "  VLM Model: ${VLM_MODEL:-qwen3}"
echo "  LLM Model: ${LLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
echo "  Embedding Model: $SCENE_INDEX_MODEL"
echo "  Note: Models are loaded when running real_time_summarizer.py"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Quick verification of LanceDB connection
echo "🔍 Verifying LanceDB connection..."
python << PYTHON_VERIFY
import os
import sys
try:
    import lancedb
    project_slug = os.getenv("LANCEDB_PROJECT_SLUG")
    api_key = os.getenv("LANCEDB_API_KEY")
    region = os.getenv("LANCEDB_REGION", "us-east-1")
    table_name = os.getenv("SCENE_INDEX_TABLE") or "scene_embeddings"
    embedding_model = os.getenv("SCENE_INDEX_MODEL", "BAAI/bge-base-en")
    
    # Ensure table_name is not empty
    if not table_name or table_name.strip() == "":
        print(f"  ⚠️  Table name is empty, using default")
        table_name = "scene_embeddings"
    
    if project_slug and api_key:
        db = lancedb.connect(
            uri=f"db://{project_slug}",
            api_key=api_key,
            region=region
        )
        try:
            table = db.open_table(table_name)
            row_count = table.count_rows()
            print(f"  ✅ LanceDB connection verified")
            print(f"  📊 Table '{table_name}' has {row_count} records")
            print(f"  🤖 Using embedding model: {embedding_model}")
        except Exception as table_error:
            print(f"  ⚠️  Table '{table_name}' not found or inaccessible")
            print(f"  ℹ️  This is normal if you haven't indexed data with {embedding_model} yet")
            print(f"  ℹ️  The table will be created automatically when you run ./run.sh")
    else:
        print(f"  ⚠️  LanceDB credentials not fully set")
except Exception as e:
    print(f"  ⚠️  Could not verify LanceDB: {str(e)[:80]}")
PYTHON_VERIFY

echo ""
echo "Starting Amber Assistant..."
echo ""

# Run Amber
python amber_assistant.py

