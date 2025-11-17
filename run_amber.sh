#!/bin/bash

echo "================================================"
echo "Amber - Video Scene Search Assistant"
echo "================================================"
echo ""

# Set LanceDB environment variables (allow overrides)
export LANCEDB_PROJECT_SLUG="${LANCEDB_PROJECT_SLUG:-descvlm2-lnh0lv}"
export LANCEDB_API_KEY="${LANCEDB_API_KEY:-sk_SJLWUL2G2JDQFLZ5WKKM5DC4KJKIVAV73IECRDHTRSBUAEMY2DSQ====}"
export LANCEDB_REGION="${LANCEDB_REGION:-us-east-1}"
export SCENE_INDEX_TABLE="${SCENE_INDEX_TABLE:-scene_embeddings}"

# Set Langfuse environment variables (if not already set)
export LANGFUSE_HOST="${LANGFUSE_HOST:-https://cloud.langfuse.com}"
export LANGFUSE_PUBLIC_KEY="${LANGFUSE_PUBLIC_KEY:-pk-lf-9d51189b-1ff3-4a6c-a899-335651915d99}"
export LANGFUSE_SECRET_KEY="${LANGFUSE_SECRET_KEY:-sk-lf-b9e367e9-8fd6-44ca-811f-13e7efe87a6e}"

echo "✓ LanceDB Integration:"
echo "  Project Slug: $LANCEDB_PROJECT_SLUG"
echo "  Region: ${LANCEDB_REGION}"
echo "  Table: $SCENE_INDEX_TABLE"
echo "  Connection: LanceDB Cloud"
echo "  Vector Search: Enabled (cosine similarity)"
echo "  Embedding Model: intfloat/e5-large-v2"
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
echo "  Embedding Model: intfloat/e5-large-v2"
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
python << 'PYTHON_VERIFY'
import os
import sys
try:
    import lancedb
    project_slug = os.getenv("LANCEDB_PROJECT_SLUG")
    api_key = os.getenv("LANCEDB_API_KEY")
    region = os.getenv("LANCEDB_REGION", "us-east-1")
    table_name = os.getenv("SCENE_INDEX_TABLE", "scene_embeddings")
    
    if project_slug and api_key:
        db = lancedb.connect(
            uri=f"db://{project_slug}",
            api_key=api_key,
            region=region
        )
        table = db.open_table(table_name)
        row_count = table.count_rows()
        print(f"  ✅ LanceDB connection verified")
        print(f"  📊 Table '{table_name}' has {row_count} records")
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

