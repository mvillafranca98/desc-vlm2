#!/bin/bash
# Setup script for LanceDB Cloud credentials

echo "=========================================="
echo "LanceDB Cloud Setup"
echo "=========================================="
echo ""
echo "To use LanceDB Cloud, you need:"
echo "1. Create a project at https://cloud.lancedb.com/dashboard/create_project"
echo "2. Get your Project Slug and API Key"
echo "3. Set the following environment variables:"
echo ""
echo "export LANCEDB_PROJECT_SLUG=\"your-project-slug\""
echo "export LANCEDB_API_KEY=\"your-api-key\""
echo "export LANCEDB_REGION=\"us-east-1\"  # Optional, defaults to us-east-1"
echo ""
echo "=========================================="
echo ""

# Check if credentials are already set
if [ -n "$LANCEDB_PROJECT_SLUG" ] && [ -n "$LANCEDB_API_KEY" ]; then
    echo "✅ LanceDB credentials found in environment"
    echo "   Project Slug: $LANCEDB_PROJECT_SLUG"
    echo "   Region: ${LANCEDB_REGION:-us-east-1}"
else
    echo "⚠️  LanceDB credentials not set"
    echo ""
    echo "For local development (without Cloud):"
    echo "  The system will automatically use a local database at: data/scenes"
    echo ""
    echo "To use LanceDB Cloud, set the environment variables above"
    echo "or add them to your shell profile (~/.zshrc or ~/.bashrc)"
fi

echo ""
echo "=========================================="

