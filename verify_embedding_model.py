#!/usr/bin/env python3
"""
Verify embedding model configuration and table status.
"""

import os
import sys
from scene_indexer import SceneIndexer, get_table_name_for_model
import lancedb

def verify_embedding_model():
    """Verify embedding model configuration and table status."""
    print("=" * 70)
    print("Embedding Model & Table Verification")
    print("=" * 70)
    print()
    
    # Load .env if it exists
    if os.path.exists('.env'):
        print("📄 Loading .env file...")
        with open('.env') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print()
    
    # Get configuration
    model = os.getenv("SCENE_INDEX_MODEL", "text-embedding-3-small")
    table_name = get_table_name_for_model(model)
    
    print("📊 Configuration:")
    print(f"   Model: {model}")
    print(f"   Expected Table: {table_name}")
    print()
    
    # Connect to LanceDB
    try:
        project_slug = os.getenv("LANCEDB_PROJECT_SLUG", "descvlm2-lnh0lv")
        api_key = os.getenv("LANCEDB_API_KEY")
        region = os.getenv("LANCEDB_REGION", "us-east-1")
        
        if not api_key:
            print("⚠️  LANCEDB_API_KEY not set")
            return
        
        print("🔗 Connecting to LanceDB...")
        db = lancedb.connect(
            uri=f"db://{project_slug}",
            api_key=api_key,
            region=region
        )
        print(f"✓ Connected to project: {project_slug}")
        print()
        
        # Check all possible tables
        print("📋 Checking tables:")
        print()
        
        tables_to_check = [
            ("scene_embeddings", "MiniLM (384 dim) or default"),
            ("scene_embeddings_e5", "e5-large-v2 (1024 dim)"),
            ("scene_embeddings_bge", "BAAI/bge-base-en (768 dim)"),
            ("scene_embeddings_openai_small", "OpenAI text-embedding-3-small (1536 dim)"),
            ("scene_index", "Generic scene index"),
        ]
        
        found_tables = []
        for table_name_check, description in tables_to_check:
            try:
                table = db.open_table(table_name_check)
                count = table.count_rows()
                schema = table.schema
                
                # Try to determine vector dimension from schema
                vector_dim = "unknown"
                for field in schema:
                    if field.name == "vector":
                        if hasattr(field.type, 'list_size'):
                            vector_dim = field.type.list_size
                        break
                
                status = "✓ EXISTS"
                if count == 0:
                    status = "✓ EXISTS (empty)"
                
                print(f"   {status:12} {table_name_check:30} - {description}")
                print(f"              Records: {count}, Vector dim: {vector_dim}")
                found_tables.append((table_name_check, count, vector_dim))
            except Exception as e:
                if "not found" in str(e).lower():
                    print(f"   ✗ NOT FOUND  {table_name_check:30} - {description}")
                else:
                    print(f"   ⚠️  ERROR     {table_name_check:30} - {str(e)[:50]}")
            print()
        
        # Check if expected table exists
        print("=" * 70)
        expected_exists = any(t[0] == table_name for t in found_tables)
        
        if expected_exists:
            table_info = next((t for t in found_tables if t[0] == table_name), None)
            if table_info:
                count, dim = table_info[1], table_info[2]
                print(f"✅ Expected table '{table_name}' exists!")
                print(f"   Records: {count}")
                print(f"   Vector dimension: {dim}")
                if count == 0:
                    print()
                    print("⚠️  Table is empty. Run ./run.sh to index scenes.")
        else:
            print(f"⚠️  Expected table '{table_name}' does not exist yet.")
            print()
            print("To create the table:")
            print(f"   1. Make sure SCENE_INDEX_MODEL={model} is set in .env")
            print(f"   2. Run: ./run.sh")
            print(f"   3. The table will be created automatically when first scene is indexed")
        
        print()
        print("=" * 70)
        print("Model-Specific Table Mapping:")
        print("=" * 70)
        models = [
            "BAAI/bge-base-en",
            "intfloat/e5-large-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "text-embedding-3-small",
        ]
        for m in models:
            t = get_table_name_for_model(m)
            print(f"   {m:40} -> {t}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_embedding_model()


