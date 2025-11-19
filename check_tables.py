#!/usr/bin/env python3
"""
Check LanceDB tables and their status.
"""

import os
import sys
from scene_indexer import SceneIndexer, get_table_name_for_model

# Set up environment
os.environ.setdefault('LANCEDB_PROJECT_SLUG', 'descvlm2-lnh0lv')
os.environ.setdefault('LANCEDB_API_KEY', 'sk_SJLWUL2G2JDQFLZ5WKKM5DC4KJKIVAV73IECRDHTRSBUAEMY2DSQ====')
os.environ.setdefault('LANCEDB_REGION', 'us-east-1')

def check_table_status():
    """Check status of different embedding model tables."""
    print("=" * 60)
    print("LanceDB Table Status Check")
    print("=" * 60)
    print()
    
    models_to_check = [
        ("BAAI/bge-base-en", "Current model (768 dimensions)"),
        ("intfloat/e5-large-v2", "Previous model (1024 dimensions)"),
        ("sentence-transformers/all-MiniLM-L6-v2", "Small model (384 dimensions)"),
    ]
    
    # Also check the old default table
    old_table = "scene_embeddings"
    
    print("Checking tables for different embedding models...")
    print()
    
    for model_name, description in models_to_check:
        table_name = get_table_name_for_model(model_name)
        print(f"Model: {model_name}")
        print(f"  Description: {description}")
        print(f"  Expected table: {table_name}")
        
        try:
            # Try to open the table
            indexer = SceneIndexer(
                table_name=table_name,
                embedding_model=model_name
            )
            
            if indexer.enabled and indexer.table:
                # Try to get count
                try:
                    # Try to get a sample to check if table has data
                    sample = indexer.table.head(1)
                    if sample:
                        count = len(indexer.table.to_pandas()) if hasattr(indexer.table, 'to_pandas') else "unknown"
                        print(f"  Status: ✓ Table exists")
                        print(f"  Records: {count} (approximate)")
                    else:
                        print(f"  Status: ✓ Table exists (empty)")
                except Exception as e:
                    print(f"  Status: ✓ Table exists (error checking count: {e})")
            else:
                print(f"  Status: ✗ Table not accessible or indexer disabled")
        except Exception as e:
            print(f"  Status: ✗ Error: {e}")
        
        print()
    
    # Check old default table
    print(f"Old default table: {old_table}")
    try:
        indexer = SceneIndexer(
            table_name=old_table,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Most likely old model
        )
        if indexer.enabled and indexer.table:
            print(f"  Status: ✓ Table exists")
            print(f"  Note: This table likely has 384-dim vectors from previous indexing")
        else:
            print(f"  Status: ✗ Table not accessible")
    except Exception as e:
        print(f"  Status: ✗ Error: {e}")
    
    print()
    print("=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print()
    print("1. To test BAAI/bge-base-en model:")
    print("   - Run: ./run.sh")
    print("   - This will index new scenes to: scene_embeddings_768_bge_base_en")
    print()
    print("2. To test with existing data:")
    print("   - Set SCENE_INDEX_TABLE=scene_embeddings")
    print("   - Set SCENE_INDEX_MODEL=sentence-transformers/all-MiniLM-L6-v2")
    print("   - Then run: python test_semantic_queries.py")
    print()
    print("3. To compare models:")
    print("   - Index scenes with each model separately")
    print("   - Each model uses its own table automatically")
    print("=" * 60)

if __name__ == "__main__":
    check_table_status()


