# Fix Dimension Mismatch Issue

## Problem
Your `.env` file has:
- `SCENE_INDEX_TABLE=scene_embeddings` (384-dim table for MiniLM)
- `SCENE_INDEX_MODEL=BAAI/bge-base-en` (768-dim model)

This causes dimension mismatch errors.

## Solution

Edit your `.env` file and either:

**Option 1 (Recommended):** Remove or comment out `SCENE_INDEX_TABLE` to let it auto-generate:
```
# SCENE_INDEX_TABLE=scene_embeddings
SCENE_INDEX_MODEL=BAAI/bge-base-en
```

**Option 2:** Set it to the correct table name:
```
SCENE_INDEX_TABLE=scene_embeddings_bge
SCENE_INDEX_MODEL=BAAI/bge-base-en
```

## After Fixing

1. Run `./run.sh` to index new data to the correct table (`scene_embeddings_bge`)
2. Run `./run_amber.sh` - it will now use the correct table automatically

## Table Name Mapping

- `BAAI/bge-base-en` → `scene_embeddings_bge` (768 dim)
- `intfloat/e5-large-v2` → `scene_embeddings_e5` (1024 dim)
- `text-embedding-3-small` → `scene_embeddings_openai_small` (1536 dim)
- `all-MiniLM-L6-v2` → `scene_embeddings` (384 dim)
