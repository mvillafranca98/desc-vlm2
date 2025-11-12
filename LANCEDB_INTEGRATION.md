# LanceDB Integration Guide

Complete guide for integrating LanceDB semantic search into the Real-Time Video Summarization System.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Architecture](#architecture)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

## Overview

LanceDB integration enables semantic search over video scene summaries. When the LLM generates a summary, the system:

1. **Generates embeddings** using sentence-transformers
2. **Saves associated frames** to disk
3. **Indexes the scene** in LanceDB with the embedding, summary text, and frame paths
4. **Enables natural language search** to find scenes by description

### Features

- ✅ Automatic scene indexing after each summary generation
- ✅ Vector embeddings for semantic search
- ✅ Frame storage with scene association
- ✅ Natural language query interface
- ✅ Support for LanceDB Cloud and local databases

## Setup

### 1. Install Dependencies

```bash
pip install lancedb sentence-transformers pandas pyarrow
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure LanceDB

**Option A: LanceDB Cloud (Recommended for Production)**

1. Create a project at https://cloud.lancedb.com/dashboard/create_project
2. Get your Project Slug and API Key
3. Set environment variables:

```bash
export LANCEDB_PROJECT_SLUG="your-project-slug"
export LANCEDB_API_KEY="your-api-key"
export LANCEDB_REGION="us-east-1"  # Optional, defaults to us-east-1
```

**Option B: Local Database (Development)**

No configuration needed! The system will automatically use a local database at `data/scenes/` if Cloud credentials are not set.

### 3. Verify Setup

Run the main application:
```bash
./run.sh
```

Look for:
```
Scene indexer initialized successfully
Connected to LanceDB Cloud: your-project-slug
# OR
Using local LanceDB at: data/scenes
```

## Architecture

### File Structure

```
scene_indexer.py          # LanceDB integration module
├── SceneIndexer          # Main indexer class
│   ├── index_scene()     # Index a scene with summary and frames
│   ├── search_scenes()   # Search scenes by natural language
│   ├── save_frame()      # Save frame to disk
│   └── generate_embedding() # Generate vector embedding
└── Integration           # Automatic indexing in real_time_summarizer.py

search_scenes.py          # CLI tool for searching scenes
frames/                   # Directory storing frame images (auto-created)
data/scenes/              # Local LanceDB database (if not using Cloud)
```

### Data Flow

```
1. Video Capture
   ↓
2. VLM Caption Generation (every 5 seconds)
   ↓
3. Frame Storage (frames stored in memory)
   ↓
4. LLM Summary Generation (every 60 seconds)
   ↓
5. Frame Saving (frames saved to disk)
   ↓
6. Embedding Generation (summary → vector)
   ↓
7. LanceDB Indexing (vector + metadata stored)
   ↓
8. Ready for Semantic Search
```

### Data Schema

Each scene record in LanceDB contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique scene identifier |
| `vector` | list[float] | Embedding vector (384 dimensions for all-MiniLM-L6-v2) |
| `summary_text` | string | LLM-generated summary text |
| `frame_paths` | string | Comma-separated list of frame file paths |
| `scene_start_time` | float | Unix timestamp when scene started |
| `timestamp` | float | Unix timestamp when scene was indexed |

## Usage

### Automatic Indexing

Scenes are automatically indexed when summaries are generated. No manual intervention needed!

**When indexing happens:**
- After each LLM summary generation (every 60 seconds)
- Only if scene indexer is enabled
- Only if frames were captured during the scene period

**What gets indexed:**
- Summary text from LLM
- Frames captured during the scene (up to 60 frames)
- Timestamp and scene metadata

### Searching Scenes

Use the `search_scenes.py` script to search indexed scenes:

**Basic Search:**
```bash
python search_scenes.py "a man wearing a hat"
```

**Advanced Options:**
```bash
# Limit results
python search_scenes.py "person sitting at desk" --limit 10

# Minimum similarity score
python search_scenes.py "outdoor scene" --min-score 0.5

# List all scenes
python search_scenes.py --list

# List with limit
python search_scenes.py --list --limit 20
```

### Programmatic Usage

```python
from scene_indexer import SceneIndexer

# Initialize
indexer = SceneIndexer()

# Search
results = indexer.search_scenes("a man wearing a hat", limit=5)

for scene in results:
    print(f"Scene: {scene['summary_text']}")
    print(f"Frames: {scene['frame_paths']}")
    print(f"Similarity: {scene['similarity']}")
```

## API Reference

### SceneIndexer Class

#### `__init__(table_name, embedding_model, frames_dir)`

Initialize the scene indexer.

**Parameters:**
- `table_name` (str): Name of LanceDB table (default: "scene_embeddings")
- `embedding_model` (str): Sentence transformer model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `frames_dir` (str): Directory for storing frames (default: "frames")

**Example:**
```python
indexer = SceneIndexer(
    table_name="my_scenes",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    frames_dir="my_frames"
)
```

#### `index_scene(summary_text, frame_paths, scene_start_time, scene_id)`

Index a scene with its summary and associated frames.

**Parameters:**
- `summary_text` (str): LLM-generated summary text
- `frame_paths` (List[str]): List of file paths to frames
- `scene_start_time` (float, optional): Timestamp when scene started
- `scene_id` (str, optional): Unique scene ID (auto-generated if not provided)

**Returns:**
- `str`: Scene ID if successful, `None` otherwise

**Example:**
```python
scene_id = indexer.index_scene(
    summary_text="A person sitting at a desk working on a computer",
    frame_paths=["frames/scene_123_frame_0001.jpg", "frames/scene_123_frame_0002.jpg"],
    scene_start_time=time.time()
)
```

#### `search_scenes(query_text, limit, min_score)`

Search for scenes using natural language query.

**Parameters:**
- `query_text` (str): Natural language search query
- `limit` (int): Maximum number of results (default: 5)
- `min_score` (float): Minimum similarity score 0.0-1.0 (default: 0.0)

**Returns:**
- `List[Dict]`: List of matching scenes with metadata

**Example:**
```python
results = indexer.search_scenes(
    query_text="a man wearing a hat",
    limit=10,
    min_score=0.3
)
```

#### `save_frame(frame, scene_id, frame_index)`

Save a frame to disk.

**Parameters:**
- `frame` (np.ndarray): Frame image as numpy array
- `scene_id` (str): Scene identifier
- `frame_index` (int): Frame index within scene

**Returns:**
- `str`: File path to saved frame, empty string on error

#### `generate_embedding(text)`

Generate embedding vector for text.

**Parameters:**
- `text` (str): Text to embed

**Returns:**
- `np.ndarray`: Embedding vector

## Integration Details

### Frame Storage

Frames are stored in the `frames/` directory with naming pattern:
```
{scene_id}_frame_{frame_index:04d}.jpg
```

Example:
```
frames/scene_1730745600_frame_0001.jpg
frames/scene_1730745600_frame_0002.jpg
```

### Embedding Model

**Default Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Speed:** Fast
- **Quality:** Good for general semantic search

**Alternative Models:**
You can use other sentence-transformer models by changing the `embedding_model` parameter:

```python
# Higher quality, slower
indexer = SceneIndexer(embedding_model="sentence-transformers/all-mpnet-base-v2")

# Multilingual support
indexer = SceneIndexer(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

### Scene Period

A "scene" is defined as the period between summary generations:
- **Duration:** 60 seconds (configurable via `summary_interval`)
- **Frames:** Up to 60 frames stored (one every 5 seconds)
- **Summary:** Single LLM summary describing the entire scene period

## Troubleshooting

### Issue: Scene Indexer Not Enabled

**Symptom:** `Scene indexer is not enabled`

**Causes:**
1. LanceDB not installed
2. sentence-transformers not installed
3. Initialization error

**Solutions:**
```bash
# Install dependencies
pip install lancedb sentence-transformers pandas pyarrow

# Check logs for initialization errors
```

### Issue: No Scenes Found

**Symptom:** Search returns no results

**Causes:**
1. No scenes indexed yet
2. Query doesn't match any scenes
3. Min score threshold too high

**Solutions:**
1. **Check if scenes are indexed:**
   ```bash
   python search_scenes.py --list
   ```

2. **Lower min score:**
   ```bash
   python search_scenes.py "your query" --min-score 0.0
   ```

3. **Wait for scenes to be indexed** (summaries generated every 60 seconds)

### Issue: Frames Not Saving

**Symptom:** Frame paths are empty in search results

**Causes:**
1. Frames directory not writable
2. Disk space full
3. OpenCV not installed

**Solutions:**
```bash
# Check directory permissions
ls -la frames/

# Check disk space
df -h

# Verify OpenCV
python -c "import cv2; print('OK')"
```

### Issue: LanceDB Connection Failed

**Symptom:** `Failed to initialize scene indexer`

**Causes:**
1. Invalid Cloud credentials
2. Network issues
3. Local database path issues

**Solutions:**

**For Cloud:**
```bash
# Verify credentials
echo $LANCEDB_PROJECT_SLUG
echo $LANCEDB_API_KEY

# Test connection
python -c "import lancedb; db = lancedb.connect(uri='db://your-slug', api_key='your-key'); print('OK')"
```

**For Local:**
```bash
# Check directory permissions
mkdir -p data/scenes
chmod 755 data/scenes
```

### Issue: Embedding Model Download Fails

**Symptom:** `Failed to load embedding model`

**Causes:**
1. Network issues
2. Hugging Face authentication required
3. Disk space full

**Solutions:**
```bash
# Manual download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Check disk space
df -h
```

## Best Practices

1. **Use Cloud for Production**: LanceDB Cloud provides better reliability and scalability
2. **Monitor Frame Storage**: Frames can consume significant disk space over time
3. **Adjust Frame Limits**: Reduce `scene_frames` limit if memory is constrained
4. **Choose Appropriate Embedding Model**: Balance quality vs. speed based on needs
5. **Regular Cleanup**: Periodically clean old frames if disk space is limited

## Example Workflow

1. **Start the application:**
   ```bash
   ./run.sh
   ```

2. **Wait for scenes to be indexed** (at least 60 seconds for first summary)

3. **Search for scenes:**
   ```bash
   python search_scenes.py "person working at computer"
   ```

4. **View results:**
   ```
   ✅ Found 2 matching scene(s):
   
   [1] Scene ID: scene_1730745600
       Summary: A person sitting at a desk working on a computer
       Similarity: 0.856
       Frames: 12 frame(s)
       Frame paths:
         - frames/scene_1730745600_frame_0001.jpg
         - frames/scene_1730745600_frame_0002.jpg
         ...
   ```

5. **Open frame images** to view the actual video frames

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LANCEDB_PROJECT_SLUG` | LanceDB Cloud project slug | For Cloud only |
| `LANCEDB_API_KEY` | LanceDB Cloud API key | For Cloud only |
| `LANCEDB_REGION` | LanceDB Cloud region | Optional (default: us-east-1) |

### Code Configuration

Edit `scene_indexer.py` to customize:

```python
# Change embedding model
indexer = SceneIndexer(embedding_model="sentence-transformers/all-mpnet-base-v2")

# Change table name
indexer = SceneIndexer(table_name="my_custom_table")

# Change frames directory
indexer = SceneIndexer(frames_dir="custom_frames")
```

## Summary

LanceDB integration provides:
- ✅ Automatic scene indexing
- ✅ Semantic search over summaries
- ✅ Frame association with scenes
- ✅ Natural language query interface
- ✅ Cloud and local database support

All scenes are automatically indexed as summaries are generated, enabling powerful semantic search capabilities over your video content.

