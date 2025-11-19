# Real-Time Video Summarization System

A comprehensive Python application that captures video from a camera, processes frames using a Vision-Language Model (VLM), generates continuous summaries with a Language Model (LLM), performs face recognition, and tracks all metrics with Langfuse.

## Features

- **Real-time video capture** at 4 FPS (configurable)
- **VLM-powered captions** using Qwen3 (qwen3 model)
- **LLM-based summarization** using Mistral-7B-Instruct-v0.3
- **Face recognition** with reference images
- **Langfuse integration** for comprehensive telemetry tracking
- **LanceDB semantic search** for scene retrieval by natural language queries
- **Live UI display** showing captions, summaries, and video feed
- **Detailed statistics** on shutdown

## Architecture

```
real_time_summarizer.py      # Main application with async video processing
├── vlm_client_module.py      # VLM API client with statistics
├── llm_summarizer.py         # LLM-based caption synthesizer
├── face_recognition_module.py # Face recognition with reference images
├── langfuse_tracker.py       # Langfuse telemetry integration
├── scene_indexer.py          # LanceDB semantic search integration
└── search_scenes.py          # CLI tool for scene search
```

## Installation

### 1. Prerequisites

- Python 3.9 or higher
- macOS (MPS), Linux (CUDA), or CPU
- Camera device (webcam)

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Note**: Installing `dlib` (required for face-recognition) may require:

**macOS:**
```bash
brew install cmake
pip install dlib
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
```

### 3. Environment Setup

**Recommended: Use .env file (keeps secrets out of git)**

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your credentials:
   ```bash
   # Langfuse (Telemetry Tracking)
   LANGFUSE_HOST=https://cloud.langfuse.com
   LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
   LANGFUSE_SECRET_KEY=your-langfuse-secret-key
   
   # LanceDB (Vector Database)
   LANCEDB_PROJECT_SLUG=your-project-slug
   LANCEDB_API_KEY=your-lancedb-api-key
   LANCEDB_REGION=us-east-1
   
   # OpenAI (for text-embedding-3-small model)
   OPENAI_API_KEY=your-openai-api-key
   ```

3. The `.env` file is automatically loaded by `./run.sh` and `./run_amber.sh`

**Alternative: Manual Environment Variables**

**Langfuse (Telemetry Tracking):**

**Easy way** - Use the provided setup script:
```bash
source setup_langfuse.sh
```

**Manual way** - Export environment variables:
```bash
export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key-here"
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key-here"
```

**LanceDB (Semantic Search) - Optional:**

**For LanceDB Cloud:**
```bash
export LANCEDB_PROJECT_SLUG="your-project-slug"
export LANCEDB_API_KEY="your-api-key"
export LANCEDB_REGION="us-east-1"  # Optional
```

**For Local Development:**
No setup needed! The system will automatically use a local database at `data/scenes/` if Cloud credentials are not set.

See `LANCEDB_INTEGRATION.md` for detailed setup instructions.

**Note**: 
- The `.env` file is in `.gitignore` and will NOT be committed to git
- `.env.example` is a template that IS committed (safe to share)
- Get your own credentials from https://cloud.langfuse.com, https://cloud.lancedb.com, and https://platform.openai.com/api-keys

### 4. Reference Faces (Optional)

To enable face recognition:

1. Create a `reference_faces/` directory:
   ```bash
   mkdir reference_faces
   ```

2. Add reference images (JPEG/PNG) named after the person:
   ```
   reference_faces/
   ├── john_doe.jpg
   ├── jane_smith.jpg
   └── alice_johnson.png
   ```

The filename (without extension) will be used as the person's name.

## Usage

### Basic Usage

**Easiest way (recommended):**
```bash
./run.sh
```

This automatically sets up Langfuse credentials and runs the application.

**Manual way:**
```bash
source venv/bin/activate
source setup_langfuse.sh
python real_time_summarizer.py
```

### Custom Camera Selection

```bash
# Use camera device 1 instead of default (0)
./run.sh --camera 1

# List available cameras first (macOS/Linux)
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### Custom Frame Rate

```bash
# Process 2 frames per second instead of default 4
./run.sh --fps 2
```

### Full Options

```bash
./run.sh --camera 0 --fps 4
```

## Controls

- **ESC** or **Q**: Exit the application
- The system will automatically print statistics on exit

## Configuration

### VLM Settings

Edit `vlm_client_module.py` to adjust:
- `max_tokens`: Maximum tokens for captions (default: 128)
- `temperature`: Sampling temperature (default: 0.2)

### LLM Settings

Edit `llm_summarizer.py` to adjust:
- `max_new_tokens`: Maximum tokens for summaries (default: 150)
- `model_name`: Change LLM model (default: Mistral-7B-Instruct-v0.3)

### Face Recognition

Edit `face_recognition_module.py` to adjust:
- `tolerance`: Face matching sensitivity (default: 0.6, lower = stricter)

## Output

### Live Display Window

The application shows a window with:
- **Top half**: Live camera feed
- **Middle section**: Latest VLM caption
- **Bottom section**: Current LLM summary
- **Footer**: Statistics (frames captured/processed)

### Console Statistics

On exit, detailed statistics are printed:

```
VLM API STATISTICS
==============================================================
Total Requests: 245
Successful: 245 (100.0%)
Failed: 0

Timing (ms):
  Image Processing: 5.23 ms (avg)
  API Request:      342.15 ms (avg)
  Total Time:       347.38 ms (avg)
  API Request Range: 298.42 - 456.78 ms
==============================================================

LLM SUMMARIZATION STATISTICS
==============================================================
Total Updates: 245
Successful: 245
Failed: 0

Processing Time: 1250.34 ms (avg)
Processing Range: 1100.23 - 1456.89 ms
==============================================================
```

### Langfuse Dashboard

View comprehensive metrics at https://cloud.langfuse.com:

**Observability Metrics:**
- Time to First Token (TTFT) by model and prompt name
- API call durations and latencies
- Frame processing times
- Success/failure rates
- Detailed traces for each frame processing cycle

**Score Metrics (Home Page Dashboard):**
- `vlm_success` - VLM caption generation success rate
- `vlm_latency_score` - Normalized latency performance (0-1 scale)
- `vlm_quality` - Caption quality based on length
- `llm_success` - LLM summary generation success rate
- `llm_processing_score` - Normalized processing time performance
- `llm_quality` - Summary quality based on length and captions processed
- `overall_quality` - Overall trace quality score

All scores are automatically logged and visible on the Langfuse home page dashboard. See `LANGFUSE_INTEGRATION.md` for detailed integration guide.

### LanceDB Semantic Search

Search indexed video scenes using natural language queries:

```bash
# Search for scenes
python search_scenes.py "a man wearing a hat"

# List all indexed scenes
python search_scenes.py --list
```

**Features:**
- Automatic scene indexing after each summary generation
- Vector embeddings for semantic similarity search
- Frame storage with scene association
- Natural language query interface

See `LANCEDB_INTEGRATION.md` for complete integration guide.

## Usage Guide

### 1. Captioning Frames and Indexing Scenes

The main application captures video frames, generates captions, and indexes scenes for semantic search.

**Basic Usage:**
```bash
# Run with interactive model selection menu
./run.sh

# Run with specific camera (default is 0)
./run.sh --camera 1

# Run with custom frame rate (default is 4 FPS)
./run.sh --fps 2

# Combine options
./run.sh --camera 0 --fps 4
```

**What it does:**
- Shows an interactive menu to select embedding model:
  - Option 1: OpenAI text-embedding-3-small (1536 dimensions, recommended)
  - Option 2: BAAI/bge-base-en (768 dimensions)
  - Option 3: intfloat/e5-large-v2 (1024 dimensions)
  - Option 4: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- Captures video frames from your camera
- Generates VLM captions for each frame
- Creates LLM summaries of recent captions
- Automatically indexes scenes to LanceDB for semantic search
- Displays live video feed with captions and summaries

**Requirements:**
- `.env` file with API keys (Langfuse, LanceDB, OpenAI if using OpenAI model)
- Camera access permissions

**Output:**
- Frames saved to `frames/` directory
- Scenes indexed to LanceDB (model-specific tables)
- Statistics printed on exit (VLM and LLM performance)

---

### 2. Amber Assistant - Interactive Scene Search

Amber is a GUI assistant for natural language scene search with time-based filtering.

**Basic Usage:**
```bash
# Launch Amber Assistant GUI
./run_amber.sh
```

**Features:**
- Natural language query interface
- Time-based filtering (e.g., "yesterday", "past 2 hours", "last week")
- Displays results with images, relevance scores, and timestamps
- Shows up to 3 frame images per result
- Processing time tracking

**Example Queries:**
- "Do you have any records of hyenas from yesterday?"
- "Show me scenes with bicycles from the past 2 hours"
- "Are there any records of lions last week?"
- "Do you have any records on a man wearing glasses from two weeks ago?"

**Configuration:**
- Uses embedding model from `.env` file (defaults to `BAAI/bge-base-en`)
- Automatically uses the correct LanceDB table based on model
- Requires `.env` file with LanceDB and Langfuse credentials

**Output:**
- Interactive GUI window
- Real-time search results with images
- Detailed statistics (vector similarity, relevance scores, timestamps)

---

### 3. Semantic Query Testing

Batch test semantic queries and generate HTML/PDF reports comparing different embedding models.

**Basic Usage:**
```bash
# Test with default model (from .env or text-embedding-3-small)
python test_semantic_queries.py

# Test with specific embedding model
python test_semantic_queries.py --embedding-model "BAAI/bge-base-en"
python test_semantic_queries.py --embedding-model "text-embedding-3-small"
python test_semantic_queries.py --embedding-model "intfloat/e5-large-v2"
python test_semantic_queries.py --embedding-model "e5-large-v2"  # Short alias works too

# Test with time filter
python test_semantic_queries.py --time-filter "past 15 minutes"
python test_semantic_queries.py --time-filter "past 30 minutes"
python test_semantic_queries.py --time-filter "past 2 hours"
python test_semantic_queries.py --time-filter "past 24 hours"
python test_semantic_queries.py --time-filter "yesterday"
python test_semantic_queries.py --time-filter "past 3 days"
python test_semantic_queries.py --time-filter "last week"

# Combine model and time filter
python test_semantic_queries.py --embedding-model "BAAI/bge-base-en" --time-filter "past 24 hours"

# Test all 4 models and generate comparison reports
python test_semantic_queries.py --all-models

# Test with specific table
python test_semantic_queries.py --scene-table "scene_embeddings_bge"

# Search all data (no time filter)
python test_semantic_queries.py --time-filter ""
```

**Supported Time Filters:**
- `"past X minutes"` - e.g., "past 15 minutes", "past 30 minutes"
- `"past X hours"` - e.g., "past 2 hours", "past 0.25 hours" (15 min)
- `"past X days"` - e.g., "past 3 days"
- `"last X minutes/hours/days"` - same as "past"
- `"yesterday"`, `"today"`, `"past week"`, `"last month"`

**Supported Embedding Models:**
- `BAAI/bge-base-en` (768 dimensions)
- `intfloat/e5-large-v2` (1024 dimensions)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `text-embedding-3-small` (1536 dimensions, requires OPENAI_API_KEY)

**Output:**
- HTML report in `reports/html/semantic_query_test_report_*.html`
- Shows query results with images, scores, and statistics
- Can be converted to PDF using `convert_report_to_pdf.py`

**View Reports:**
```bash
# Serve HTML reports locally
python serve_report.py
# Opens http://localhost:8000

# Or open directly
open reports/html/semantic_query_test_report_*.html
```

---

### 4. Model Comparison Report

Generate comprehensive comparison reports including VLM/LLM statistics and semantic query results across all embedding models.

**Basic Usage:**
```bash
# Generate comparison report
python generate_model_comparison_report.py
```

**What it includes:**
- Overall performance comparison table
- VLM API statistics for each model (request counts, timing metrics)
- LLM summarization statistics (processing times, success rates)
- Semantic query results summary from HTML reports
- Visual comparison with best/worst performers highlighted

**Output:**
- HTML report in `reports/html/model_comparison_report_*.html`
- Can be converted to PDF using `convert_report_to_pdf.py`

**Convert to PDF:**
```bash
# Convert latest HTML report to PDF
python convert_report_to_pdf.py
```

**Note:** The comparison report uses hardcoded statistics from the script. To update with your own statistics, edit `generate_model_comparison_report.py` and update the `MODEL_STATS` dictionary with your VLM/LLM statistics and HTML report paths.

---

### Quick Workflow Example

```bash
# 1. Capture and index frames (with OpenAI model)
./run.sh
# Select option 1 for OpenAI text-embedding-3-small

# 2. Test semantic queries on recent data
python test_semantic_queries.py --embedding-model "text-embedding-3-small" --time-filter "past 2 hours"

# 3. Use Amber Assistant for interactive search
./run_amber.sh
# Ask: "Do you have any records of people from the past hour?"

# 4. Generate comparison report
python generate_model_comparison_report.py

# 5. View results
python serve_report.py
```

## Troubleshooting

### Quick Diagnostics

Run the test script to verify setup:
```bash
python test_setup.py
```

This checks all dependencies, camera access, and environment configuration.

### Common Issues

#### Camera Not Found

**Error:** `Failed to open camera 0`

**Solutions:**
1. Try different camera index:
   ```bash
   ./run.sh --camera 1
   ```

2. List available cameras:
   ```bash
   python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```

3. Check camera permissions (macOS):
   - System Settings → Privacy & Security → Camera
   - Ensure Terminal/iTerm has camera access

#### LLM Model Not Loading

**Error:** `ValueError: Cannot instantiate this tokenizer from a slow version...`

**Solution:**
```bash
pip install sentencepiece protobuf
# Or reinstall all dependencies
pip install -r requirements.txt
```

**Memory Issues:**
The Mistral-7B model requires ~14GB RAM. If you encounter memory issues:
- Use a smaller model (e.g., `mistralai/Mistral-7B-Instruct-v0.2`)
- Reduce `max_new_tokens` in `llm_summarizer.py`
- Close other applications

#### Langfuse Not Tracking

**Symptom:** `Langfuse credentials not found in environment`

**Solutions:**
1. Use the run script (easiest):
   ```bash
   ./run.sh
   ```

2. Source the setup script:
   ```bash
   source setup_langfuse.sh
   python real_time_summarizer.py
   ```

3. Export variables manually:
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-lf-..."
   export LANGFUSE_SECRET_KEY="sk-lf-..."
   ```

**Verify:** Look for `Langfuse tracking enabled` in console logs.

#### VLM API Connection Issues

**Error:** API requests failing or timing out

**Solutions:**
1. Check endpoint is accessible:
   ```bash
   curl http://38.80.152.249:30447/v1/models
   ```

2. Verify network connection and firewall settings

#### Face Recognition Not Working

**Solutions:**
1. Install dlib properly:
   ```bash
   # macOS
   brew install cmake
   pip install dlib
   
   # Linux
   sudo apt-get install build-essential cmake
   sudo apt-get install libopenblas-dev liblapack-dev
   pip install dlib
   ```

2. Add reference images to `reference_faces/` directory:
   ```bash
   cp your_photo.jpg reference_faces/your_name.jpg
   ```

3. Use clear, frontal face photos with good lighting

4. Face recognition will automatically disable if dependencies are missing

#### Slow Performance

**Solutions:**
1. Lower frame rate:
   ```bash
   ./run.sh --fps 2
   ```

2. Close other applications (LLM needs ~14GB RAM)

3. Check system resources and ensure GPU is being used (if available)

#### Virtual Environment Issues

**Symptoms:** "Module not found" errors

**Solutions:**
1. Ensure venv is activated:
   ```bash
   source venv/bin/activate
   # Should see (venv) in prompt
   ```

2. Recreate virtual environment:
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Verify Python version (3.9+):
   ```bash
   python --version
   ```

## Performance Tips

1. **GPU Acceleration**: 
   - CUDA (NVIDIA): Automatically used if available
   - MPS (Apple Silicon): Automatically used on M1/M2/M3 Macs
   - CPU: Falls back automatically

2. **Reduce FPS**: Lower frame rate for slower systems:
   ```bash
   python real_time_summarizer.py --fps 2
   ```

3. **Disable Face Recognition**: Remove `reference_faces/` directory or keep it empty

## Architecture Details

### Async Processing

The system uses asynchronous processing to ensure:
- Video capture is never blocked
- VLM/LLM inference happens in background
- UI remains responsive

### Langfuse Integration

All operations are tracked with:
- **Traces**: One per frame processing cycle
- **Spans**: Individual operations (face recognition, VLM, LLM)
- **Generations**: Model outputs with TTFT metrics and timing
- **Scores**: Quality, performance, and success metrics for dashboard visualization
- **Metadata**: Timing, success/failure, model details

**Key Features:**
- Proper TTFT calculation (generation created before API call)
- Score logging for home page dashboard
- Trace-level and observation-level metrics
- Automatic data synchronization

See `LANGFUSE_INTEGRATION.md` for complete integration details.

### LanceDB Semantic Search

Scenes are automatically indexed with:
- **Embeddings**: Vector representations of summaries using sentence-transformers
- **Frame Storage**: Associated frames saved to disk
- **Metadata**: Timestamps, scene IDs, and frame paths

**Search Capabilities:**
- Natural language queries (e.g., "a man wearing a hat")
- Similarity-based retrieval
- Frame path association for viewing actual frames

See `LANCEDB_INTEGRATION.md` for complete integration details.

### Caption Synthesis

The LLM summarizer:
1. Filters duplicate/similar captions
2. Uses only recent captions (last 10) to avoid token limits
3. Generates concise, non-repetitive summaries
4. Focuses on new information and changes

## References

This project integrates patterns from:
- `/Users/nestor/armando_new/vlm_client/qwenunc.py` - VLM model usage
- `/Users/nestor/armando_new/vlm_client/vlm_client.txt` - API client patterns
- `/Users/nestor/armando_new/facerecognition2 copy/facerecog2.py` - Face recognition

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues or questions:
- Check the troubleshooting section
- Review Langfuse dashboard for detailed metrics
- Check console logs for detailed error messages

