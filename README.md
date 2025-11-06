# Real-Time Video Summarization System

A comprehensive Python application that captures video from a camera, processes frames using a Vision-Language Model (VLM), generates continuous summaries with a Language Model (LLM), performs face recognition, and tracks all metrics with Langfuse.

## Features

- **Real-time video capture** at 4 FPS (configurable)
- **VLM-powered captions** using Qwen3 (qwen3 model)
- **LLM-based summarization** using Mistral-7B-Instruct-v0.3
- **Face recognition** with reference images
- **Langfuse integration** for comprehensive telemetry tracking
- **Live UI display** showing captions, summaries, and video feed
- **Detailed statistics** on shutdown

## Architecture

```
real_time_summarizer.py      # Main application with async video processing
├── vlm_client_module.py      # VLM API client with statistics
├── llm_summarizer.py         # LLM-based caption synthesizer
├── face_recognition_module.py # Face recognition with reference images
└── langfuse_tracker.py       # Langfuse telemetry integration
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

**Easy way** - Use the provided setup script:
```bash
source setup_langfuse.sh
```

**Manual way** - Export environment variables:
```bash
export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-61ba616e-de46-4d72-a848-753fd9a5b3fb"
export LANGFUSE_SECRET_KEY="sk-lf-e8c7bb1b-554d-4396-95ad-95c30593d6c8"
```

**Note**: These are the provided credentials for this project. For production use, get your own from https://cloud.langfuse.com

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

