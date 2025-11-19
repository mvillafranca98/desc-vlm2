# Quick Start Guide

Get up and running in 5 minutes!

## 1. Install Dependencies

```bash
# Run the setup script
bash setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Set Up Langfuse (Optional but Recommended)

**Easy way** - Use the setup script:
```bash
source setup_langfuse.sh
```

**Manual way** - Export environment variables:
```bash
export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key-here"
export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key-here"
```

## 3. Add Reference Faces (Optional)

```bash
# Add your face photos to reference_faces/
cp /path/to/your_photo.jpg reference_faces/your_name.jpg
```

## 4. Run the Application

**Easiest way** - Use the run script (sets up everything automatically):
```bash
./run.sh
```

**Manual way** - Set variables yourself:
```bash
source venv/bin/activate
source setup_langfuse.sh
python real_time_summarizer.py
```

That's it! The application will:
- Open your camera
- Start processing frames at 4 FPS
- Generate captions with the VLM
- Create summaries with the LLM
- Display everything in a window

## Controls

- Press **ESC** or **Q** to exit
- Statistics will be printed on exit

## Common Issues

### Camera not found?
```bash
# Try a different camera index
python real_time_summarizer.py --camera 1
```

### Too slow?
```bash
# Reduce frame rate
python real_time_summarizer.py --fps 2
```

### VLM API not responding?
Check that the endpoint is accessible:
```bash
curl http://38.80.152.249:30447/v1/models
```

### Memory issues with LLM?
The Mistral-7B model needs ~14GB RAM. Close other applications or use a smaller model.

## What to Expect

### First Run
- LLM model will download (may take several minutes)
- First few frames may be slower as models warm up
- After warm-up, should process at ~4 FPS

### Display Window
- **Top**: Live camera feed
- **Middle**: Latest caption from VLM
- **Bottom**: Current summary from LLM
- **Footer**: Frame statistics

### Console Output
Real-time logging shows:
- Frames captured and processed
- VLM caption generation
- LLM summary updates
- Any errors or warnings

### On Exit
Detailed statistics including:
- Total requests and success rates
- Average processing times
- API request timings

## Next Steps

1. **View Langfuse Dashboard**: Check https://cloud.langfuse.com for detailed metrics
2. **Customize**: Edit configuration in module files
3. **Add Reference Faces**: Improve face recognition accuracy
4. **Adjust FPS**: Balance between real-time and accuracy

Enjoy! 🎥

