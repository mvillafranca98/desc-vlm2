# Langfuse Integration Guide

Complete guide for integrating Langfuse telemetry tracking into the Real-Time Video Summarization System, including dashboard metrics, scores, and TTFT tracking.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Architecture](#architecture)
4. [Key Implementation Details](#key-implementation-details)
5. [Dashboard Metrics](#dashboard-metrics)
6. [Score Logging](#score-logging)
7. [Troubleshooting](#troubleshooting)

## Overview

This system integrates Langfuse for comprehensive observability:
- **Traces**: Track complete frame processing cycles
- **Spans**: Monitor individual operations (VLM, LLM, face recognition)
- **Generations**: Track model outputs with Time to First Token (TTFT)
- **Scores**: Log quality and performance metrics for dashboard visualization

## Setup

### 1. Install Langfuse SDK

```bash
pip install langfuse
```

### 2. Configure Credentials

**Option A: Environment Variables (Recommended)**

```bash
export LANGFUSE_HOST="https://cloud.langfuse.com"
export LANGFUSE_PUBLIC_KEY="pk-lf-61ba616e-de46-4d72-a848-753fd9a5b3fb"
export LANGFUSE_SECRET_KEY="sk-lf-e8c7bb1b-554d-4396-95ad-95c30593d6c8"
```

**Option B: Use Setup Script**

```bash
source setup_langfuse.sh
```

### 3. Verify Integration

The system will automatically initialize Langfuse on startup. Look for:
```
Langfuse tracking enabled: https://cloud.langfuse.com
```

## Architecture

### File Structure

```
langfuse_tracker.py          # Main Langfuse integration module
├── LangfuseTracker          # Main tracker class
│   ├── create_trace()       # Create trace context
│   ├── create_span()        # Create span context
│   ├── start_generation()   # Start generation BEFORE API call
│   ├── log_generation()     # Log generation (legacy, not recommended)
│   └── log_score()          # Log scores for dashboard
└── Context Managers         # TraceContext, SpanContext, Dummy classes
```

### Integration Points

1. **real_time_summarizer.py**: Main application using LangfuseTracker
2. **vlm_client_module.py**: Captures timing for VLM API calls
3. **llm_summarizer.py**: Captures timing for LLM generation

## Key Implementation Details

### 1. Correct TTFT Calculation

**Problem**: If generation is created AFTER the API call, TTFT becomes negative.

**Solution**: Create generation BEFORE the API call, then update it after.

```python
# ✅ CORRECT: Create generation before API call
generation_context = self.langfuse_tracker.start_generation(
    trace_id=trace.id,
    name="vlm_caption",
    model=self.vlm_client.model,
    input=prompt,
    metadata={'faces_detected': len(recognized_faces)}
)

generation_context.__enter__()  # Sets correct start_time

try:
    # Make API call (timing captured here)
    caption_result = self.vlm_client.generate_caption(frame, prompt)
    
    # Update generation with results
    self.langfuse_tracker.client.update_current_generation(
        output=caption,
        completion_start_time=datetime.fromtimestamp(completion_start_time)
    )
finally:
    generation_context.__exit__(None, None, None)
```

**Why This Works:**
- Generation's `start_time` = when we create it (before API call) ✅
- `completion_start_time` = when first token arrives (during API call) ✅
- TTFT = `completion_start_time - start_time` = positive value ✅

### 2. Timing Data Collection

**VLM Client** (`vlm_client_module.py`):
```python
start_time = time.time()  # When API request starts
response = self.client.chat.completions.create(...)
completion_start_time = time.time()  # When response received (TTFT)
end_time = time.time()  # When processing completes

return {
    'start_time': start_time,
    'completion_start_time': completion_start_time,
    'end_time': end_time
}
```

**LLM Summarizer** (`llm_summarizer.py`):
```python
start_time = time.time()  # When generation starts
generation_start_time = time.time()  # After tokenization, before generation
outputs = self.model.generate(...)
completion_start_time = generation_start_time + 0.01  # Approximation for local models
end_time = time.time()

return {
    'start_time': start_time,
    'completion_start_time': completion_start_time,
    'end_time': end_time
}
```

### 3. Passing Timing to Langfuse

```python
# Convert to datetime objects (required by Langfuse)
from datetime import datetime

update_params = {
    "output": caption,
    "completion_start_time": datetime.fromtimestamp(completion_start_time)
}

self.langfuse_tracker.client.update_current_generation(**update_params)
```

## Dashboard Metrics

### Time to First Token (TTFT)

**Location**: Dashboards > Langfuse Latency Dashboard

**Charts:**
- **"Avg Time To First Token by Prompt Name"**: Shows average TTFT for `vlm_caption` and `llm_summary`
- **"P 95 Time To First Token by Model"**: Shows 95th percentile TTFT for `qwen3` and `mistralai/Mistral-7B-Instruct-v0.3`

**Requirements:**
- Generation must be created BEFORE API call (correct `start_time`)
- `completion_start_time` must be passed as datetime object
- Both parameters must be set for TTFT calculation

### Latency Metrics

**Location**: Dashboards > Langfuse Latency Dashboard

**Charts:**
- **"P 95 Latency by Model"**: Shows latency percentiles for each model
- **"Max Latency by User Id"**: Shows maximum latency per trace

### Score Metrics

**Location**: Home Page Dashboard

**Available Scores:**

| Score Name | Type | Description | Range |
|------------|------|-------------|-------|
| `vlm_success` | NUMERIC | VLM caption generation success | 0.0 (failure) or 1.0 (success) |
| `vlm_latency_score` | NUMERIC | Normalized latency performance | 0.0-1.0 (higher is better) |
| `vlm_quality` | NUMERIC | Caption quality based on length | 0.0-1.0 (higher is better) |
| `llm_success` | NUMERIC | LLM summary generation success | 0.0 (failure) or 1.0 (success) |
| `llm_processing_score` | NUMERIC | Normalized processing time | 0.0-1.0 (higher is better) |
| `llm_quality` | NUMERIC | Summary quality score | 0.0-1.0 (higher is better) |
| `overall_quality` | NUMERIC | Overall trace quality | 0.0-1.0 (higher is better) |

## Score Logging

### Basic Usage

```python
# Log a numeric score
self.langfuse_tracker.log_score(
    trace_id=trace.id,
    name="vlm_success",
    value=1.0,  # Success
    data_type="NUMERIC",
    comment="VLM caption generation succeeded",
    score_trace=False  # Score the observation, not the trace
)

# Log a trace-level score
self.langfuse_tracker.log_score(
    trace_id=trace.id,
    name="overall_quality",
    value=0.85,
    data_type="NUMERIC",
    comment="Overall trace quality score",
    score_trace=True  # Score the entire trace
)
```

### Score Types

**NUMERIC**: Float values (0.0-1.0 scale recommended)
```python
self.langfuse_tracker.log_score(
    trace_id=trace.id,
    name="latency_score",
    value=0.75,  # Automatically detected as NUMERIC
    comment="Normalized latency: 1250ms"
)
```

**BOOLEAN**: Automatically converted to 0.0 or 1.0
```python
self.langfuse_tracker.log_score(
    trace_id=trace.id,
    name="success",
    value=True,  # Automatically converted to 1.0
    comment="Operation succeeded"
)
```

**CATEGORICAL**: String values
```python
self.langfuse_tracker.log_score(
    trace_id=trace.id,
    name="quality_level",
    value="high",  # Automatically detected as CATEGORICAL
    comment="Quality assessment"
)
```

### Score Calculation Examples

**Latency Score** (normalized, higher is better):
```python
api_time_ms = caption_result.get('api_time', 0)
# Normalize: assume 5000ms is worst, 0ms is best
latency_score = max(0.0, min(1.0, 1.0 - (api_time_ms / 5000.0)))
```

**Quality Score** (based on output length):
```python
caption_length = len(caption)
quality_score = min(1.0, caption_length / 100.0)  # 100 chars = perfect
```

**Overall Quality** (composite score):
```python
overall_score = (success_score + latency_score + quality_score) / 3.0
```

## Implementation Checklist

### ✅ Required Changes for Langfuse Integration

1. **Install Langfuse SDK**
   ```bash
   pip install langfuse
   ```

2. **Set Environment Variables**
   ```bash
   export LANGFUSE_PUBLIC_KEY="your-public-key"
   export LANGFUSE_SECRET_KEY="your-secret-key"
   ```

3. **Initialize LangfuseTracker**
   ```python
   from langfuse_tracker import LangfuseTracker
   self.langfuse_tracker = LangfuseTracker()
   ```

4. **Create Traces for Each Frame**
   ```python
   with self.langfuse_tracker.create_trace("frame_processing") as trace:
       # Process frame
   ```

5. **Create Spans for Operations**
   ```python
   with self.langfuse_tracker.create_span(trace.id, "vlm_caption"):
       # VLM operation
   ```

6. **Start Generations BEFORE API Calls**
   ```python
   generation_context = self.langfuse_tracker.start_generation(...)
   generation_context.__enter__()
   try:
       # Make API call
       result = api_call()
       # Update generation
   finally:
       generation_context.__exit__(None, None, None)
   ```

7. **Pass Timing Parameters**
   ```python
   from datetime import datetime
   update_params = {
       "completion_start_time": datetime.fromtimestamp(completion_start_time)
   }
   self.langfuse_tracker.client.update_current_generation(**update_params)
   ```

8. **Log Scores for Dashboard**
   ```python
   self.langfuse_tracker.log_score(
       trace_id=trace.id,
       name="score_name",
       value=score_value,
       data_type="NUMERIC"
   )
   ```

9. **Flush on Exit**
   ```python
   self.langfuse_tracker.flush()  # Ensures all data is sent
   ```

### ✅ Timing Data Collection

**VLM Client:**
- Capture `start_time` before API call
- Capture `completion_start_time` when response received
- Capture `end_time` after processing
- Return all timing in result dictionary

**LLM Summarizer:**
- Capture `start_time` at function start
- Capture `generation_start_time` before model.generate()
- Approximate `completion_start_time` (for local models)
- Capture `end_time` after decoding
- Return all timing in result dictionary

## Troubleshooting

### Issue: Negative TTFT Values

**Symptom**: Dashboard shows negative TTFT values

**Cause**: Generation created AFTER API call completes

**Solution**: Use `start_generation()` BEFORE the API call:
```python
# ❌ WRONG: Creates generation after API call
result = api_call()
self.langfuse_tracker.log_generation(...)  # start_time is wrong!

# ✅ CORRECT: Create generation before API call
generation_context = self.langfuse_tracker.start_generation(...)
generation_context.__enter__()
result = api_call()  # start_time is correct!
generation_context.__exit__(None, None, None)
```

### Issue: No Data in Dashboard

**Symptom**: Charts show "n/a" or empty

**Possible Causes:**
1. **Missing timing parameters**: Ensure `completion_start_time` is passed
2. **Generation created at wrong time**: Use `start_generation()` before API call
3. **Data not flushed**: Call `flush()` before exit
4. **Wrong model names**: Ensure model names match exactly

**Solution:**
```python
# Ensure completion_start_time is passed
update_params = {
    "completion_start_time": datetime.fromtimestamp(completion_start_time)
}
self.langfuse_tracker.client.update_current_generation(**update_params)

# Flush before exit
self.langfuse_tracker.flush()
```

### Issue: Scores Not Appearing

**Symptom**: Scores don't show on home page dashboard

**Possible Causes:**
1. **Score not logged**: Ensure `log_score()` is called
2. **Wrong trace_id**: Use correct trace ID
3. **Data type mismatch**: Ensure value matches data_type

**Solution:**
```python
# Verify score is logged
self.langfuse_tracker.log_score(
    trace_id=trace.id,  # Must be valid trace ID
    name="vlm_success",
    value=1.0,
    data_type="NUMERIC"  # Must match value type
)

# Check logs for errors
# Should see: "Score logged: vlm_success=1.0 (type=NUMERIC)"
```

### Issue: Only One Model Showing Data

**Symptom**: Only VLM or only LLM data appears

**Cause**: One model not logging with proper timing

**Solution**: Ensure both VLM and LLM use the same pattern:
1. Create generation before API call
2. Pass `completion_start_time` correctly
3. Update generation with output

## Best Practices

1. **Always create generations before API calls** - Ensures correct TTFT
2. **Pass timing as datetime objects** - Required by Langfuse API
3. **Flush before exit** - Ensures all data is sent
4. **Use consistent score names** - Easier to track in dashboard
5. **Normalize scores to 0-1 range** - Better visualization
6. **Add meaningful comments** - Helps with debugging
7. **Log both success and failure** - Complete picture of system health

## Viewing Data in Langfuse

### Home Page Dashboard

1. Navigate to https://cloud.langfuse.com
2. Select your project
3. View scores in the home page dashboard
4. Create custom widgets to visualize specific metrics

### Latency Dashboard

1. Go to **Dashboards** > **Langfuse Latency Dashboard**
2. View:
   - Avg Time To First Token by Prompt Name
   - P 95 Time To First Token by Model
   - P 95 Latency by Model

### Traces View

1. Go to **Observability** > **Tracing**
2. Click on any trace to see:
   - Complete hierarchy of operations
   - Timing for each span
   - Generations with TTFT
   - Scores associated with observations

### Generations View

1. Go to **Observability** > **Generations**
2. Filter by:
   - Model name (`qwen3` or `mistralai/Mistral-7B-Instruct-v0.3`)
   - Generation name (`vlm_caption` or `llm_summary`)
3. View detailed metrics including TTFT

## Summary

This integration provides:
- ✅ Complete observability of all operations
- ✅ Accurate TTFT metrics for both models
- ✅ Score tracking for dashboard visualization
- ✅ Trace-level and observation-level metrics
- ✅ Automatic data synchronization

All metrics are automatically uploaded to Langfuse and visible in the dashboard within 30 seconds of generation.

