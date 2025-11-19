#!/usr/bin/env python3
"""
Real-Time Video Summarization System with VLM, LLM, and Langfuse Integration
"""

import os
import cv2
import time
import asyncio
import logging
import threading
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
import numpy as np

# Set environment variables before importing other modules
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Provide default LanceDB Cloud credentials if not already set
os.environ.setdefault('LANCEDB_PROJECT_SLUG', 'descvlm2-lnh0lv')
os.environ.setdefault('LANCEDB_API_KEY', 'sk_SJLWUL2G2JDQFLZ5WKKM5DC4KJKIVAV73IECRDHTRSBUAEMY2DSQ====')
os.environ.setdefault('LANCEDB_REGION', 'us-east-1')

# Set OpenCV to use single thread to avoid segfaults on macOS
cv2.setNumThreads(1)

from vlm_client_module import VLMClientWrapper
from llm_summarizer import LLMSummarizer
from face_recognition_module import FaceRecognizer
from langfuse_tracker import LangfuseTracker
from scene_indexer import SceneIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific loggers to reduce noise
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


class VideoSummarizer:
    """Main video summarization system."""
    
    def __init__(
        self,
        camera_index: int = 0,
        fps: int = 4,
        vlm_url: str = "http://38.80.152.249:30447/v1",
        vlm_model: str = "qwen3",
        reference_faces_dir: str = "reference_faces",
        caption_interval: float = 5.0,  # Generate caption every 5 seconds
        scene_index_model: Optional[str] = None,
        scene_index_table: Optional[str] = None
    ):
        self.camera_index = camera_index
        self.fps = fps
        self.frame_interval = 1.0 / fps  # For display update rate
        self.caption_interval = caption_interval  # For VLM processing (every 5 seconds)
        self.running = False
        self.cap = None
        
        # Initialize components
        logger.info("Initializing Video Summarizer components...")
        self.vlm_client = VLMClientWrapper(vlm_url, vlm_model)
        self.llm_summarizer = LLMSummarizer()
        self.face_recognizer = FaceRecognizer(reference_faces_dir)
        self.langfuse_tracker = LangfuseTracker()
        
        # Resolve embedding configuration for scene indexing
        embedding_model = scene_index_model or os.getenv("SCENE_INDEX_MODEL", "text-embedding-3-small")
        table_name = scene_index_table or os.getenv("SCENE_INDEX_TABLE")  # None if not set
        
        # Let SceneIndexer auto-generate model-specific table name if SCENE_INDEX_TABLE not set
        self.scene_indexer = SceneIndexer(
            table_name=table_name if table_name else "scene_index",  # Use default to trigger auto-generation
            embedding_model=embedding_model,
            frames_dir=os.getenv("SCENE_INDEX_FRAMES_DIR", "frames"),
            enable_chunking=os.getenv("SCENE_INDEX_ENABLE_CHUNKING", "true").lower() == "true",
            chunk_duration=float(os.getenv("SCENE_INDEX_CHUNK_SECONDS", "10")),
            enable_caption_embeddings=os.getenv("SCENE_INDEX_ENABLE_CAPTION", "true").lower() == "true"
        )
        
        # Log embedding model configuration
        if self.scene_indexer.enabled:
            logger.info(f"✓ Scene indexing enabled")
            logger.info(f"  Model: {self.scene_indexer.embedding_model_name}")
            logger.info(f"  Table: {self.scene_indexer.table_name}")
            logger.info(f"  Dimension: {self.scene_indexer.embedding_dim}")
        else:
            logger.warning("⚠ Scene indexing disabled - check LanceDB configuration")
        
        # State management
        self.latest_caption = "Waiting for first frame..."
        self.current_summary = "No summary yet. Gathering captions..."
        self.caption_queue = deque(maxlen=100)
        self.last_summary_time = 0
        self.last_caption_time = 0  # Track when last caption was generated
        self.summary_interval = 60  # Update summary every 60 seconds (1 minute)
        self.last_langfuse_flush_time = 0  # Track when Langfuse was last flushed
        self.langfuse_flush_interval = 30  # Flush Langfuse every 30 seconds
        
        # Frame storage for scene indexing
        self.scene_frames = []  # Store frames for current scene
        self.scene_start_time = None  # When current scene started
        self.caption_records = []  # Store captions metadata for current scene
        self.scene_face_names = set()
        
        # UI window
        self.window_name = "Video Summarization System"
        self.display_width = 800
        self.display_height = 600
        
        # Statistics
        self.frames_captured = 0
        self.frames_processed = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
        try:
            logger.info(f"Opening camera {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
                
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the pipeline."""
        with self.langfuse_tracker.create_trace("frame_processing") as trace:
            try:
                # Face recognition (skip if face recognizer not available to avoid segfaults)
                recognized_faces = []
                try:
                    with self.langfuse_tracker.create_span(trace.id, "face_recognition"):
                        if self.face_recognizer.enabled:
                            recognized_faces = self.face_recognizer.recognize_faces(frame)
                except Exception as face_error:
                    logger.warning(f"Face recognition error (continuing without): {face_error}")
                
                # VLM captioning
                with self.langfuse_tracker.create_span(trace.id, "vlm_caption"):
                    # Build prompt with face information
                    prompt = "Describe what you see in this image in one concise sentence."
                    if recognized_faces:
                        names = [face.get('name') for face in recognized_faces if face.get('name')]
                        if names:
                            self.scene_face_names.update(names)
                            prompt += f" People visible: {', '.join(names)}."
                    
                    # CRITICAL FIX: Start generation BEFORE API call to ensure correct start_time
                    # This fixes negative TTFT values in the dashboard
                    generation_context = self.langfuse_tracker.start_generation(
                        trace_id=trace.id,
                        name="vlm_caption",
                        model=self.vlm_client.model,
                        input=prompt,
                        metadata={
                            'faces_detected': len(recognized_faces)
                        }
                    )
                    
                    # Enter the generation context BEFORE the API call
                    # This sets the generation's start_time correctly
                    generation_context.__enter__()
                    
                    try:
                        # Make API call (this is where start_time is captured)
                        caption_result = self.vlm_client.generate_caption(frame, prompt)
                        
                        if caption_result['success']:
                            caption = caption_result['caption']
                            # Update caption immediately (thread-safe)
                            self.latest_caption = caption
                            self.caption_queue.append(caption)
                            
                            caption_timestamp = time.time()
                            self.caption_records.append({
                                'text': caption,
                                'timestamp': caption_timestamp,
                                'frame_index': len(self.scene_frames) - 1 if self.scene_frames else None,
                                'frame_path': None,
                                'metadata': {
                                    'faces_detected': [face.get('name') for face in recognized_faces if face.get('name')],
                                    'api_time_ms': caption_result.get('api_time', 0)
                                }
                            })
                            
                            # Log to console
                            logger.info(f"Caption: {caption}")
                            
                            # Update generation with output and timing
                            from datetime import datetime
                            update_params = {
                                "output": caption[:1000],
                                "metadata": {
                                    'faces_detected': len(recognized_faces),
                                    'api_time': caption_result.get('api_time', 0)
                                }
                            }
                            
                            # Set completion_start_time for TTFT calculation
                            if caption_result.get('completion_start_time'):
                                update_params["completion_start_time"] = datetime.fromtimestamp(
                                    caption_result.get('completion_start_time')
                                )
                            
                            self.langfuse_tracker.client.update_current_generation(**update_params)
                            
                            # Log scores for VLM caption generation
                            # Success score (1.0 = success, 0.0 = failure)
                            self.langfuse_tracker.log_score(
                                trace_id=trace.id,
                                name="vlm_success",
                                value=1.0,
                                data_type="NUMERIC",
                                comment="VLM caption generation succeeded",
                                score_trace=False
                            )
                            
                            # Latency score (normalized: lower is better, 0-1 scale)
                            api_time_ms = caption_result.get('api_time', 0)
                            # Normalize: assume 5000ms is worst, 0ms is best
                            latency_score = max(0.0, min(1.0, 1.0 - (api_time_ms / 5000.0)))
                            self.langfuse_tracker.log_score(
                                trace_id=trace.id,
                                name="vlm_latency_score",
                                value=latency_score,
                                data_type="NUMERIC",
                                comment=f"VLM latency: {api_time_ms:.1f}ms",
                                score_trace=False
                            )
                            
                            # Caption quality score (based on length - reasonable length = good)
                            caption_length = len(caption)
                            quality_score = min(1.0, caption_length / 100.0)  # 100 chars = perfect
                            self.langfuse_tracker.log_score(
                                trace_id=trace.id,
                                name="vlm_quality",
                                value=quality_score,
                                data_type="NUMERIC",
                                comment=f"Caption quality based on length: {caption_length} chars",
                                score_trace=False
                            )
                        else:
                            # Update with error
                            self.langfuse_tracker.client.update_current_generation(
                                output="",
                                metadata={
                                    'error': caption_result.get('error', 'Unknown error'),
                                    'faces_detected': len(recognized_faces)
                                }
                            )
                            
                            # Log failure score
                            self.langfuse_tracker.log_score(
                                trace_id=trace.id,
                                name="vlm_success",
                                value=0.0,
                                data_type="NUMERIC",
                                comment=f"VLM caption generation failed: {caption_result.get('error', 'Unknown error')}",
                                score_trace=False
                            )
                    finally:
                        # Exit the generation context
                        generation_context.__exit__(None, None, None)
                
                # LLM summarization - only update every 60 seconds (1 minute)
                current_time = time.time()
                if current_time - self.last_summary_time >= self.summary_interval:
                    logger.info(f"Updating summary with {len(self.caption_queue)} captions...")
                    
                    with self.langfuse_tracker.create_span(trace.id, "llm_summarization"):
                        # Build input for logging
                        recent_captions = list(self.caption_queue)[-10:] if len(self.caption_queue) > 10 else list(self.caption_queue)
                        input_text = str(recent_captions)
                        
                        # CRITICAL FIX: Start generation BEFORE LLM call to ensure correct start_time
                        # This fixes missing TTFT data for Mistral-7B in the dashboard
                        generation_context = self.langfuse_tracker.start_generation(
                            trace_id=trace.id,
                            name="llm_summary",
                            model=self.llm_summarizer.model_name,
                            input=input_text,
                            metadata={
                                'captions_count': len(self.caption_queue)
                            }
                        )
                        
                        # Enter the generation context BEFORE the LLM call
                        generation_context.__enter__()
                        
                        try:
                            # Make LLM call (this is where start_time is captured)
                            summary_result = self.llm_summarizer.update_summary(
                                list(self.caption_queue)
                            )
                            
                            if summary_result['success']:
                                self.current_summary = summary_result['summary']
                                self.last_summary_time = current_time
                                logger.info(f"Summary updated: {self.current_summary[:100]}...")
                                
                                # Update generation with output and timing
                                from datetime import datetime
                                update_params = {
                                    "output": summary_result['summary'][:1000],
                                    "metadata": {
                                        'captions_count': len(self.caption_queue),
                                        'processing_time': summary_result.get('processing_time', 0)
                                    }
                                }
                                
                                # Set completion_start_time for TTFT calculation
                                if summary_result.get('completion_start_time'):
                                    update_params["completion_start_time"] = datetime.fromtimestamp(
                                        summary_result.get('completion_start_time')
                                    )
                                
                                self.langfuse_tracker.client.update_current_generation(**update_params)
                                
                                # Flush Langfuse after summary update (important event)
                                try:
                                    self.langfuse_tracker.flush()
                                    logger.debug("Langfuse events flushed (after summary update)")
                                except Exception as e:
                                    logger.warning(f"Error flushing Langfuse after summary update: {e}")
                                
                                # Index scene in LanceDB for semantic search
                                if not self.scene_indexer.enabled:
                                    logger.debug("Scene indexing skipped: indexer not enabled")
                                elif len(self.scene_frames) == 0:
                                    logger.warning(f"Scene indexing skipped: no frames captured (scene_indexer.enabled={self.scene_indexer.enabled}, scene_frames={len(self.scene_frames)})")
                                
                                if self.scene_indexer.enabled and len(self.scene_frames) > 0:
                                    try:
                                        # Generate unique scene ID
                                        scene_id = f"scene_{int(current_time)}"
                                        
                                        # Save frames to disk and collect paths
                                        frame_paths = []
                                        frame_index_map = {}
                                        for idx, frame_data in enumerate(self.scene_frames):
                                            frame_path = self.scene_indexer.save_frame(
                                                frame_data['frame'],
                                                scene_id,
                                                idx
                                            )
                                            if frame_path:
                                                frame_paths.append(frame_path)
                                                frame_index_map[idx] = frame_path
                                        
                                        caption_records_with_paths = []
                                        for caption_record in self.caption_records:
                                            record_copy = dict(caption_record)
                                            frame_index = record_copy.get('frame_index')
                                            if frame_index is not None:
                                                record_copy['frame_path'] = frame_index_map.get(frame_index)
                                            caption_records_with_paths.append(record_copy)
                                        
                                        scene_metadata = {
                                            'caption_count': len(self.caption_records),
                                            'summary_length': len(summary_result['summary']),
                                            'recognized_faces': list(self.scene_face_names),
                                            'chunking_enabled': self.scene_indexer.enable_chunking,
                                            'caption_embeddings_enabled': self.scene_indexer.enable_caption_embeddings,
                                            'session_id': self.scene_indexer.session_id
                                        }
                                        
                                        # Index the scene with summary and frame paths
                                        if frame_paths:
                                            # Instrument LanceDB indexing with Langfuse
                                            lancedb_generation = self.langfuse_tracker.start_generation(
                                                trace_id=trace.id,
                                                name="lancedb_index_scene",
                                                model="lancedb_scene_indexer",
                                                input=summary_result['summary'][:500],
                                                metadata={
                                                    'frame_count': len(frame_paths),
                                                    'scene_id': scene_id,
                                                    'caption_count': len(caption_records_with_paths),
                                                    'recognized_faces': list(self.scene_face_names)
                                                }
                                            )
                                            lancedb_generation.__enter__()
                                            index_start = time.time()
                                            indexed_id = None
                                            try:
                                                # Index scene (non-blocking: process_frame runs in thread pool)
                                                indexed_id = self.scene_indexer.index_scene(
                                                    scene_id=scene_id,
                                                    frame_paths=frame_paths,
                                                    summary_text=summary_result['summary'],
                                                    captions=caption_records_with_paths,
                                                    scene_start_time=self.scene_start_time,
                                                    scene_end_time=current_time,
                                                    metadata=scene_metadata
                                                )
                                                index_end = time.time()
                                                duration_ms = (index_end - index_start) * 1000.0
                                                
                                                if indexed_id:
                                                    logger.info(f"Scene indexed: {indexed_id} with {len(frame_paths)} frames")
                                                    
                                                    # Flush Langfuse after scene indexing (important event)
                                                    try:
                                                        self.langfuse_tracker.flush()
                                                        logger.debug("Langfuse events flushed (after scene indexing)")
                                                    except Exception as e:
                                                        logger.warning(f"Error flushing Langfuse after scene indexing: {e}")
                                                    
                                                    from datetime import datetime
                                                    self.langfuse_tracker.client.update_current_generation(
                                                        output=f"Indexed scene {indexed_id}",
                                                        completion_start_time=datetime.fromtimestamp(index_end),
                                                        metadata={
                                                            'frame_count': len(frame_paths),
                                                            'duration_ms': duration_ms,
                                                            'scene_id': scene_id,
                                                            'caption_count': len(caption_records_with_paths),
                                                            'recognized_faces': list(self.scene_face_names),
                                                            'chunking_enabled': self.scene_indexer.enable_chunking,
                                                            'caption_embeddings_enabled': self.scene_indexer.enable_caption_embeddings
                                                        }
                                                    )
                                                    
                                                    # Log LanceDB success score
                                                    self.langfuse_tracker.log_score(
                                                        trace_id=trace.id,
                                                        name="lancedb_success",
                                                        value=1.0,
                                                        data_type="NUMERIC",
                                                        comment=f"LanceDB scene indexing succeeded (scene_id={indexed_id})",
                                                        score_trace=False
                                                    )
                                                    
                                                    # Log LanceDB latency score (normalize 0-3s range)
                                                    latency_score = max(0.0, min(1.0, 1.0 - (duration_ms / 3000.0)))
                                                    self.langfuse_tracker.log_score(
                                                        trace_id=trace.id,
                                                        name="lancedb_latency_score",
                                                        value=latency_score,
                                                        data_type="NUMERIC",
                                                        comment=f"LanceDB indexing latency: {duration_ms:.1f}ms",
                                                        score_trace=False
                                                    )
                                                    
                                                    # Log LanceDB frame coverage score
                                                    coverage_score = min(1.0, len(frame_paths) / 10.0)
                                                    self.langfuse_tracker.log_score(
                                                        trace_id=trace.id,
                                                        name="lancedb_frame_coverage",
                                                        value=coverage_score,
                                                        data_type="NUMERIC",
                                                        comment=f"LanceDB frames indexed: {len(frame_paths)}",
                                                        score_trace=False
                                                    )
                                                else:
                                                    logger.warning("LanceDB index_scene returned None")
                                                    self.langfuse_tracker.client.update_current_generation(
                                                        output="",
                                                        completion_start_time=datetime.fromtimestamp(index_start),
                                                        metadata={
                                                            'frame_count': len(frame_paths),
                                                            'scene_id': scene_id,
                                                            'caption_count': len(caption_records_with_paths),
                                                            'error': 'index_scene returned None'
                                                        }
                                                    )
                                                    self.langfuse_tracker.log_score(
                                                        trace_id=trace.id,
                                                        name="lancedb_success",
                                                        value=0.0,
                                                        data_type="NUMERIC",
                                                        comment="LanceDB indexing failed (no scene_id returned)",
                                                        score_trace=False
                                                    )
                                            except Exception as index_error:
                                                index_end = time.time()
                                                logger.warning(f"Error indexing scene (continuing): {index_error}", exc_info=False)
                                                from datetime import datetime
                                                self.langfuse_tracker.client.update_current_generation(
                                                    output="",
                                                    completion_start_time=datetime.fromtimestamp(index_end),
                                                        metadata={
                                                            'frame_count': len(frame_paths),
                                                            'scene_id': scene_id,
                                                            'caption_count': len(caption_records_with_paths),
                                                            'error': str(index_error)
                                                        }
                                                )
                                                self.langfuse_tracker.log_score(
                                                    trace_id=trace.id,
                                                    name="lancedb_success",
                                                    value=0.0,
                                                    data_type="NUMERIC",
                                                    comment=f"LanceDB indexing error: {index_error}",
                                                    score_trace=False
                                                )
                                            finally:
                                                lancedb_generation.__exit__(None, None, None)
                                        
                                        # Reset scene frames for next scene
                                        self.scene_frames = []
                                        self.caption_records = []
                                        self.scene_start_time = None  # Start new scene on next frame
                                        self.scene_face_names.clear()
                                        
                                    except Exception as scene_error:
                                        logger.warning(f"Error indexing scene (continuing): {scene_error}", exc_info=False)
                                
                                # Log scores for LLM summarization
                                # Success score
                                self.langfuse_tracker.log_score(
                                    trace_id=trace.id,
                                    name="llm_success",
                                    value=1.0,
                                    data_type="NUMERIC",
                                    comment="LLM summary generation succeeded",
                                    score_trace=False
                                )
                                
                                # Processing time score (normalized)
                                processing_time_ms = summary_result.get('processing_time', 0)
                                # Normalize: assume 5000ms is worst, 0ms is best
                                processing_score = max(0.0, min(1.0, 1.0 - (processing_time_ms / 5000.0)))
                                self.langfuse_tracker.log_score(
                                    trace_id=trace.id,
                                    name="llm_processing_score",
                                    value=processing_score,
                                    data_type="NUMERIC",
                                    comment=f"LLM processing time: {processing_time_ms:.1f}ms",
                                    score_trace=False
                                )
                                
                                # Summary quality score (based on length and captions processed)
                                summary_length = len(summary_result['summary'])
                                captions_count = len(self.caption_queue)
                                quality_score = min(1.0, (summary_length / 200.0) * (captions_count / 10.0))
                                self.langfuse_tracker.log_score(
                                    trace_id=trace.id,
                                    name="llm_quality",
                                    value=quality_score,
                                    data_type="NUMERIC",
                                    comment=f"Summary quality: {summary_length} chars, {captions_count} captions",
                                    score_trace=False
                                )
                                
                                # Overall trace score (average of all metrics)
                                overall_score = (1.0 + processing_score + quality_score) / 3.0
                                self.langfuse_tracker.log_score(
                                    trace_id=trace.id,
                                    name="overall_quality",
                                    value=overall_score,
                                    data_type="NUMERIC",
                                    comment="Overall trace quality score",
                                    score_trace=True  # Score the entire trace
                                )
                            else:
                                # Update with error
                                self.langfuse_tracker.client.update_current_generation(
                                    output="",
                                    metadata={
                                        'error': summary_result.get('error', 'Unknown error'),
                                        'captions_count': len(self.caption_queue)
                                    }
                                )
                                
                                # Log failure score
                                self.langfuse_tracker.log_score(
                                    trace_id=trace.id,
                                    name="llm_success",
                                    value=0.0,
                                    data_type="NUMERIC",
                                    comment=f"LLM summary generation failed: {summary_result.get('error', 'Unknown error')}",
                                    score_trace=False
                                )
                        finally:
                            # Exit the generation context
                            generation_context.__exit__(None, None, None)
                
                self.frames_processed += 1
                
                return {
                    'success': True,
                    'caption': self.latest_caption,
                    'summary': self.current_summary,
                    'faces': recognized_faces
                }
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
                return {'success': False, 'error': str(e)}
    
    def create_display_frame(self, camera_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Create the display frame with captions and summary."""
        # Create black canvas
        display = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Add camera feed if available (top portion)
        if camera_frame is not None:
            # Resize camera frame to fit top half
            cam_height = self.display_height // 2
            cam_width = self.display_width
            resized_cam = cv2.resize(camera_frame, (cam_width, cam_height))
            display[0:cam_height, :] = resized_cam
        
        # Add caption section
        caption_y = self.display_height // 2 + 50
        cv2.putText(
            display,
            "Latest Caption:",
            (10, caption_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # Wrap caption text
        caption_lines = self._wrap_text(self.latest_caption, 70)
        for i, line in enumerate(caption_lines[:3]):  # Max 3 lines
            cv2.putText(
                display,
                line,
                (10, caption_y + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Add summary section
        summary_y = caption_y + 120
        cv2.putText(
            display,
            "Current Summary:",
            (10, summary_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Wrap summary text
        summary_lines = self._wrap_text(self.current_summary, 70)
        for i, line in enumerate(summary_lines[:4]):  # Max 4 lines
            cv2.putText(
                display,
                line,
                (10, summary_y + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        # Add statistics
        stats_y = self.display_height - 30
        stats_text = f"Frames: {self.frames_captured} | Processed: {self.frames_processed}"
        cv2.putText(
            display,
            stats_text,
            (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )
        
        return display
    
    def _wrap_text(self, text: str, max_chars: int) -> list:
        """Wrap text to fit within specified character width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    async def capture_loop(self):
        """Async loop for capturing frames at specified FPS."""
        last_capture_time = 0
        last_process_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Always update display and check for keys
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if ret:
                    # Store frame for scene indexing (keep last N frames for current scene)
                    # Store a sample of frames (every 5 seconds) to avoid storing too many
                    if current_time - self.last_caption_time >= self.caption_interval:
                        # Initialize scene start time on first frame
                        if self.scene_start_time is None:
                            self.scene_start_time = current_time
                        
                        # Store frame copy for scene indexing
                        self.scene_frames.append({
                            'frame': frame.copy(),
                            'timestamp': current_time,
                            'frame_index': len(self.scene_frames)
                        })
                        
                        # Limit stored frames to avoid memory issues (keep last 60 frames max)
                        if len(self.scene_frames) > 60:
                            self.scene_frames.pop(0)
                    
                    # Generate caption every 5 seconds (real-time processing)
                    if current_time - self.last_caption_time >= self.caption_interval:
                        self.frames_captured += 1
                        self.last_caption_time = current_time
                        logger.info(f"Scheduling frame {self.frames_captured} for caption generation (every {self.caption_interval}s)")
                        # Process frame in background (non-blocking)
                        task = asyncio.create_task(self.async_process_frame(frame.copy()))
                        # Don't await - let it run in background
                    
                    # Update display every frame (with latest caption/summary)
                    display_frame = self.create_display_frame(frame)
                    cv2.imshow(self.window_name, display_frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC or 'q'
                        logger.info("Exit key pressed")
                        self.running = False
                        break
                    
                    # Periodic Langfuse flush (every 30 seconds)
                    if current_time - self.last_langfuse_flush_time >= self.langfuse_flush_interval:
                        try:
                            self.langfuse_tracker.flush()
                            self.last_langfuse_flush_time = current_time
                            logger.debug("Langfuse events flushed (periodic)")
                        except Exception as e:
                            logger.warning(f"Error during periodic Langfuse flush: {e}")
                    
                    # Small delay to allow async tasks to execute
                    await asyncio.sleep(0.001)
                else:
                    logger.warning("Failed to read frame from camera")
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)
    
    async def async_process_frame(self, frame: np.ndarray):
        """Process frame asynchronously to avoid blocking."""
        try:
            logger.info("Starting async frame processing")
            # Use thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.process_frame,
                frame
            )
            if result.get('success'):
                logger.info(f"Frame processed successfully: caption='{result.get('caption', '')[:50]}...'")
            else:
                logger.warning(f"Frame processing failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error in async frame processing: {e}", exc_info=True)
    
    def display_loop(self):
        """Update display in separate thread."""
        while self.running:
            try:
                display_frame = self.create_display_frame()
                cv2.imshow(self.window_name, display_frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == 27 or key == ord('q'):
                    self.running = False
                    break
                    
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                break
            
            time.sleep(0.03)
    
    async def run(self):
        """Main run loop."""
        logger.info("Starting Video Summarization System...")
        
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return
        
        self.running = True
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        try:
            # Run capture loop
            await self.capture_loop()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down Video Summarization System...")
        
        self.running = False
        
        # Print statistics
        self.vlm_client.print_statistics()
        self.llm_summarizer.print_statistics()
        
        # Flush Langfuse
        self.langfuse_tracker.flush()
        
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        logger.info("Shutdown complete")


def signal_handler(sig, frame):
    """Handle CTRL+C gracefully."""
    logger.info("Received interrupt signal, shutting down...")
    sys.exit(0)


async def main():
    """Main entry point."""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments for camera selection
    import argparse
    parser = argparse.ArgumentParser(description='Real-Time Video Summarization System')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument('--fps', type=int, default=4, help='Frames per second to process (default: 4)')
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=os.getenv("SCENE_INDEX_MODEL"),
        help='Embedding model to use for LanceDB exports (overrides SCENE_INDEX_MODEL env)'
    )
    parser.add_argument(
        '--scene-table',
        type=str,
        default=os.getenv("SCENE_INDEX_TABLE"),
        help='Scene index table name (overrides SCENE_INDEX_TABLE env)'
    )
    args = parser.parse_args()
    
    # Create and run summarizer
    summarizer = VideoSummarizer(
        camera_index=args.camera,
        fps=args.fps,
        scene_index_model=args.embedding_model,
        scene_index_table=args.scene_table
    )
    
    await summarizer.run()


if __name__ == "__main__":
    asyncio.run(main())
