#!/usr/bin/env python3
"""
VLM Client Wrapper with Statistics Tracking
"""

import base64
import logging
import time
from typing import Dict, Any, Optional
from io import BytesIO
import numpy as np
from PIL import Image
from openai import OpenAI

logger = logging.getLogger(__name__)


class VLMClientWrapper:
    """Wrapper for VLM API with statistics tracking."""
    
    def __init__(
        self,
        url: str = "http://38.80.152.249:30447/v1",
        model: str = "qwen3",
        api_key: str = "dummy",
        max_tokens: int = 128,
        temperature: float = 0.2
    ):
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key, base_url=url)
            logger.info(f"VLM client initialized: {model} at {url}")
        except Exception as e:
            logger.error(f"Failed to initialize VLM client: {e}")
            self.client = None
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'image_processing_times': [],
            'api_request_times': [],
            'total_times': []
        }
    
    def generate_caption(self, frame: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Generate caption for a single frame."""
        if not self.client:
            return {
                'success': False,
                'error': 'VLM client not initialized',
                'caption': ''
            }
        
        total_start = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # Convert frame to base64
            img_start = time.time()
            base64_image = self._frame_to_base64(frame)
            img_time = (time.time() - img_start) * 1000  # ms
            self.stats['image_processing_times'].append(img_time)
            
            if not base64_image:
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': 'Failed to encode image',
                    'caption': ''
                }
            
            # Build message content
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
            
            # Make API call
            api_start = time.time()
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # Track completion start time (TTFT approximation)
            completion_start_time = time.time()
            
            api_time = (time.time() - api_start) * 1000  # ms
            self.stats['api_request_times'].append(api_time)
            
            # Extract caption
            caption = response.choices[0].message.content.strip()
            
            # Parse if JSON format
            caption = self._parse_caption(caption)
            
            end_time = time.time()
            total_time = (end_time - total_start) * 1000  # ms
            self.stats['total_times'].append(total_time)
            self.stats['successful_requests'] += 1
            
            logger.debug(f"VLM caption generated: {caption[:100]}...")
            
            return {
                'success': True,
                'caption': caption,
                'api_time': api_time,
                'start_time': start_time,
                'completion_start_time': completion_start_time,
                'end_time': end_time
            }
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            total_time = (time.time() - total_start) * 1000
            self.stats['total_times'].append(total_time)
            
            logger.error(f"VLM API error: {e}", exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'caption': '',
                'api_time': 0
            }
    
    def _frame_to_base64(self, frame: np.ndarray) -> Optional[str]:
        """Convert frame to base64 JPEG."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize if too large (for faster processing)
            max_size = 1024
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to JPEG bytes
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            img_bytes = buffered.getvalue()
            
            # Encode to base64
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{img_b64}"
            
        except Exception as e:
            logger.error(f"Failed to convert frame to base64: {e}")
            return None
    
    def _parse_caption(self, caption: str) -> str:
        """Parse caption from potential JSON response."""
        try:
            # Try to parse as JSON
            import json
            
            # Remove markdown code blocks
            if caption.startswith("```"):
                lines = caption.split("\n")
                caption = "\n".join([l for l in lines if not l.strip().startswith("```")])
            
            # Try to find JSON object
            if "{" in caption and "}" in caption:
                start = caption.find("{")
                end = caption.rfind("}") + 1
                json_str = caption[start:end]
                
                parsed = json.loads(json_str)
                
                # Extract description from common field names
                for field in ['description', 'caption', 'text', 'content', 'response']:
                    if field in parsed:
                        return str(parsed[field])
            
            # Return as-is if not JSON
            return caption
            
        except Exception:
            return caption
    
    def print_statistics(self):
        """Print comprehensive statistics."""
        logger.info("")
        logger.info("=" * 62)
        logger.info("VLM API STATISTICS")
        logger.info("=" * 62)
        logger.info(f"Total Requests: {self.stats['total_requests']}")
        
        success_pct = (
            (self.stats['successful_requests'] / self.stats['total_requests'] * 100)
            if self.stats['total_requests'] > 0 else 0
        )
        logger.info(f"Successful: {self.stats['successful_requests']} ({success_pct:.1f}%)")
        logger.info(f"Failed: {self.stats['failed_requests']}")
        logger.info("")
        logger.info("Timing (ms):")
        
        # Calculate averages
        avg_img = sum(self.stats['image_processing_times']) / len(self.stats['image_processing_times']) if self.stats['image_processing_times'] else 0
        avg_api = sum(self.stats['api_request_times']) / len(self.stats['api_request_times']) if self.stats['api_request_times'] else 0
        avg_total = sum(self.stats['total_times']) / len(self.stats['total_times']) if self.stats['total_times'] else 0
        
        min_api = min(self.stats['api_request_times']) if self.stats['api_request_times'] else 0
        max_api = max(self.stats['api_request_times']) if self.stats['api_request_times'] else 0
        
        logger.info(f"  Image Processing: {avg_img:.2f} ms (avg)")
        logger.info(f"  API Request:      {avg_api:.2f} ms (avg)")
        logger.info(f"  Total Time:       {avg_total:.2f} ms (avg)")
        logger.info(f"  API Request Range: {min_api:.2f} - {max_api:.2f} ms")
        logger.info("=" * 62)
        logger.info("")


# Import cv2 here to avoid circular imports
import cv2

