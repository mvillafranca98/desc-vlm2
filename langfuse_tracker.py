#!/usr/bin/env python3
"""
Langfuse Integration for Telemetry Tracking - Simplified Working Version
"""

import logging
import os
import time
from typing import Optional, Dict, Any, Union
import uuid

logger = logging.getLogger(__name__)

# Try to import Langfuse
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.warning("Langfuse library not available")
    LANGFUSE_AVAILABLE = False


class LangfuseTracker:
    """Langfuse tracker using simplified API."""
    
    def __init__(self):
        self.enabled = False
        self.client = None
        
        if not LANGFUSE_AVAILABLE:
            logger.warning("Langfuse tracking disabled (library not available)")
            return
        
        # Get credentials from environment
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        
        if not public_key or not secret_key:
            logger.warning("Langfuse credentials not found in environment")
            logger.info("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracking")
            return
        
        try:
            self.client = Langfuse(
                host=host,
                public_key=public_key,
                secret_key=secret_key
            )
            self.enabled = True
            logger.info(f"Langfuse tracking enabled: {host}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self.enabled = False
    
    def create_trace(self, name: str, metadata: Optional[Dict] = None) -> Any:
        """Create a trace context manager."""
        if not self.enabled:
            return DummyTrace()
        
        return TraceContext(self.client, name, metadata or {})
    
    def create_span(
        self,
        trace_id: str,
        name: str,
        metadata: Optional[Dict] = None
    ) -> Any:
        """Create a span context manager."""
        if not self.enabled:
            return DummySpan()
        
        return SpanContext(self.client, name, metadata or {})
    
    def end_span(self, span: Any):
        """End a span."""
        pass
    
    def start_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: str,
        metadata: Optional[Dict] = None
    ) -> Any:
        """Start a generation BEFORE the API call. Returns a context manager that should be updated after."""
        if not self.enabled:
            return DummyGeneration()
        
        try:
            from datetime import datetime
            
            # Prepare metadata
            gen_metadata = metadata or {}
            
            # Create generation BEFORE API call - this ensures correct start_time
            generation_params = {
                "name": name,
                "as_type": "generation",
                "model": model,
                "input": input[:1000] if input else "",
                "model_parameters": {"temperature": 0.2},
                "metadata": gen_metadata
            }
            
            # Return the generation context manager
            return self.client.start_as_current_observation(**generation_params)
            
        except Exception as e:
            logger.error(f"Error starting generation in Langfuse: {e}", exc_info=False)
            return DummyGeneration()
    
    def log_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: str,
        output: str,
        metadata: Optional[Dict] = None,
        start_time: Optional[float] = None,
        completion_start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Log a generation event using Langfuse API (creates generation after API call - not ideal for TTFT)."""
        if not self.enabled:
            return
        
        try:
            from datetime import datetime
            
            # Prepare metadata
            gen_metadata = metadata or {}
            if start_time and completion_start_time:
                ttft_ms = (completion_start_time - start_time) * 1000
                gen_metadata["ttft_ms"] = ttft_ms
            
            # Create generation - it will be a child of the current span
            generation_params = {
                "name": name,
                "as_type": "generation",
                "model": model,
                "input": input[:1000] if input else "",
                "output": output[:1000] if output else "",
                "model_parameters": {"temperature": 0.2},
                "metadata": gen_metadata
            }
            
            # Set completion_start_time for TTFT calculation
            if completion_start_time:
                generation_params["completion_start_time"] = datetime.fromtimestamp(completion_start_time)
            
            with self.client.start_as_current_observation(**generation_params):
                # Update with completion_start_time to ensure it's set
                if completion_start_time:
                    self.client.update_current_generation(
                        completion_start_time=datetime.fromtimestamp(completion_start_time)
                    )
                
        except Exception as e:
            logger.error(f"Error logging generation to Langfuse: {e}", exc_info=False)
    
    def log_score(
        self,
        trace_id: str,
        observation_id: Optional[str] = None,
        name: str = "score",
        value: Union[float, str, bool] = 0.0,
        data_type: Optional[str] = None,
        comment: Optional[str] = None,
        score_trace: bool = False
    ):
        """Log a score to Langfuse for display on home page dashboard.
        
        Args:
            trace_id: The trace ID to associate the score with
            observation_id: Optional observation ID (for observation-level scores)
            name: Name of the score (e.g., "accuracy", "latency_score", "quality")
            value: Score value (float for NUMERIC, bool for BOOLEAN, str for CATEGORICAL)
            data_type: Type of score - "NUMERIC", "BOOLEAN", or "CATEGORICAL" (auto-detected if None)
            comment: Optional comment explaining the score
            score_trace: If True, score the entire trace; if False, score the observation
        """
        if not self.enabled:
            return
        
        try:
            # Auto-detect data type if not provided
            if data_type is None:
                if isinstance(value, bool):
                    data_type = "BOOLEAN"
                    value = 1.0 if value else 0.0
                elif isinstance(value, str):
                    data_type = "CATEGORICAL"
                else:
                    data_type = "NUMERIC"
            
            # Convert value to appropriate type for API
            if data_type == "NUMERIC" or data_type == "BOOLEAN":
                # Numeric and boolean scores should be float
                score_value: Union[float, str] = float(value)
            else:
                # Categorical scores should be string
                score_value = str(value)
            
            # Create score using Langfuse client
            self.client.create_score(
                name=name,
                value=score_value,
                trace_id=trace_id,
                observation_id=observation_id if not score_trace else None,
                data_type=data_type,
                comment=comment
            )
            
            logger.debug(f"Score logged: {name}={value} (type={data_type})")
            
        except Exception as e:
            logger.error(f"Error logging score to Langfuse: {e}", exc_info=False)
    
    def flush(self):
        """Flush pending events to Langfuse."""
        if not self.enabled:
            return
        
        try:
            self.client.flush()
            logger.debug("Langfuse events flushed")
        except Exception as e:
            logger.error(f"Error flushing Langfuse: {e}")


class TraceContext:
    """Context manager for Langfuse traces."""
    def __init__(self, client, name: str, metadata: Dict):
        self.client = client
        self.name = name
        self.metadata = metadata
        self.trace = None
        self.id = str(uuid.uuid4())
    
    def __enter__(self):
        try:
            self.trace = self.client.start_as_current_span(
                name=self.name,
                metadata=self.metadata
            )
            self.trace.__enter__()
            return self
        except Exception as e:
            logger.error(f"Error creating trace: {e}")
            return DummyTrace()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace:
            try:
                self.trace.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass


class SpanContext:
    """Context manager for Langfuse spans."""
    def __init__(self, client, name: str, metadata: Dict):
        self.client = client
        self.name = name
        self.metadata = metadata
        self.span = None
    
    def __enter__(self):
        try:
            self.span = self.client.start_as_current_span(
                name=self.name,
                metadata=self.metadata
            )
            self.span.__enter__()
            return self
        except Exception as e:
            logger.error(f"Error creating span: {e}")
            return DummySpan()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            try:
                self.span.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass


class DummyTrace:
    """Dummy trace for when Langfuse is disabled."""
    def __init__(self):
        self.id = "dummy"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummySpan:
    """Dummy span for when Langfuse is disabled."""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyGeneration:
    """Dummy generation for when Langfuse is disabled."""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def update(self, **kwargs):
        """Dummy update method."""
        pass
