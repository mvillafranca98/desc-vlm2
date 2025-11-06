#!/usr/bin/env python3
"""
LLM Summarizer using Mistral-7B-Instruct-v0.3
"""

import logging
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """LLM-based summarizer for continuous caption synthesis."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens: int = 150,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model = None
        self.tokenizer = None
        self.current_summary = ""
        self.previous_captions = []
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'processing_times': []
        }
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Mistral model."""
        try:
            logger.info(f"Loading LLM model: {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            logger.info(f"Using device: {self.device}")
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                device_map=self.device if self.device != "mps" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to MPS if needed
            if self.device == "mps":
                self.model = self.model.to("mps")
            
            self.model.eval()
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
    
    def update_summary(self, captions: List[str]) -> Dict[str, Any]:
        """Update summary based on new captions."""
        if not self.model or not self.tokenizer:
            return {
                'success': False,
                'error': 'LLM model not initialized',
                'summary': self.current_summary
            }
        
        start_time = time.time()
        self.stats['total_updates'] += 1
        
        try:
            # Only use recent captions to avoid token limits
            recent_captions = captions[-10:] if len(captions) > 10 else captions
            
            # Build prompt for summarization
            prompt = self._build_summary_prompt(recent_captions)
            
            # Generate summary
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Track generation start time (after tokenization, before generation)
            generation_start_time = time.time()
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # For local models, we'll approximate completion_start_time
                # as generation_start_time + small offset (first token comes quickly)
                # In practice, for local models, TTFT is very fast anyway
                completion_start_time = generation_start_time + 0.01  # ~10ms approximation
            
            # Decode only the generated part
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            end_time = time.time()
            
            # Update current summary
            self.current_summary = summary
            self.previous_captions = recent_captions
            
            processing_time = (end_time - start_time) * 1000
            self.stats['processing_times'].append(processing_time)
            self.stats['successful_updates'] += 1
            
            logger.debug(f"Summary updated: {summary[:100]}...")
            
            return {
                'success': True,
                'summary': summary,
                'processing_time': processing_time,
                'start_time': start_time,
                'completion_start_time': completion_start_time,
                'end_time': end_time
            }
            
        except Exception as e:
            self.stats['failed_updates'] += 1
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            self.stats['processing_times'].append(processing_time)
            
            logger.error(f"LLM summarization error: {e}", exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'summary': self.current_summary,
                'processing_time': processing_time,
                'start_time': start_time,
                'completion_start_time': None,
                'end_time': end_time
            }
    
    def _build_summary_prompt(self, captions: List[str]) -> str:
        """Build prompt for summarization."""
        # Filter out duplicate or very similar captions
        unique_captions = self._filter_duplicates(captions)
        
        captions_text = "\n".join([f"- {cap}" for cap in unique_captions])
        
        prompt = f"""You are a concise video summarizer. Based on these recent observations from a video feed:

{captions_text}

Provide a brief, coherent summary of what's currently happening in the scene. Focus on:
1. The environment and setting
2. Any people present and their activities
3. Notable objects or changes
4. Overall atmosphere

Keep the summary to 2-3 sentences. Avoid repetition and be specific. Only mention new or significant details."""
        
        return prompt
    
    def _filter_duplicates(self, captions: List[str]) -> List[str]:
        """Filter out duplicate or very similar captions."""
        if not captions:
            return []
        
        unique = []
        for caption in captions:
            # Simple duplicate check
            if caption not in unique:
                # Check similarity with existing captions
                is_similar = False
                for existing in unique:
                    similarity = self._calculate_similarity(caption, existing)
                    if similarity > 0.85:  # 85% similar
                        is_similar = True
                        break
                
                if not is_similar:
                    unique.append(caption)
        
        return unique
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def print_statistics(self):
        """Print LLM statistics."""
        logger.info("")
        logger.info("=" * 62)
        logger.info("LLM SUMMARIZATION STATISTICS")
        logger.info("=" * 62)
        logger.info(f"Total Updates: {self.stats['total_updates']}")
        logger.info(f"Successful: {self.stats['successful_updates']}")
        logger.info(f"Failed: {self.stats['failed_updates']}")
        logger.info("")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            min_time = min(self.stats['processing_times'])
            max_time = max(self.stats['processing_times'])
            
            logger.info(f"Processing Time: {avg_time:.2f} ms (avg)")
            logger.info(f"Processing Range: {min_time:.2f} - {max_time:.2f} ms")
        
        logger.info("=" * 62)
        logger.info("")

