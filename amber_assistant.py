#!/usr/bin/env python3
"""
Amber - Video Scene Search Assistant

A GUI assistant for querying indexed video scenes with natural language.
Supports time-based queries and displays results with frames.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading

from scene_indexer import SceneIndexer
from search_scenes import calculate_relevance_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeQueryParser:
    """Parse natural language time expressions."""
    
    @staticmethod
    def parse_time_query(query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract time constraints from query.
        
        Returns:
            (start_time, end_time) tuple, or (None, None) if no time constraint
        """
        query_lower = query.lower()
        now = datetime.now()
        
        # Patterns for time expressions
        patterns = {
            'yesterday': (now - timedelta(days=1), now),
            'today': (now.replace(hour=0, minute=0, second=0), now),
            'past 3 days': (now - timedelta(days=3), now),
            'past week': (now - timedelta(days=7), now),
            'past month': (now - timedelta(days=30), now),
            'last 3 days': (now - timedelta(days=3), now),
            'last week': (now - timedelta(days=7), now),
            'last month': (now - timedelta(days=30), now),
        }
        
        # Check for patterns
        for pattern, (start, end) in patterns.items():
            if pattern in query_lower:
                return start, end
        
        # Check for "X days ago" pattern
        days_ago_match = re.search(r'(\d+)\s+days?\s+ago', query_lower)
        if days_ago_match:
            days = int(days_ago_match.group(1))
            start = now - timedelta(days=days)
            return start, now
        
        # Check for "X weeks ago" pattern - return specific week range, not range until now
        weeks_ago_match = re.search(r'(\d+)\s+weeks?\s+ago', query_lower)
        if weeks_ago_match:
            weeks = int(weeks_ago_match.group(1))
            # Calculate the start of that week (Monday)
            target_date = now - timedelta(weeks=weeks)
            # Get Monday of that week
            days_since_monday = target_date.weekday()
            week_start = target_date - timedelta(days=days_since_monday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            # End of that week (Sunday 23:59:59)
            week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return week_start, week_end
        
        # Check for "two weeks ago" / "2 weeks ago" (textual)
        if 'two weeks ago' in query_lower or '2 weeks ago' in query_lower:
            # Calculate the start of that week (Monday)
            target_date = now - timedelta(weeks=2)
            days_since_monday = target_date.weekday()
            week_start = target_date - timedelta(days=days_since_monday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            # End of that week (Sunday 23:59:59)
            week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return week_start, week_end
        
        return None, None
    
    @staticmethod
    def extract_search_terms(query: str) -> str:
        """Remove time expressions from query to get search terms."""
        query_lower = query.lower()
        
        # Remove common time expressions
        time_phrases = [
            'yesterday', 'today', 'past 3 days', 'past week', 'past month',
            'last 3 days', 'last week', 'last month',
            r'\d+\s+days?\s+ago',
            r'\d+\s+weeks?\s+ago',
            'two weeks ago', '2 weeks ago',
            'in the past', 'recently'
        ]
        
        # Remove greetings
        greetings = ['hi amber', 'hello amber', 'hey amber', 'hi', 'hello', 'hey']
        for greeting in greetings:
            if query_lower.startswith(greeting):
                query_lower = query_lower[len(greeting):].strip()
                # Remove comma or other punctuation after greeting
                query_lower = query_lower.lstrip(',.?!').strip()
                break
        
        cleaned = query_lower
        for phrase in time_phrases:
            cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE)
        
        # Remove common question words and phrases that don't help search
        question_words = [
            r'\bdo you have\b',
            r'\bany records\b',
            r'\bany record\b',
            r'\brecords of\b',
            r'\brecord of\b',
            r'\bfrom\b',
            r'\bon\b',
            r'\bthe\b',
            r'\ba\b',
            r'\ban\b',
        ]
        for word in question_words:
            cleaned = re.sub(word, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        cleaned = ' '.join(cleaned.split())
        cleaned = cleaned.strip(' ,.?!')
        return cleaned.strip()


class AmberAssistant:
    """Main GUI application for Amber assistant."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Amber - Video Scene Search Assistant")
        self.root.geometry("1200x800")
        
        # Initialize scene indexer
        self.indexer = None
        self._initialize_indexer()
        
        # UI components
        self.setup_ui()
        
        # Check and display system status
        self._check_system_status()
        
        # Welcome message
        self.add_message("Amber", "Hello! I'm Amber, your video scene search assistant. "
                                   "Ask me questions like:\n"
                                   "- 'Do you have any records of hyenas in the past 3 days?'\n"
                                   "- 'Are there any records of lions yesterday?'\n"
                                   "- 'Show me scenes with blue aprons'", "system")
    
    def _initialize_indexer(self):
        """Initialize the scene indexer."""
        try:
            from scene_indexer import get_table_name_for_model
            
            # Get embedding model from environment or use default
            embedding_model = os.getenv("SCENE_INDEX_MODEL", "text-embedding-3-small")
            table_name = os.getenv("SCENE_INDEX_TABLE")
            
            # Auto-generate table name if not explicitly set
            if not table_name:
                table_name = get_table_name_for_model(embedding_model)
                logger.info(f"Auto-generated table name: {table_name} for model: {embedding_model}")
            
            self.indexer = SceneIndexer(
                table_name=table_name,
                embedding_model=embedding_model
            )
            if not self.indexer.enabled:
                self.add_message("Amber", "⚠️ Scene indexer is not enabled. Please check LanceDB configuration.", "error")
            else:
                logger.info(f"Amber initialized with model: {embedding_model}, table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}", exc_info=True)
            self.add_message("Amber", f"⚠️ Error initializing scene indexer: {e}", "error")
    
    def _check_system_status(self):
        """Check and display status of all integrations."""
        status_lines = []
        status_lines.append("🔍 System Status Check:")
        status_lines.append("")
        
        # Check LanceDB
        status_lines.append("📊 LanceDB Integration:")
        if self.indexer and self.indexer.enabled:
            try:
                table_name = self.indexer.table_name
                project_slug = os.getenv("LANCEDB_PROJECT_SLUG", "Not set")
                region = os.getenv("LANCEDB_REGION", "Not set")
                
                status_lines.append(f"   ✅ Connection: Active (LanceDB Cloud)")
                status_lines.append(f"   📁 Project: {project_slug}")
                status_lines.append(f"   🌍 Region: {region}")
                status_lines.append(f"   📋 Table: {table_name}")
                
                # Check table and row count
                if self.indexer.table:
                    try:
                        row_count = self.indexer.table.count_rows()
                        status_lines.append(f"   📊 Records: {row_count} indexed scenes")
                    except Exception as e:
                        status_lines.append(f"   ⚠️  Records: Unable to count ({str(e)[:50]})")
                else:
                    status_lines.append(f"   ⚠️  Table: Not accessible")
                
                # Check embedding model (works for both local and OpenAI models)
                if self.indexer.use_openai_api or self.indexer.embedding_model:
                    model_name = self.indexer.embedding_model_name
                    model_type = "OpenAI API" if self.indexer.use_openai_api else "Local"
                    status_lines.append(f"   🤖 Embedding Model: {model_name}")
                    status_lines.append(f"   📦 Model Type: {model_type}")
                    status_lines.append(f"   📏 Dimension: {self.indexer.embedding_dim}")
                    status_lines.append(f"   ✅ Model Status: Loaded and ready")
                else:
                    status_lines.append(f"   ⚠️  Embedding Model: Not loaded")
                    status_lines.append(f"   ℹ️  Model will be loaded when first query is made")
                    
            except Exception as e:
                status_lines.append(f"   ❌ Error checking LanceDB: {str(e)[:100]}")
        else:
            status_lines.append("   ❌ Connection: Not available")
            status_lines.append("   ⚠️  Check LANCEDB_PROJECT_SLUG and LANCEDB_API_KEY")
        
        status_lines.append("")
        
        # Check Langfuse
        status_lines.append("📈 Langfuse Integration:")
        langfuse_host = os.getenv("LANGFUSE_HOST", "")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        
        if langfuse_host and langfuse_public_key and langfuse_secret_key:
            status_lines.append(f"   ✅ Connection: Configured")
            status_lines.append(f"   🌐 Host: {langfuse_host}")
            status_lines.append(f"   🔑 Public Key: {langfuse_public_key[:20]}...")
            status_lines.append(f"   ℹ️  Note: Langfuse is used in real_time_summarizer.py for telemetry")
            status_lines.append(f"   ℹ️  Data export: Traces, spans, generations, and scores are logged")
        else:
            status_lines.append("   ⚠️  Connection: Not configured")
            status_lines.append("   ℹ️  Langfuse is optional (used for observability in main app)")
        
        status_lines.append("")
        
        # Check models used in main app
        status_lines.append("🎯 Models Used in Main Application:")
        vlm_model = os.getenv("VLM_MODEL", "qwen3")
        llm_model = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        
        status_lines.append(f"   🖼️  VLM Model: {vlm_model}")
        status_lines.append(f"   📝 LLM Model: {llm_model}")
        status_lines.append(f"   🔍 Embedding Model: {self.indexer.embedding_model_name if self.indexer and self.indexer.enabled else 'N/A'}")
        status_lines.append(f"   ℹ️  Note: Models are loaded in real_time_summarizer.py")
        status_lines.append(f"   ℹ️  Amber uses the embedding model for semantic search")
        
        status_lines.append("")
        status_lines.append("✅ Status check complete. Ready to answer questions!")
        
        # Display status (but wait for UI to be ready)
        self.root.after(100, lambda: self.add_message("Amber", "\n".join(status_lines), "system"))
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Chat area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Arial", 11),
            state=tk.DISABLED
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for styling
        self.chat_display.tag_config("user", foreground="blue", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("amber", foreground="green", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("system", foreground="gray", font=("Arial", 10, "italic"))
        self.chat_display.tag_config("error", foreground="red", font=("Arial", 11))
        self.chat_display.tag_config("stats", foreground="purple", font=("Arial", 10, "bold"))
        
        # Input area
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        self.query_entry = ttk.Entry(input_frame, font=("Arial", 11))
        self.query_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.query_entry.bind("<Return>", lambda e: self.handle_query())
        
        self.send_button = ttk.Button(
            input_frame,
            text="Ask Amber",
            command=self.handle_query
        )
        self.send_button.grid(row=0, column=1)
        
        # Results frame (for images)
        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.results_frame.columnconfigure(0, weight=1)
    
    def add_message(self, sender: str, message: str, tag: str = "normal"):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n", "normal")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def handle_query(self):
        """Handle user query."""
        query = self.query_entry.get().strip()
        if not query:
            return
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Display user query
        self.add_message("You", query, "user")
        
        # Disable input during processing
        self.query_entry.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.add_message("Amber", "Searching...", "amber")
        
        # Process query in background thread
        thread = threading.Thread(target=self.process_query, args=(query,))
        thread.daemon = True
        thread.start()
    
    def process_query(self, query: str):
        """Process query in background thread."""
        query_start_time = time.time()
        try:
            if not self.indexer or not self.indexer.enabled:
                query_end_time = time.time()
                processing_time_ms = (query_end_time - query_start_time) * 1000.0
                self.root.after(0, lambda: self.add_message(
                    "Amber", f"⚠️ Scene indexer is not available. (thought for {processing_time_ms:.2f} ms)", "error"
                ))
                self.root.after(0, self._reenable_input)
                return
            
            # Parse time constraints
            parser = TimeQueryParser()
            start_time, end_time = parser.parse_time_query(query)
            search_terms = parser.extract_search_terms(query)
            
            if not search_terms:
                search_terms = query  # Fallback to original query
            
            # Perform search
            all_results = self.indexer.query_scenes(
                search_terms,
                limit=20,  # Get more results to filter by time and for fallback
                min_similarity=0.0
            )
            
            # Filter by time if specified
            exact_period_results = []
            nearby_period_results = []
            using_nearby_period = False
            
            if start_time and end_time:
                # First, try exact time period
                for result in all_results:
                    timestamp = result.get("timestamp")
                    if timestamp:
                        result_time = datetime.fromtimestamp(timestamp)
                        if start_time <= result_time <= end_time:
                            exact_period_results.append(result)
                
                # If no results in exact period, search nearby periods (±1 week)
                if not exact_period_results:
                    nearby_start = start_time - timedelta(weeks=1)
                    nearby_end = end_time + timedelta(weeks=1)
                    for result in all_results:
                        timestamp = result.get("timestamp")
                        if timestamp:
                            result_time = datetime.fromtimestamp(timestamp)
                            if nearby_start <= result_time <= nearby_end:
                                nearby_period_results.append(result)
                    if nearby_period_results:
                        using_nearby_period = True
                        results = nearby_period_results
                    else:
                        results = []
                else:
                    results = exact_period_results
            else:
                results = all_results
            
            # Calculate scores first (needed for flexible filtering)
            scored_results = []
            for result in results:
                score_info = calculate_relevance_score(result, search_terms)
                scored_result = {
                    **result,
                    "relevance_score": score_info["total_score"],
                    "scoring_breakdown": score_info
                }
                scored_results.append(scored_result)
            
            # Apply flexible keyword filtering: allow high vector similarity even without keywords
            # This enables semantic matches like "man playing piano" matching "male human performing art"
            filtered_results = self._filter_by_keywords_flexible(scored_results, search_terms)
            
            # Sort by relevance score
            filtered_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Limit to top results
            top_results = filtered_results[:5]
            
            # Calculate processing time
            query_end_time = time.time()
            processing_time_ms = (query_end_time - query_start_time) * 1000.0
            
            # Update UI in main thread
            self.root.after(0, lambda: self.display_results(
                query, search_terms, start_time, end_time, top_results, using_nearby_period, processing_time_ms
            ))
            
        except Exception as e:
            query_end_time = time.time()
            processing_time_ms = (query_end_time - query_start_time) * 1000.0
            logger.error(f"Error processing query: {e}", exc_info=True)
            self.root.after(0, lambda: self.add_message(
                "Amber", f"⚠️ Error processing query: {e} (thought for {processing_time_ms:.2f} ms)", "error"
            ))
            self.root.after(0, self._reenable_input)
    
    def display_results(
        self,
        original_query: str,
        search_terms: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        results: List[Dict[str, Any]],
        using_nearby_period: bool = False,
        processing_time_ms: float = 0.0
    ):
        """Display search results in the UI."""
        # Remove "Searching..." message
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("end-2l", "end-1l")
        self.chat_display.config(state=tk.DISABLED)
        
        if not results:
            period_info = ""
            if start_time and end_time:
                period_info = f" between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')}"
            self.add_message(
                "Amber",
                f"❌ No records found for '{search_terms}'{period_info}\n"
                f"💭 Thought for {processing_time_ms:.2f} ms",
                "amber"
            )
            self._reenable_input()
            return
        
        # Display message if using nearby period
        if using_nearby_period and start_time and end_time:
            nearby_message = (
                f"ℹ️ There are no records for '{search_terms}' during the period "
                f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}, "
                f"but here are some records I found from time periods close to that period:"
            )
            self.add_message("Amber", nearby_message, "amber")
        
        # Display stats
        time_info = ""
        if start_time and end_time and not using_nearby_period:
            time_info = f" (between {start_time.strftime('%Y-%m-%d')} and {end_time.strftime('%Y-%m-%d')})"
        elif using_nearby_period:
            # Show the actual time range of the nearby results
            if results:
                result_times = [datetime.fromtimestamp(r.get("timestamp", 0)) for r in results if r.get("timestamp")]
                if result_times:
                    min_time = min(result_times)
                    max_time = max(result_times)
                    time_info = f" (from nearby periods: {min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')})"
        
        avg_score = sum(r["relevance_score"] for r in results) / len(results)
        max_score = max(r["relevance_score"] for r in results)
        
        stats_message = (
            f"📊 Found {len(results)} record(s) for '{search_terms}'{time_info}\n"
            f"   • Average Relevance Score: {avg_score:.3f}\n"
            f"   • Highest Score: {max_score:.3f}\n"
            f"   • Showing top {min(3, len(results))} results with frames\n"
            f"💭 Thought for {processing_time_ms:.2f} ms"
        )
        self.add_message("Amber", stats_message, "stats")
        
        # Display top results with frames
        display_count = min(3, len(results))
        for i, result in enumerate(results[:display_count], 1):
            self._display_result(i, result, i-1)  # Pass row index for grid
        
        self._reenable_input()
    
    def _filter_by_keywords_flexible(self, results: List[Dict[str, Any]], search_terms: str) -> List[Dict[str, Any]]:
        """Flexible keyword filtering that allows semantic matches.
        
        Strategy:
        - High vector similarity (> 0.3): Allow through even without keyword matches (semantic understanding)
        - Low vector similarity (< 0.3): Require keyword matches to reduce false positives
        
        This enables queries like "male human performing art" to match "man playing piano"
        when the vector embeddings are semantically similar.
        
        Args:
            results: List of scored search result dictionaries (must have 'similarity' and 'relevance_score')
            search_terms: Search query terms
            
        Returns:
            Filtered list of results
        """
        if not search_terms or not results:
            return results
        
        # Extract important keywords (remove common stop words)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'on', 'in', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
            'any', 'some', 'all', 'no', 'not', 'only', 'just', 'also', 'very', 'too'
        }
        
        words = search_terms.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stop_words and len(w.strip('.,!?;:')) > 2]
        
        if not keywords:
            keywords = [w.strip('.,!?;:') for w in words if len(w.strip('.,!?;:')) > 2]
        
        if not keywords:
            # No keywords to filter by, return all results
            return results
        
        # Threshold for semantic matching: if vector similarity is above this, allow without keywords
        SEMANTIC_SIMILARITY_THRESHOLD = 0.3
        
        filtered = []
        for result in results:
            vector_similarity = result.get("similarity", 0.0)
            distance = result.get("distance")  # Raw distance from LanceDB (0-2 range for cosine)
            
            text = (result.get("summary_text") or "").lower()
            
            # Use distance for semantic matching (more reliable than capped similarity)
            # Cosine distance: 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite
            # For semantic matching, allow distance < 1.3 (similarity > -0.3)
            # This enables "man playing piano" to match "male human performing art"
            SEMANTIC_DISTANCE_THRESHOLD = 1.3  # Allow results with distance < 1.3
            if distance is not None and distance < SEMANTIC_DISTANCE_THRESHOLD:
                # High enough semantic similarity - allow through without keyword requirement
                filtered.append(result)
                continue
            elif distance is None and vector_similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                # Fallback: use capped similarity if distance not available
                filtered.append(result)
                continue
            
            # For low similarity results, require keyword matching to reduce false positives
            matches = 0
            for keyword in keywords:
                if keyword in text:
                    matches += 1
                    continue
                # Check variations
                if keyword + 's' in text or keyword + 'es' in text:
                    matches += 1
                    continue
                if keyword + 'ing' in text or (keyword.endswith('ing') and keyword[:-3] in text):
                    matches += 1
                    continue
                if keyword + 'ed' in text or (keyword.endswith('ed') and keyword[:-2] in text):
                    matches += 1
                    continue
                if ' ' in keyword:
                    keyword_parts = keyword.split()
                    if all(part in text for part in keyword_parts):
                        matches += 1
                        continue
            
            # Require at least 50% of keywords to match for low similarity results
            min_matches = max(1, len(keywords) // 2) if len(keywords) > 2 else 1
            
            # For queries with 3+ keywords, require the longest keyword (main object) to be present
            # Only apply this strict check for low similarity results (distance >= threshold)
            if len(keywords) >= 3 and (distance is None or distance >= SEMANTIC_DISTANCE_THRESHOLD) and vector_similarity < SEMANTIC_SIMILARITY_THRESHOLD:
                sorted_keywords = sorted(keywords, key=len, reverse=True)
                longest_keyword = sorted_keywords[0]
                longest_matched = (
                    longest_keyword in text or
                    longest_keyword + 's' in text or
                    longest_keyword + 'es' in text
                )
                if not longest_matched:
                    continue  # Skip if main object not found (for low similarity only)
            
            if matches >= min_matches:
                filtered.append(result)
        
        return filtered
    
    def _filter_by_keywords(self, results: List[Dict[str, Any]], search_terms: str) -> List[Dict[str, Any]]:
        """Filter results to only include those that contain the query keywords.
        
        Args:
            results: List of search result dictionaries
            search_terms: Search query terms
            
        Returns:
            Filtered list of results that contain the keywords
        """
        if not search_terms or not results:
            return results
        
        # Extract important keywords (remove common stop words)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'on', 'in', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
            'any', 'some', 'all', 'no', 'not', 'only', 'just', 'also', 'very', 'too'
        }
        
        # Split search terms into words and filter out stop words
        words = search_terms.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stop_words and len(w.strip('.,!?;:')) > 2]
        
        # If no meaningful keywords after filtering, use original words (but still filter very short words)
        if not keywords:
            keywords = [w.strip('.,!?;:') for w in words if len(w.strip('.,!?;:')) > 2]
        
        if not keywords:
            # If still no keywords, return all results (very short query)
            return results
        
        # Filter results that contain at least one keyword
        filtered = []
        for result in results:
            text = (result.get("summary_text") or "").lower()
            if not text:
                continue
            
            # Check if text contains any of the keywords (with word boundary matching)
            # Also check for variations (plural, -ing forms, etc.)
            matches = 0
            for keyword in keywords:
                # Direct match
                if keyword in text:
                    matches += 1
                    continue
                
                # Check for variations (simple stemming)
                # Plural forms
                if keyword + 's' in text or keyword + 'es' in text:
                    matches += 1
                    continue
                
                # -ing forms
                if keyword.endswith('ing') and keyword[:-3] in text:
                    matches += 1
                    continue
                elif keyword + 'ing' in text:
                    matches += 1
                    continue
                
                # Past tense forms
                if keyword.endswith('ed') and keyword[:-2] in text:
                    matches += 1
                    continue
                elif keyword + 'ed' in text:
                    matches += 1
                    continue
                
                # Check for compound words (e.g., "bicycle" in "bicycle riding")
                # Split compound phrases and check individual words
                if ' ' in keyword:
                    keyword_parts = keyword.split()
                    if all(part in text for part in keyword_parts):
                        matches += 1
                        continue
            
            # Require at least 50% of keywords to match (or at least 1 if only 1-2 keywords)
            # But also ensure that if there are 3+ keywords, we require at least 2 matches
            min_matches = max(1, len(keywords) // 2) if len(keywords) > 2 else 1
            
            # For queries with objects/actions (like "man riding bicycle"), require the object to be present
            # Identify potential objects (longer words, or words that might be nouns)
            if len(keywords) >= 3:
                # Sort by length (longer words are often objects/nouns)
                sorted_keywords = sorted(keywords, key=len, reverse=True)
                # Require the longest keyword (likely the main object) to be present
                longest_keyword = sorted_keywords[0]
                longest_matched = (
                    longest_keyword in text or
                    longest_keyword + 's' in text or
                    longest_keyword + 'es' in text
                )
                if not longest_matched:
                    continue  # Skip if main object not found
            
            if matches >= min_matches:
                filtered.append(result)
        
        return filtered
    
    def _display_result(self, rank: int, result: Dict[str, Any], row_index: int = 0):
        """Display a single result with its frame."""
        # Format date and time from timestamp
        timestamp = result.get("timestamp")
        date_time_str = "N/A"
        if timestamp:
            try:
                dt = datetime.fromtimestamp(timestamp)
                date_time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError, OSError):
                date_time_str = str(timestamp)
        
        result_text = (
            f"[{rank}] Scene ID: {result.get('scene_id', 'N/A')}\n"
            f"    📅 Date & Time: {date_time_str}\n"
            f"    Level: {result.get('level', 'N/A')}\n"
            f"    Relevance Score: {result.get('relevance_score', 0.0):.3f}\n"
        )
        
        # Show vector similarity and distance
        similarity = result.get('similarity', 0.0)
        distance = result.get('distance')
        if distance is not None:
            # Calculate raw similarity from distance for display
            raw_similarity = 1.0 - distance
            result_text += f"    Vector Similarity: {similarity:.3f} (raw: {raw_similarity:.3f}, distance: {distance:.3f})\n"
        else:
            result_text += f"    Vector Similarity: {similarity:.3f}\n"
        
        if result.get("summary_text"):
            text_preview = result["summary_text"][:150]
            result_text += f"    Text: {text_preview}{'...' if len(result['summary_text']) > 150 else ''}\n"
        
        result_text += f"    Frames: {len(result.get('frame_paths', []))} frame(s)"
        
        self.add_message("Amber", result_text, "amber")
        
        # Display frames (up to 3)
        frame_paths = result.get("frame_paths", [])[:3]
        if frame_paths:
            frame_row = ttk.Frame(self.results_frame)
            frame_row.grid(row=row_index, column=0, sticky=(tk.W, tk.E), pady=5)
            
            for j, frame_path in enumerate(frame_paths):
                self._display_frame(frame_row, frame_path, j)
    
    def _display_frame(self, parent: ttk.Frame, frame_path: str, index: int):
        """Display a single frame image."""
        try:
            full_path = Path(frame_path)
            if not full_path.is_absolute():
                full_path = Path.cwd() / frame_path
            
            if not full_path.exists():
                logger.warning(f"Frame not found: {full_path}")
                # Show placeholder
                placeholder = ttk.Label(
                    parent,
                    text=f"Frame not found\n{Path(frame_path).name}",
                    foreground="gray",
                    font=("Arial", 8)
                )
                placeholder.grid(row=0, column=index, padx=5, ipadx=10, ipady=10)
                return
            
            # Load and resize image
            img = Image.open(full_path)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Create label with image
            frame_label = ttk.Label(parent, image=photo)
            frame_label.image = photo  # Keep a reference
            frame_label.grid(row=0, column=index, padx=5)
            
            # Add filename label below image
            filename_label = ttk.Label(
                parent,
                text=Path(frame_path).name,
                font=("Arial", 8),
                foreground="gray"
            )
            filename_label.grid(row=1, column=index, padx=5)
            
        except Exception as e:
            logger.error(f"Error displaying frame {frame_path}: {e}", exc_info=True)
            # Show error placeholder
            error_label = ttk.Label(
                parent,
                text=f"Error loading\n{Path(frame_path).name}",
                foreground="red",
                font=("Arial", 8)
            )
            error_label.grid(row=0, column=index, padx=5, ipadx=10, ipady=10)
    
    def _show_tooltip(self, event, text: str):
        """Show tooltip on hover."""
        # Simple tooltip implementation
        pass  # Can be enhanced with a tooltip widget
    
    def _hide_tooltip(self, event):
        """Hide tooltip."""
        pass
    
    def _reenable_input(self):
        """Re-enable input after processing."""
        self.query_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.query_entry.focus()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Amber - Video Scene Search Assistant")
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="LanceDB table name (default: from SCENE_INDEX_TABLE env var or 'scene_embeddings')"
    )
    
    args = parser.parse_args()
    
    if args.table:
        os.environ["SCENE_INDEX_TABLE"] = args.table
    
    root = tk.Tk()
    app = AmberAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()

