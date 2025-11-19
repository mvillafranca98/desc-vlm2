#!/usr/bin/env python3
"""
LanceDB Scene Indexer for Semantic Search

Stores vector embeddings of scene summaries and enables semantic search
over video scenes using natural language queries.
"""

import os
import json
import math
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

# Provide default LanceDB Cloud credentials if not explicitly set
os.environ.setdefault("LANCEDB_PROJECT_SLUG", "descvlm2-lnh0lv")
os.environ.setdefault("LANCEDB_API_KEY", "sk_SJLWUL2G2JDQFLZ5WKKM5DC4KJKIVAV73IECRDHTRSBUAEMY2DSQ====")
os.environ.setdefault("LANCEDB_REGION", "us-east-1")

# Try to import LanceDB
try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    logger.warning("LanceDB library not available")
    LANCEDB_AVAILABLE = False

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers library not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import OpenAI for API-based embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("openai library not available")
    OPENAI_AVAILABLE = False


def get_table_name_for_model(embedding_model: str, base_table: Optional[str] = None) -> str:
    """Generate a table name based on the embedding model to avoid dimension mismatches.
    
    Different embedding models produce different vector dimensions:
    - all-MiniLM-L6-v2: 384 dimensions
    - e5-large-v2: 1024 dimensions  
    - bge-base-en: 768 dimensions
    
    Args:
        embedding_model: Name of the embedding model
        base_table: Optional base table name (if None, uses model-specific default)
        
    Returns:
        Table name that includes model identifier
    """
    if base_table:
        # If base table is provided, use it as-is (user override)
        return base_table
    
    # Generate model-specific table name
    model_short = embedding_model.split("/")[-1]  # Get last part of model name
    model_short = model_short.replace("-", "_")  # Replace hyphens with underscores
    
    # Map common models to simple table names (matching existing pattern)
    if "minilm" in model_short.lower():
        return "scene_embeddings"  # Keep original table for MiniLM
    elif "e5" in model_short.lower() or "e5_large" in model_short.lower():
        return "scene_embeddings_e5"  # Match existing e5 table pattern
    elif "bge" in model_short.lower():
        return "scene_embeddings_bge"  # Simple name like scene_embeddings_e5
    elif "text-embedding-3-small" in embedding_model.lower() or "text-embedding-3" in embedding_model.lower():
        return "scene_embeddings_openai_small"  # OpenAI embedding-3-small
    elif "openai" in embedding_model.lower():
        return "scene_embeddings_openai"  # Generic OpenAI embeddings
    else:
        # Generic fallback - will use actual dimension when model is loaded
        return f"scene_embeddings_{model_short}"


class SceneIndexer:
    """Manages scene embeddings and semantic search using LanceDB.
    
    This class handles:
    - LanceDB connection and table management (model-specific tables)
    - Embedding generation (local sentence-transformers or OpenAI API)
    - Scene indexing with multi-level embeddings (scene, chunk, caption)
    - Semantic search queries
    
    Each embedding model uses its own table to avoid dimension mismatches.
    Table names are auto-generated based on the model (e.g., scene_embeddings_bge for BAAI/bge-base-en).
    """
    
    def __init__(
        self,
        table_name: str = "scene_index",
        embedding_model: str = "text-embedding-3-small",
        frames_dir: str = "frames",
        enable_chunking: bool = True,
        chunk_duration: float = 10.0,
        enable_caption_embeddings: bool = True,
        session_id: Optional[str] = None
    ):
        self.embedding_model_name = embedding_model
        
        # Determine table name: use env var if set, otherwise generate model-specific name
        env_table = os.getenv("SCENE_INDEX_TABLE")
        if env_table:
            # User explicitly set table name via environment variable
            self.table_name = env_table
        elif table_name == "scene_index":
            # Default table name - generate model-specific name
            self.table_name = get_table_name_for_model(embedding_model)
        else:
            # User explicitly provided table name in code
            self.table_name = table_name
        self.frames_dir = Path(frames_dir)
        self.db = None
        self.table = None
        self.embedding_model = None
        self.openai_client = None
        self.use_openai_api = False
        self.enabled = False
        self.enable_chunking = enable_chunking
        self.chunk_duration = chunk_duration
        self.enable_caption_embeddings = enable_caption_embeddings
        self.session_id = session_id or os.getenv("SCENE_INDEX_SESSION_ID", str(uuid.uuid4()))
        self.embedding_dim: Optional[int] = None
        self.table_schema: Optional[pa.Schema] = None
        
        # Create frames directory if it doesn't exist
        self.frames_dir.mkdir(exist_ok=True)
        
        # Initialize components
        if not LANCEDB_AVAILABLE:
            logger.warning("LanceDB not available - scene indexing disabled")
            return
        
        # Check if we need sentence-transformers or OpenAI
        is_openai_model = "text-embedding" in self.embedding_model_name.lower() or "openai" in self.embedding_model_name.lower()
        
        if not is_openai_model and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available - scene indexing disabled")
            return
        
        if is_openai_model and not OPENAI_AVAILABLE:
            logger.warning("openai library not available - scene indexing disabled")
            return
        
        try:
            self._initialize_lancedb()
            self._initialize_embedding_model()
            # Table name is already set correctly by get_table_name_for_model()
            logger.info(f"Using table: {self.table_name} (model: {self.embedding_model_name}, dimension: {self.embedding_dim})")
            self._ensure_table_schema()
            self.enabled = True
            logger.info("Scene indexer initialized successfully")
            logger.info(f"✓ Embedding model: {self.embedding_model_name} ({'OpenAI API' if self.use_openai_api else 'Local'}, {self.embedding_dim} dimensions)")
            logger.info(
                "Scene indexer config: table=%s, model=%s, dimension=%s, chunking=%s (%.1fs), caption_embeddings=%s, session_id=%s",
                self.table_name,
                self.embedding_model_name,
                self.embedding_dim or "unknown",
                self.enable_chunking,
                self.chunk_duration,
                self.enable_caption_embeddings,
                self.session_id
            )
        except Exception as e:
            logger.error(f"Failed to initialize scene indexer: {e}", exc_info=True)
            self.enabled = False
    
    def _initialize_lancedb(self):
        """Initialize LanceDB connection."""
        # Get credentials from environment
        project_slug = os.getenv("LANCEDB_PROJECT_SLUG")
        api_key = os.getenv("LANCEDB_API_KEY")
        region = os.getenv("LANCEDB_REGION", "us-east-1")
        
        if not project_slug or not api_key:
            logger.warning("LanceDB credentials not found in environment")
            logger.info("Set LANCEDB_PROJECT_SLUG and LANCEDB_API_KEY to enable scene indexing")
            logger.info("For local testing, you can use: db = lancedb.connect('data/scenes')")
            # Fallback to local database for development
            local_db_path = "data/scenes"
            Path(local_db_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = lancedb.connect(local_db_path)
            logger.info(f"Using local LanceDB at: {local_db_path}")
        else:
            # Connect to LanceDB Cloud
            self.db = lancedb.connect(
                uri=f"db://{project_slug}",
                api_key=api_key,
                region=region
            )
            logger.info(f"Connected to LanceDB Cloud: {project_slug}")
        
        # Create or get table
        try:
            self.table = self.db.open_table(self.table_name)
            logger.info(f"Opened existing table: {self.table_name}")
            
            # Check if table dimension matches model dimension (after model is initialized)
            # This check will be done in _initialize_embedding_model after dimensions are known
        except Exception:
            # Table doesn't exist, will be created on first insert
            logger.info(f"Table {self.table_name} will be created on first insert")
            self.table = None
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model (local or OpenAI API)."""
        try:
            # Check if this is an OpenAI model
            if "text-embedding" in self.embedding_model_name.lower() or "openai" in self.embedding_model_name.lower():
                if not OPENAI_AVAILABLE:
                    raise RuntimeError("OpenAI library not available. Install with: pip install openai")
                
                # Get API key from environment
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY environment variable not set. "
                        "Get your API key from https://platform.openai.com/api-keys"
                    )
                
                logger.info(f"Initializing OpenAI embedding model: {self.embedding_model_name}...")
                self.openai_client = OpenAI(api_key=api_key)
                self.use_openai_api = True
                
                # Get dimension by making a test API call
                # text-embedding-3-small has 1536 dimensions
                # text-embedding-3-large has 3072 dimensions
                # Default to 1536 for small model
                if "small" in self.embedding_model_name.lower():
                    self.embedding_dim = 1536
                elif "large" in self.embedding_model_name.lower():
                    self.embedding_dim = 3072
                else:
                    # Make a test call to get actual dimension
                    try:
                        response = self.openai_client.embeddings.create(
                            model=self.embedding_model_name,
                            input="dimension probe"
                        )
                        self.embedding_dim = len(response.data[0].embedding)
                    except Exception as e:
                        logger.warning(f"Could not determine dimension from API call: {e}, defaulting to 1536")
                        self.embedding_dim = 1536
                
                logger.info(f"OpenAI embedding model initialized (dimension: {self.embedding_dim})")
            else:
                # Use local sentence-transformers model
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise RuntimeError("sentence-transformers library not available")
                
                logger.info(f"Loading embedding model: {self.embedding_model_name}...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                probe_embedding = self.embedding_model.encode(
                    "dimension probe", convert_to_numpy=True
                )
                self.embedding_dim = int(len(probe_embedding))
                logger.info(f"Embedding model loaded successfully (dimension: {self.embedding_dim})")
            
            # Create schema with determined dimension
            list_type = pa.list_(pa.float32(), self.embedding_dim)
            self.table_schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("scene_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("level", pa.string()),
                pa.field("embedding", list_type),
                pa.field("vector", list_type),
                pa.field("frame_paths", pa.list_(pa.string())),
                pa.field("summary_text", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("scene_start_time", pa.float64()),
                pa.field("scene_end_time", pa.float64()),
                pa.field("timestamp", pa.float64())
            ])
            
            # Check if existing table has dimension mismatch
            if self.table is not None and self.embedding_dim is not None:
                try:
                    # Try to get schema to check vector dimension
                    schema = self.table.schema
                    for field in schema:
                        if field.name == "vector":
                            # Check dimension from the field type
                            field_type = field.type
                            table_dim = None
                            
                            # Check if it's a FixedSizeListType (has list_size attribute)
                            if hasattr(field_type, 'list_size'):
                                table_dim = field_type.list_size
                            else:
                                # Fallback: try to get a sample record
                                try:
                                    sample = self.table.head(1)
                                    if sample and len(sample) > 0:
                                        sample_vector = sample[0].get("vector")
                                        if sample_vector:
                                            table_dim = len(sample_vector)
                                except Exception:
                                    pass
                            
                            if table_dim is not None and table_dim != self.embedding_dim:
                                expected_table = get_table_name_for_model(self.embedding_model_name)
                                logger.error(
                                    f"❌ DIMENSION MISMATCH DETECTED!\n"
                                    f"   Current table '{self.table_name}' has {table_dim}-dim vectors\n"
                                    f"   Model '{self.embedding_model_name}' generates {self.embedding_dim}-dim vectors\n"
                                    f"   Expected table name: '{expected_table}'\n"
                                    f"   Solution: Remove SCENE_INDEX_TABLE from .env or set it to '{expected_table}'"
                                )
                                raise ValueError(
                                    f"Table dimension mismatch: table has {table_dim} dims, "
                                    f"model generates {self.embedding_dim} dims. "
                                    f"Use table '{expected_table}' instead."
                                )
                            break
                except ValueError:
                    # Re-raise ValueError (dimension mismatch)
                    raise
                except Exception as schema_error:
                    # If we can't check schema, log warning but continue
                    logger.warning(f"Could not verify table schema dimensions: {schema_error}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise

    def _ensure_table_schema(self):
        """Ensure existing table matches expected schema; migrate if necessary."""
        if self.table is None or self.table_schema is None or self.embedding_dim is None:
            return
        try:
            schema_names = set(self.table.schema.names)
        except Exception as schema_error:
            logger.warning(f"Unable to inspect table schema: {schema_error}")
            return
        if "embedding" in schema_names and "vector" in schema_names:
            return  # Table already uses new schema
        logger.warning(
            "Legacy LanceDB table detected (missing 'embedding' column). "
            "Migrating existing records to new schema..."
        )
        try:
            legacy_arrow = self.table.to_arrow()
            legacy_records = legacy_arrow.to_pylist()
        except Exception as read_error:
            logger.error(f"Failed to read legacy LanceDB table for migration: {read_error}", exc_info=True)
            legacy_records = []
        records = []
        if legacy_records:
            for row in legacy_records:
                vector = row.get("embedding") or row.get("vector")
                if vector is None:
                    continue
                try:
                    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
                except Exception:
                    continue
                if embedding.size != self.embedding_dim:
                    if embedding.size > self.embedding_dim:
                        embedding = embedding[:self.embedding_dim]
                    else:
                        padding = np.zeros(self.embedding_dim - embedding.size, dtype=np.float32)
                        embedding = np.concatenate([embedding, padding])
                frame_paths_value = row.get("frame_paths")
                if isinstance(frame_paths_value, str):
                    frame_paths = [p.strip() for p in frame_paths_value.split(",") if p.strip()]
                elif isinstance(frame_paths_value, list):
                    frame_paths = [p for p in frame_paths_value if p]
                else:
                    frame_paths = []
                metadata_value = row.get("metadata")
                if isinstance(metadata_value, str):
                    try:
                        metadata = json.loads(metadata_value)
                    except json.JSONDecodeError:
                        metadata = {"legacy_raw_metadata": metadata_value}
                elif isinstance(metadata_value, dict):
                    metadata = metadata_value
                else:
                    metadata = {}
                metadata.setdefault("legacy_schema", True)
                metadata.setdefault("legacy_columns", list(row.keys()))
                record = {
                    "id": str(row.get("id") or uuid.uuid4()),
                    "scene_id": str(row.get("scene_id") or row.get("id") or uuid.uuid4()),
                    "session_id": str(row.get("session_id") or self.session_id),
                    "level": row.get("level") or "scene",
                    "embedding": embedding.astype(np.float32).tolist(),
                    "vector": embedding.astype(np.float32).tolist(),
                    "frame_paths": frame_paths,
                    "summary_text": row.get("summary_text"),
                    "metadata": json.dumps(metadata),
                    "scene_start_time": row.get("scene_start_time"),
                    "scene_end_time": row.get("scene_end_time"),
                    "timestamp": row.get("timestamp") or time.time()
                }
                records.append(record)
        backup_dir = Path("data")
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"{self.table_name}_legacy_backup_{int(time.time())}.json"
        try:
            if legacy_records:
                with open(backup_path, "w", encoding="utf-8") as backup_file:
                    json.dump(legacy_records, backup_file, indent=2, default=str)
                logger.info("Legacy LanceDB data exported to %s", backup_path)
        except Exception as backup_error:
            logger.warning(f"Failed to export legacy LanceDB data: {backup_error}")
        try:
            self.db.drop_table(self.table_name)
            logger.info("Legacy LanceDB table '%s' dropped", self.table_name)
        except Exception as drop_error:
            logger.error(f"Failed to drop legacy LanceDB table: {drop_error}", exc_info=True)
            return
        if records:
            try:
                self.table = self.db.create_table(self.table_name, data=records, schema=self.table_schema)
                logger.info(
                    "Migrated %d legacy record(s) to new LanceDB schema (embedding column).",
                    len(records)
                )
            except Exception as create_error:
                logger.error(f"Failed to recreate LanceDB table with new schema: {create_error}", exc_info=True)
                self.table = None
        else:
            logger.info("No legacy rows to migrate; new LanceDB table will be created on first insert.")
            self.table = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text.
        
        Uses OpenAI API if use_openai_api is True, otherwise uses local sentence-transformers model.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding vector (float32)
            
        Raises:
            RuntimeError: If indexer not initialized or model not available
        """
        if not self.enabled:
            raise RuntimeError("Scene indexer not properly initialized")
        
        try:
            if self.use_openai_api:
                if not self.openai_client:
                    raise RuntimeError("OpenAI client not initialized")
                
                # Call OpenAI API
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=text
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding
            else:
                if not self.embedding_model:
                    raise RuntimeError("Embedding model not initialized")
                
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    def save_frame(self, frame: np.ndarray, scene_id: str, frame_index: int) -> str:
        """Save a frame to disk and return the file path."""
        try:
            # Create filename: scene_id_frame_index.jpg
            filename = f"{scene_id}_frame_{frame_index:04d}.jpg"
            filepath = self.frames_dir / filename
            
            # Save frame
            import cv2
            cv2.imwrite(str(filepath), frame)
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving frame: {e}", exc_info=True)
            return ""
    
    def index_scene(
        self,
        scene_id: str,
        frame_paths: List[str],
        summary_text: Optional[str] = None,
        captions: Optional[List[Dict[str, Any]]] = None,
        scene_start_time: Optional[float] = None,
        scene_end_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Index embeddings for a scene and optional sub-components.
        
        Args:
            scene_id: Unique identifier for the scene/window.
            frame_paths: List of file paths to frames associated with this scene.
            summary_text: Optional summary text used for scene-level embedding.
            captions: Optional list of caption dicts with keys: text, timestamp, frame_index, frame_path, metadata.
            scene_start_time: Optional timestamp when the scene started.
            scene_end_time: Optional timestamp when the scene ended.
            metadata: Optional additional metadata dict.
        
        Returns:
            The scene_id if successful, None otherwise.
        """
        if not self.enabled:
            return None
        
        try:
            records = []
            timestamp = time.time()
            metadata = metadata or {}
            metadata.setdefault("frame_count", len(frame_paths))
            metadata.setdefault("session_id", self.session_id)
            
            # Ensure frame paths stored as list
            frame_paths_clean = [p for p in frame_paths if p]
            
            # Build scene-level embedding text
            scene_text = summary_text
            if not scene_text and captions:
                scene_text = " ".join([c.get("text", "") for c in captions if c.get("text")])
                scene_text = scene_text.strip() or None
            
            if scene_text:
                scene_embedding = self.generate_embedding(scene_text)
                records.append(self._build_record(
                    base_id=scene_id,
                    level="scene",
                    embedding=scene_embedding,
                    frame_paths=frame_paths_clean,
                    summary_text=summary_text,
                    metadata=metadata,
                    scene_start_time=scene_start_time,
                    scene_end_time=scene_end_time,
                    timestamp=timestamp
                ))
            else:
                logger.warning("No scene text available for embedding; skipping scene-level record")
            
            # Optional caption-level embeddings
            if self.enable_caption_embeddings and captions:
                for idx, caption in enumerate(captions):
                    caption_text = caption.get("text")
                    if not caption_text:
                        continue
                    caption_embedding = self.generate_embedding(caption_text)
                    caption_metadata = {
                        "type": "caption",
                        "caption_index": idx,
                        "caption_timestamp": caption.get("timestamp"),
                        "original_metadata": caption.get("metadata"),
                        "session_id": self.session_id
                    }
                    if caption.get("frame_path"):
                        caption_frame_paths = [caption["frame_path"]]
                    else:
                        caption_frame_paths = frame_paths_clean
                    records.append(self._build_record(
                        base_id=scene_id,
                        level="caption",
                        suffix=str(idx),
                        embedding=caption_embedding,
                        frame_paths=caption_frame_paths,
                        summary_text=caption_text,
                        metadata=caption_metadata,
                        scene_start_time=caption.get("timestamp", scene_start_time),
                        scene_end_time=caption.get("timestamp", scene_end_time),
                        timestamp=timestamp
                    ))
            
            # Optional chunk-level embeddings
            if self.enable_chunking and captions:
                chunk_records = self._build_chunk_records(
                    scene_id=scene_id,
                    captions=captions,
                    frame_paths=frame_paths_clean,
                    scene_start_time=scene_start_time,
                    scene_end_time=scene_end_time,
                    timestamp=timestamp
                )
                records.extend(chunk_records)
            
            if not records:
                logger.warning("No records generated for scene indexing")
                return None
            
            # Create table if it doesn't exist, otherwise append
            if self.table is None:
                try:
                    self.table = self.db.create_table(
                        self.table_name,
                        data=records,
                        schema=self.table_schema
                    )
                    logger.info(f"✓ Created table: {self.table_name} (dimension: {self.embedding_dim})")
                    try:
                        self.table.create_index(metric="cosine", vector_column_name="vector")
                        logger.info("✓ Created vector index for semantic search")
                    except Exception as idx_error:
                        logger.warning(f"Could not create index (continuing): {idx_error}")
                except Exception as create_error:
                    logger.error(f"Failed to create LanceDB table '{self.table_name}': {create_error}", exc_info=True)
                    return None
            else:
                try:
                    self.table.add(records)
                    logger.debug(f"Added {len(records)} records for scene: {scene_id}")
                except Exception as add_error:
                    logger.error(f"Failed to add records to LanceDB table: {add_error}", exc_info=True)
                    return None
            
            logger.info(f"Indexed scene '{scene_id}' with {len(records)} records")
            return scene_id
            
        except Exception as e:
            logger.error(f"Error indexing scene: {e}", exc_info=True)
            return None
    
    def _build_record(
        self,
        base_id: str,
        level: str,
        embedding: np.ndarray,
        frame_paths: List[str],
        summary_text: Optional[str],
        metadata: Optional[Dict[str, Any]],
        scene_start_time: Optional[float],
        scene_end_time: Optional[float],
        timestamp: float,
        suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        record_id = f"{base_id}::{self.session_id}::{level}"
        if suffix is not None:
            record_id = f"{record_id}::{suffix}"
        
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        metadata.setdefault("level", level)
        metadata.setdefault("session_id", self.session_id)
        
        # LanceDB currently stores native lists; ensure serialization-friendly copy
        if embedding.size != self.embedding_dim:
            if embedding.size > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                padding = np.zeros(self.embedding_dim - embedding.size, dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
        record = {
            "id": record_id,
            "scene_id": base_id,
            "session_id": self.session_id,
            "level": level,
            "embedding": embedding.astype(np.float32).tolist(),
            "vector": embedding.astype(np.float32).tolist(),
            "frame_paths": list(dict.fromkeys(frame_paths)),  # remove duplicates preserving order
            "summary_text": summary_text,
            "metadata": json.dumps(metadata),
            "scene_start_time": scene_start_time,
            "scene_end_time": scene_end_time,
            "timestamp": timestamp
        }
        return record
    
    def _build_chunk_records(
        self,
        scene_id: str,
        captions: List[Dict[str, Any]],
        frame_paths: List[str],
        scene_start_time: Optional[float],
        scene_end_time: Optional[float],
        timestamp: float
    ) -> List[Dict[str, Any]]:
        if not captions or self.chunk_duration <= 0:
            return []
        
        sorted_captions = sorted(
            (c for c in captions if c.get("text")),
            key=lambda c: c.get("timestamp", 0.0)
        )
        if not sorted_captions:
            return []
        
        first_ts = sorted_captions[0].get("timestamp", scene_start_time or timestamp)
        chunk_groups: Dict[int, Dict[str, Any]] = {}
        
        for caption in sorted_captions:
            cap_ts = caption.get("timestamp", first_ts)
            try:
                window_index = int(math.floor((cap_ts - first_ts) / self.chunk_duration))
            except Exception:
                window_index = 0
            
            group = chunk_groups.setdefault(window_index, {
                "texts": [],
                "frame_paths": set(),
                "start": cap_ts,
                "end": cap_ts
            })
            group["texts"].append(caption["text"])
            group["start"] = min(group["start"], cap_ts)
            group["end"] = max(group["end"], cap_ts)
            frame_path = caption.get("frame_path")
            if frame_path:
                group["frame_paths"].add(frame_path)
        
        chunk_records = []
        for idx, (window, group) in enumerate(sorted(chunk_groups.items())):
            chunk_text = " ".join(group["texts"]).strip()
            if not chunk_text:
                continue
            chunk_embedding = self.generate_embedding(chunk_text)
            chunk_metadata = {
                "type": "chunk",
                "chunk_index": window,
                "chunk_start": group["start"],
                "chunk_end": group["end"],
                "session_id": self.session_id
            }
            chunk_frame_paths = list(group["frame_paths"]) or frame_paths
            chunk_records.append(self._build_record(
                base_id=scene_id,
                level="chunk",
                suffix=str(window),
                embedding=chunk_embedding,
                frame_paths=chunk_frame_paths,
                summary_text=chunk_text,
                metadata=chunk_metadata,
                scene_start_time=group["start"],
                scene_end_time=group["end"],
                timestamp=timestamp
            ))
        return chunk_records
    
    def query_scenes(
        self,
        query_text: str,
        limit: int = 5,
        min_similarity: float = 0.0,
        level_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for scenes or sub-scenes using VECTOR EMBEDDING similarity.
        
        IMPORTANT: This is a VECTOR-BASED search, not a text-based search.
        - The query text is converted to an embedding vector using the embedding model
        - The search is performed against the 'embedding'/'vector' column in LanceDB
        - Results are ranked by cosine similarity between query embedding and stored embeddings
        - The 'summary_text' field is NOT used for searching, only for display and post-filtering
        
        This enables semantic search where queries like "male human performing art" can
        match scenes with "a man playing piano" based on semantic similarity in vector space.
        
        Args:
            query_text: Natural language search query (e.g., "a man wearing a hat").
            limit: Maximum number of results to return.
            min_similarity: Minimum similarity score (0.0 to 1.0) required for inclusion.
            level_filter: Optional list of levels to include (e.g., ["scene", "chunk", "caption"]).
        
        Returns:
            List of dictionaries containing id, similarity, frame_paths, and metadata.
        """
        if not self.enabled or self.table is None:
            logger.warning("Scene indexer not available or table is empty")
            return []
        try:
            if self.table.count_rows() == 0:
                logger.warning("Scene index is empty; no vectors available")
                return []
        except Exception:
            pass
        
        try:
            # IMPORTANT: This search is VECTOR-BASED, not text-based
            # The query text is converted to an embedding vector, and we search against
            # the 'embedding'/'vector' column in LanceDB, NOT the 'summary_text' column.
            # This enables semantic search (e.g., "man playing piano" matches "male human performing art")
            
            # Generate embedding for query text
            query_embedding = self.generate_embedding(query_text)
            
            # Perform vector similarity search in LanceDB
            # LanceDB search expects the vector directly, not as list
            # This searches through the embedding vectors, not the text_summary field
            search = self.table.search(query_embedding.tolist(), vector_column_name="vector")
            if level_filter:
                # LanceDB currently doesn't filter via search; filter after retrieval
                pass
            results = search.limit(limit).to_list()
            
            # Debug: Log structure of first result to understand what fields are available
            if results and logger.isEnabledFor(logging.DEBUG):
                first_result = results[0]
                logger.debug(f"First result keys: {list(first_result.keys())}")
                for key in ['_distance', 'distance', '_l2_distance', '_similarity', 'similarity']:
                    if key in first_result:
                        logger.debug(f"Found distance field '{key}': {first_result[key]}")
            
            # Convert results to list of dictionaries
            matches = []
            for row in results:
                level = row.get("level")
                if level_filter and level not in level_filter:
                    continue
                
                frame_paths_value = row.get("frame_paths", [])
                if isinstance(frame_paths_value, str):
                    frame_paths = [p.strip() for p in frame_paths_value.split(",") if p.strip()]
                else:
                    frame_paths = list(frame_paths_value) if frame_paths_value is not None else []
                
                metadata_value = row.get("metadata")
                if isinstance(metadata_value, str):
                    try:
                        metadata = json.loads(metadata_value)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata_value}
                else:
                    metadata = metadata_value or {}
                
                # Try multiple possible field names for distance/similarity
                # LanceDB may return distance in different fields depending on version/cloud
                distance_or_similarity = (
                    row.get("_distance") or 
                    row.get("distance") or 
                    row.get("_l2_distance") or
                    row.get("_similarity") or
                    row.get("similarity") or
                    None
                )
                
                # Debug: Log available fields for first result only
                if len(matches) == 0 and distance_or_similarity is None:
                    logger.debug(f"Distance field not found. Available keys in result: {list(row.keys())[:10]}")
                
                # Calculate similarity from distance
                if distance_or_similarity is None:
                    # Distance field not found - compute cosine similarity manually as fallback
                    try:
                        # Get embedding from result
                        result_embedding = row.get("embedding") or row.get("vector")
                        if result_embedding and query_embedding is not None:
                            # Convert to numpy arrays
                            result_vec = np.array(result_embedding, dtype=np.float32)
                            query_vec = np.array(query_embedding, dtype=np.float32)
                            
                            # Compute cosine similarity: dot product / (norm1 * norm2)
                            dot_product = np.dot(result_vec, query_vec)
                            norm1 = np.linalg.norm(result_vec)
                            norm2 = np.linalg.norm(query_vec)
                            
                            if norm1 > 0 and norm2 > 0:
                                similarity = float(dot_product / (norm1 * norm2))
                                # Cosine similarity ranges from -1 to 1, normalize to 0-1
                                similarity = max(0.0, similarity)  # Cap at 0.0 minimum
                            else:
                                similarity = 0.0
                        else:
                            logger.warning("Cannot compute similarity: embedding not found in result")
                            similarity = 0.0
                    except Exception as e:
                        logger.warning(f"Error computing cosine similarity: {e}, using 0.0")
                        similarity = 0.0
                else:
                    # LanceDB with cosine metric returns cosine distance (0-2 range):
                    # Cosine distance = 1 - cosine_similarity
                    # So: cosine_similarity = 1 - cosine_distance
                    # 
                    # Range mapping:
                    # distance = 0.0 → similarity = 1.0 (identical)
                    # distance = 1.0 → similarity = 0.0 (orthogonal)
                    # distance = 2.0 → similarity = -1.0 (opposite)
                    #
                    # For display, we normalize to 0-1 range (negative similarities become 0.0)
                    distance = float(distance_or_similarity)
                    
                    # Convert cosine distance to cosine similarity
                    # Cosine distance = 1 - cosine_similarity
                    # So: cosine_similarity = 1 - cosine_distance
                    cosine_similarity = 1.0 - distance
                    
                    # For display, we can show negative similarities as very small positive values
                    # OR we can show the raw similarity (which can be negative)
                    # Let's normalize: negative similarities become 0.0, but log the raw value
                    similarity = max(0.0, cosine_similarity)
                    
                    # Store raw similarity in metadata for debugging/analysis
                    # This allows us to see actual semantic similarity even when negative
                    if cosine_similarity < 0.0:
                        logger.debug(f"Cosine similarity is negative ({cosine_similarity:.3f}), capping at 0.0. Distance: {distance:.3f}")
                        # Store raw similarity for potential use in flexible filtering
                        if "metadata" in row:
                            if isinstance(metadata, dict):
                                metadata["raw_cosine_similarity"] = cosine_similarity
                            elif isinstance(metadata, str):
                                try:
                                    metadata_dict = json.loads(metadata)
                                    metadata_dict["raw_cosine_similarity"] = cosine_similarity
                                    metadata = json.dumps(metadata_dict)
                                except:
                                    pass
                
                if similarity < min_similarity:
                    continue
                
                match = {
                    "id": row.get("id"),
                    "scene_id": row.get("scene_id"),
                    "level": level,
                    "similarity": similarity,
                    "distance": distance_or_similarity if distance_or_similarity is not None else None,  # Include raw distance for flexible filtering
                    "frame_paths": frame_paths,
                    "summary_text": row.get("summary_text"),
                    "metadata": metadata,
                    "scene_start_time": row.get("scene_start_time"),
                    "scene_end_time": row.get("scene_end_time"),
                    "timestamp": row.get("timestamp")
                }
                matches.append(match)
            
            logger.info(f"Found {len(matches)} matching scenes for query: '{query_text}'")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching scenes: {e}", exc_info=True)
            return []
    
    
    def get_all_scenes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all indexed scenes.
        
        Args:
            limit: Optional limit on number of scenes to return
        
        Returns:
            List of all scenes with their metadata
        """
        if not self.enabled or self.table is None:
            return []
        
        try:
            total_rows = self.table.count_rows()
            if total_rows == 0:
                return []
            sample_limit = total_rows if limit is None else min(limit, total_rows)
            if self.embedding_dim is None:
                # Determine dimension using probe embedding
                if self.use_openai_api:
                    if "small" in self.embedding_model_name.lower():
                        self.embedding_dim = 1536
                    elif "large" in self.embedding_model_name.lower():
                        self.embedding_dim = 3072
                    else:
                        # Make API call to determine dimension
                        try:
                            response = self.openai_client.embeddings.create(
                                model=self.embedding_model_name,
                                input="scene listing probe"
                            )
                            self.embedding_dim = len(response.data[0].embedding)
                        except Exception:
                            self.embedding_dim = 1536  # Default fallback
                else:
                    probe_embedding = self.embedding_model.encode(
                        "scene listing probe", convert_to_numpy=True
                    )
                    self.embedding_dim = int(len(probe_embedding))
            zero_vector = np.zeros(self.embedding_dim, dtype=np.float32).tolist()
            records = (
                self.table
                .search(zero_vector, vector_column_name="vector")
                .limit(sample_limit)
                .to_list()
            )
            scenes = []
            for row in records:
                frame_paths_value = row.get("frame_paths", [])
                if isinstance(frame_paths_value, str):
                    frame_paths = [p.strip() for p in frame_paths_value.split(",") if p.strip()]
                else:
                    frame_paths = list(frame_paths_value) if frame_paths_value is not None else []
                
                metadata_value = row.get("metadata")
                if isinstance(metadata_value, str):
                    try:
                        metadata = json.loads(metadata_value)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata_value}
                else:
                    metadata = metadata_value or {}
                
                scene = {
                    "id": row.get("id"),
                    "scene_id": row.get("scene_id"),
                    "level": row.get("level"),
                    "summary_text": row.get("summary_text"),
                    "frame_paths": frame_paths,
                    "scene_start_time": row.get("scene_start_time"),
                    "scene_end_time": row.get("scene_end_time"),
                    "timestamp": row.get("timestamp"),
                    "metadata": metadata
                }
                scenes.append(scene)
            
            return scenes
            
        except Exception as e:
            logger.error(f"Error getting all scenes: {e}", exc_info=True)
            return []
    
    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene from the index.
        
        Args:
            scene_id: ID of the scene to delete
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or self.table is None:
            return False
        
        try:
            # LanceDB delete operation
            self.table.delete(f"id = '{scene_id}'")
            logger.info(f"Deleted scene: {scene_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting scene: {e}", exc_info=True)
            return False

