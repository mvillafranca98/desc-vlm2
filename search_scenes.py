#!/usr/bin/env python3
"""
Semantic Scene Search Script

Allows natural language search over indexed video scenes using LanceDB.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from scene_indexer import SceneIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_relevance_score(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Calculate a relevance score for a search result.
    
    Args:
        result: Search result dictionary
        query: Original search query
        
    Returns:
        Dictionary with scoring breakdown and total score
    """
    query_lower = query.lower()
    text = (result.get("summary_text") or "").lower()
    
    # Base similarity score from vector search (0.0 to 1.0)
    vector_similarity = result.get("similarity", 0.0)
    
    # Keyword matching score
    query_words = set(query_lower.split())
    text_words = set(text.split())
    if query_words:
        keyword_overlap = len(query_words.intersection(text_words)) / len(query_words)
    else:
        keyword_overlap = 0.0
    
    # Exact phrase matching bonus
    exact_match = 1.0 if query_lower in text else 0.0
    
    # Level-based weighting (caption-level is more specific)
    level_weights = {
        "caption": 1.2,
        "chunk": 1.1,
        "scene": 1.0
    }
    level = result.get("level", "scene")
    level_weight = level_weights.get(level, 1.0)
    
    # Frame count bonus (more frames = more context)
    frame_count = len(result.get("frame_paths", []))
    frame_bonus = min(0.1, frame_count * 0.01)  # Max 0.1 bonus
    
    # Calculate weighted total score
    base_score = vector_similarity * 0.5  # 50% weight on vector similarity
    keyword_score = keyword_overlap * 0.3  # 30% weight on keyword matching
    exact_score = exact_match * 0.2  # 20% weight on exact phrase match
    
    raw_score = (base_score + keyword_score + exact_score) * level_weight + frame_bonus
    total_score = min(1.0, raw_score)  # Cap at 1.0
    
    return {
        "total_score": total_score,
        "vector_similarity": vector_similarity,
        "keyword_overlap": keyword_overlap,
        "exact_match": exact_match > 0,
        "level_weight": level_weight,
        "frame_bonus": frame_bonus,
        "breakdown": {
            "vector_component": base_score,
            "keyword_component": keyword_score,
            "exact_component": exact_score,
            "level_adjusted": (base_score + keyword_score + exact_score) * level_weight,
            "final_with_bonus": total_score
        }
    }


def export_results(
    results: List[Dict[str, Any]],
    query: str,
    output_format: str = "json",
    output_file: Optional[str] = None
) -> str:
    """Export search results to a file.
    
    Args:
        results: List of search result dictionaries
        query: Original search query
        output_format: Export format ('json' or 'csv')
        output_file: Optional output file path
        
    Returns:
        Path to the exported file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query[:50])
    
    if output_file is None:
        if output_format == "json":
            output_file = f"search_results_{query_safe}_{timestamp}.json"
        else:
            output_file = f"search_results_{query_safe}_{timestamp}.csv"
    
    output_path = Path(output_file)
    
    # Add scores to results
    scored_results = []
    for result in results:
        score_info = calculate_relevance_score(result, query)
        scored_result = {
            **result,
            "relevance_score": score_info["total_score"],
            "scoring_breakdown": score_info
        }
        scored_results.append(scored_result)
    
    if output_format == "json":
        export_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result_count": len(scored_results),
            "results": scored_results
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    else:  # CSV
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "Rank", "Record ID", "Scene ID", "Level", "Similarity", "Relevance Score",
                "Vector Sim", "Keyword Overlap", "Exact Match", "Frame Count",
                "Timestamp", "Text Preview"
            ])
            # Data rows
            for i, result in enumerate(scored_results, 1):
                writer.writerow([
                    i,
                    result.get("id", ""),
                    result.get("scene_id", ""),
                    result.get("level", ""),
                    f"{result.get('similarity', 0.0):.3f}",
                    f"{result.get('relevance_score', 0.0):.3f}",
                    f"{result['scoring_breakdown']['vector_similarity']:.3f}",
                    f"{result['scoring_breakdown']['keyword_overlap']:.3f}",
                    "Yes" if result['scoring_breakdown']['exact_match'] else "No",
                    len(result.get("frame_paths", [])),
                    result.get("timestamp", ""),
                    (result.get("summary_text", "")[:100] or "").replace("\n", " ")
                ])
    
    return str(output_path)


def search_scenes(
    query: str,
    limit: int = 5,
    min_score: float = 0.0,
    level: Optional[str] = None,
    export: bool = False,
    export_format: str = "json",
    export_file: Optional[str] = None,
    embedding_model: Optional[str] = None,
    table_name: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """Search for scenes matching a natural language query.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        min_score: Minimum similarity score (0.0 to 1.0)
        level: Optional level filter (scene, chunk, caption, all)
        export: Whether to export results
        export_format: Export format (json or csv)
        export_file: Optional output file path
        embedding_model: Optional embedding model name (overrides env var)
        table_name: Optional table name (overrides env var and auto-generation)
    """
    from scene_indexer import get_table_name_for_model
    
    # Resolve model and table
    resolved_model = embedding_model or os.getenv("SCENE_INDEX_MODEL", "text-embedding-3-small")
    resolved_table = table_name or os.getenv("SCENE_INDEX_TABLE")
    
    # Auto-generate table name if not provided
    if not resolved_table:
        resolved_table = get_table_name_for_model(resolved_model)
    
    # Initialize scene indexer with resolved configuration
    indexer = SceneIndexer(
        table_name=resolved_table,
        embedding_model=resolved_model
    )
    
    if not indexer.enabled:
        logger.error("Scene indexer is not enabled. Please check LanceDB configuration.")
        logger.info("Set LANCEDB_PROJECT_SLUG and LANCEDB_API_KEY environment variables")
        logger.info("Or use local database (will be created automatically)")
        return
    
    # Perform search
    logger.info(f"Searching for: '{query}'")
    level_filter = None
    if level and level.lower() != "all":
        level_filter = [level.lower()]
    
    results = indexer.query_scenes(query, limit=limit, min_similarity=min_score, level_filter=level_filter)
    
    if not results:
        print(f"\n❌ No scenes found matching: '{query}'")
        print("\nTry:")
        print("  - Using different keywords")
        print("  - Lowering the min_score threshold")
        print("  - Checking if scenes have been indexed")
        return None
    
    # Calculate scores for all results
    scored_results = []
    for result in results:
        score_info = calculate_relevance_score(result, query)
        scored_result = {
            **result,
            "relevance_score": score_info["total_score"],
            "scoring_breakdown": score_info
        }
        scored_results.append(scored_result)
    
    # Sort by relevance score (descending)
    scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Display results
    print(f"\n✅ Found {len(scored_results)} matching scene(s):\n")
    print("=" * 80)
    
    for i, scene in enumerate(scored_results, 1):
        metadata = scene.get("metadata") or {}
        recognized_faces = metadata.get("recognized_faces") if isinstance(metadata, dict) else None
        score_info = scene.get("scoring_breakdown", {})
        
        print(f"\n[{i}] Record ID: {scene['id']}")
        print(f"    Scene ID: {scene.get('scene_id')}")
        print(f"    Level: {scene.get('level')}")
        print(f"    Vector Similarity: {scene.get('similarity', 0.0):.3f}")
        print(f"    📊 Relevance Score: {scene.get('relevance_score', 0.0):.3f}")
        print(f"       └─ Vector: {score_info.get('vector_similarity', 0.0):.3f} | "
              f"Keywords: {score_info.get('keyword_overlap', 0.0):.3f} | "
              f"Exact: {'Yes' if score_info.get('exact_match') else 'No'}")
        print(f"    Timestamp: {scene.get('timestamp', 'N/A')}")
        if scene.get("summary_text"):
            print(f"    Text: {scene['summary_text'][:120]}{'...' if len(scene['summary_text']) > 120 else ''}")
        if recognized_faces:
            print(f"    Faces: {recognized_faces}")
        print(f"    Frames: {len(scene['frame_paths'])} frame(s)")
        
        if scene['frame_paths']:
            print(f"    Frame paths:")
            for frame_path in scene['frame_paths'][:5]:  # Show first 5
                print(f"      - {frame_path}")
            if len(scene['frame_paths']) > 5:
                print(f"      ... and {len(scene['frame_paths']) - 5} more")
        
        print("-" * 80)
    
    # Export if requested
    if export:
        export_path = export_results(
            scored_results[:limit],  # Export only the requested limit
            query,
            export_format,
            export_file
        )
        print(f"\n💾 Results exported to: {export_path}")
        print(f"   Format: {export_format.upper()}")
        print(f"   Results: {len(scored_results[:limit])} (with relevance scores)")
    
    print("\n")
    return scored_results


def list_all_scenes(limit: Optional[int] = None, embedding_model: Optional[str] = None, table_name: Optional[str] = None):
    """List all indexed scenes.
    
    Args:
        limit: Optional limit on number of scenes to display
        embedding_model: Optional embedding model name (overrides env var)
        table_name: Optional table name (overrides env var and auto-generation)
    """
    from scene_indexer import get_table_name_for_model
    
    # Resolve model and table
    resolved_model = embedding_model or os.getenv("SCENE_INDEX_MODEL", "text-embedding-3-small")
    resolved_table = table_name or os.getenv("SCENE_INDEX_TABLE")
    
    # Auto-generate table name if not provided
    if not resolved_table:
        resolved_table = get_table_name_for_model(resolved_model)
    
    # Initialize scene indexer with resolved configuration
    indexer = SceneIndexer(
        table_name=resolved_table,
        embedding_model=resolved_model
    )
    
    if not indexer.enabled:
        logger.error("Scene indexer is not enabled.")
        return
    
    scenes = indexer.get_all_scenes(limit=limit)
    
    if not scenes:
        print("\n❌ No scenes indexed yet.")
        print("Run the main application to generate and index scenes.")
        return
    
    print(f"\n📋 Found {len(scenes)} indexed scene(s):\n")
    print("=" * 80)
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n[{i}] Record ID: {scene['id']}")
        print(f"    Scene ID: {scene.get('scene_id')}")
        print(f"    Level: {scene.get('level')}")
        if scene.get("summary_text"):
            print(f"    Text: {scene['summary_text'][:100]}...")
        print(f"    Timestamp: {scene.get('timestamp', 'N/A')}")
        print(f"    Frames: {len(scene['frame_paths'])} frame(s)")
        print("-" * 80)
    
    print("\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic search over indexed video scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for scenes
  python search_scenes.py "a man wearing a hat"
  python search_scenes.py "person sitting at desk" --limit 10
  python search_scenes.py "outdoor scene" --min-score 0.5
  
  # Export results with scores
  python search_scenes.py "blue apron" --limit 5 --export
  python search_scenes.py "lion" --export --export-format csv
  python search_scenes.py "hyenas" --export --export-file my_results.json
  
  # List all scenes
  python search_scenes.py --list
  python search_scenes.py --list --limit 20
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language search query (e.g., 'a man wearing a hat')"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score (0.0 to 1.0, default: 0.0)"
    )
    
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        help="Filter by record level (scene, chunk, caption, all)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all indexed scenes instead of searching"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export results to a file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--export-format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Export format: json or csv (default: json)"
    )
    
    parser.add_argument(
        "--export-file",
        type=str,
        default=None,
        help="Output file path for export (default: auto-generated filename)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model to use (e.g., 'BAAI/bge-base-en', 'text-embedding-3-small'). Overrides SCENE_INDEX_MODEL env var."
    )
    parser.add_argument(
        "--scene-table",
        type=str,
        default=None,
        help="LanceDB table name to query. Overrides SCENE_INDEX_TABLE env var and auto-generation."
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_all_scenes(
            limit=args.limit if args.limit != 5 else None,
            embedding_model=args.embedding_model,
            table_name=args.scene_table
        )
    elif args.query:
        search_scenes(
            args.query,
            limit=args.limit,
            min_score=args.min_score,
            level=args.level,
            export=args.export,
            export_format=args.export_format,
            export_file=args.export_file,
            embedding_model=args.embedding_model,
            table_name=args.scene_table
        )
    else:
        parser.print_help()
        print("\n❌ Error: Please provide a search query or use --list")
        sys.exit(1)


if __name__ == "__main__":
    main()

