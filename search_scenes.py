#!/usr/bin/env python3
"""
Semantic Scene Search Script

Allows natural language search over indexed video scenes using LanceDB.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from scene_indexer import SceneIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def search_scenes(query: str, limit: int = 5, min_score: float = 0.0, level: Optional[str] = None):
    """Search for scenes matching a natural language query.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        min_score: Minimum similarity score (0.0 to 1.0)
    """
    # Initialize scene indexer
    indexer = SceneIndexer()
    
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
        return
    
    # Display results
    print(f"\n✅ Found {len(results)} matching scene(s):\n")
    print("=" * 80)
    
    for i, scene in enumerate(results, 1):
        metadata = scene.get("metadata") or {}
        recognized_faces = metadata.get("recognized_faces") if isinstance(metadata, dict) else None
        
        print(f"\n[{i}] Record ID: {scene['id']}")
        print(f"    Scene ID: {scene.get('scene_id')}")
        print(f"    Level: {scene.get('level')}")
        print(f"    Similarity: {scene.get('similarity', 0.0):.3f}")
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
    
    print("\n")


def list_all_scenes(limit: Optional[int] = None):
    """List all indexed scenes.
    
    Args:
        limit: Optional limit on number of scenes to display
    """
    indexer = SceneIndexer()
    
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
    
    args = parser.parse_args()
    
    if args.list:
        list_all_scenes(limit=args.limit if args.limit != 5 else None)
    elif args.query:
        search_scenes(args.query, limit=args.limit, min_score=args.min_score, level=args.level)
    else:
        parser.print_help()
        print("\n❌ Error: Please provide a search query or use --list")
        sys.exit(1)


if __name__ == "__main__":
    main()

