#!/usr/bin/env python3
"""Batch evaluator for LanceDB vector scene retrieval."""

import argparse
import logging
from typing import List

from scene_indexer import SceneIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_QUERIES: List[str] = [
    "a man wearing a hat",
    "an empty room with a chair",
    "two people talking",
]


def run_queries(queries: List[str], top_k: int, min_similarity: float, level: str) -> None:
    indexer = SceneIndexer()
    if not indexer.enabled:
        logger.error("Scene indexer is disabled. LanceDB connection failed.")
        return

    level_filter = None if level.lower() == "all" else [level.lower()]

    for query in queries:
        print("=" * 100)
        print(f"Query: {query}")
        matches = indexer.query_scenes(
            query_text=query,
            limit=top_k,
            min_similarity=min_similarity,
            level_filter=level_filter
        )
        if not matches:
            print("  No matches found.\n")
            continue
        for rank, match in enumerate(matches, 1):
            metadata = match.get("metadata") or {}
            recognized_faces = metadata.get("recognized_faces") if isinstance(metadata, dict) else None
            print(f"  [{rank}] id={match['id']} | scene_id={match.get('scene_id')} | level={match.get('level')} | sim={match['similarity']:.3f}")
            if match.get("summary_text"):
                text = match["summary_text"]
                print(f"       text: {text[:160]}{'...' if len(text) > 160 else ''}")
            if recognized_faces:
                print(f"       faces: {recognized_faces}")
            frame_paths = match.get("frame_paths", [])
            if frame_paths:
                preview = frame_paths[:3]
                print(f"       frames ({len(frame_paths)}): {preview}{' ...' if len(frame_paths) > len(preview) else ''}")
            print()
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LanceDB vector search with multiple prompts"
    )
    parser.add_argument(
        "queries",
        nargs="*",
        default=DEFAULT_QUERIES,
        help="Queries to run (defaults to a small evaluation suite)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return per query"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum cosine similarity (0-1)"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        help="Filter by record level (scene, chunk, caption, all)"
    )
    args = parser.parse_args()

    run_queries(args.queries, top_k=args.top_k, min_similarity=args.min_similarity, level=args.level)


if __name__ == "__main__":
    main()
