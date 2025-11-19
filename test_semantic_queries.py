#!/usr/bin/env python3
"""
Batch test semantic queries and generate HTML report with images.
"""

import os
import sys
import json
import base64
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables from .env file if it exists (before importing other modules)
# Load .env values and ALWAYS override existing env vars for SCENE_INDEX_* to ensure consistency
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # ALWAYS override existing env vars with .env values for SCENE_INDEX_* vars
                # This ensures .env takes precedence for these specific variables
                if key.startswith('SCENE_INDEX_'):
                    old_value = os.environ.get(key, 'NOT SET')
                    os.environ[key] = value
                    # Debug: only show if value changed
                    if old_value != value and old_value != 'NOT SET':
                        print(f"🔧 Overriding {key}: {old_value} → {value}", file=sys.stderr)
                elif key not in os.environ:
                    os.environ[key] = value

from scene_indexer import SceneIndexer
from search_scenes import calculate_relevance_score

# Semantic test queries - Office/Real-world content
SEMANTIC_QUERIES = [
    ("A person holding a rock", "Objects & Actions"),
    ("A YouTube video shown on a screen", "Media & Technology"),
    ("A shelf with office supplies", "Office Environment"),
    ("Night time in the office", "Time & Environment"),
    ("People in a van", "Vehicles"),
    ("People in a car", "Vehicles"),
    ("Someone speaking on the phone while seated in the desk", "Office Activities"),
    ("An office printer", "Office Equipment"),
    ("People wearing casual clothing", "People & Clothing"),
    ("People wearing office clothing", "People & Clothing"),
    ("A yellow cardigan", "People & Clothing"),
]


def parse_time_filter(time_str: str) -> tuple[datetime | None, datetime | None]:
    """Parse natural language time expressions for filtering.
    
    Args:
        time_str: Time expression like "past 15 minutes", "past 2 hours", "yesterday", "last week"
        
    Returns:
        (start_time, end_time) tuple, or (None, None) if invalid
    """
    if not time_str:
        return None, None
    
    time_str_lower = time_str.lower().strip()
    now = datetime.now()
    
    # Patterns for time expressions
    patterns = {
        'yesterday': (now - timedelta(days=1), now),
        'today': (now.replace(hour=0, minute=0, second=0, microsecond=0), now),
        'past 15 minutes': (now - timedelta(minutes=15), now),
        'past 30 minutes': (now - timedelta(minutes=30), now),
        'past 1 hour': (now - timedelta(hours=1), now),
        'past 2 hours': (now - timedelta(hours=2), now),
        'past 3 hours': (now - timedelta(hours=3), now),
        'past 6 hours': (now - timedelta(hours=6), now),
        'past 12 hours': (now - timedelta(hours=12), now),
        'past 24 hours': (now - timedelta(hours=24), now),
        'past 3 days': (now - timedelta(days=3), now),
        'past week': (now - timedelta(days=7), now),
        'past 2 weeks': (now - timedelta(days=14), now),
        'past month': (now - timedelta(days=30), now),
        'last week': (now - timedelta(days=7), now),
        'last 2 weeks': (now - timedelta(days=14), now),
        'last month': (now - timedelta(days=30), now),
    }
    
    # Check for exact pattern matches
    for pattern, (start, end) in patterns.items():
        if pattern in time_str_lower:
            return start, end
    
    # Check for "past X minutes" pattern (supports decimals like "past 0.25 hours" = 15 minutes)
    minutes_match = re.search(r'past\s+(\d+(?:\.\d+)?)\s+minutes?', time_str_lower)
    if minutes_match:
        minutes = float(minutes_match.group(1))
        return now - timedelta(minutes=minutes), now
    
    # Check for "past X hours" pattern (supports decimals like "past 0.25 hours")
    hours_match = re.search(r'past\s+(\d+(?:\.\d+)?)\s+hours?', time_str_lower)
    if hours_match:
        hours = float(hours_match.group(1))
        return now - timedelta(hours=hours), now
    
    # Check for "past X days" pattern
    days_match = re.search(r'past\s+(\d+)\s+days?', time_str_lower)
    if days_match:
        days = int(days_match.group(1))
        return now - timedelta(days=days), now
    
    # Check for "last X minutes" pattern
    last_minutes_match = re.search(r'last\s+(\d+(?:\.\d+)?)\s+minutes?', time_str_lower)
    if last_minutes_match:
        minutes = float(last_minutes_match.group(1))
        return now - timedelta(minutes=minutes), now
    
    # Check for "last X hours" pattern
    last_hours_match = re.search(r'last\s+(\d+)\s+hours?', time_str_lower)
    if last_hours_match:
        hours = int(last_hours_match.group(1))
        return now - timedelta(hours=hours), now
    
    # Check for "last X days" pattern
    last_days_match = re.search(r'last\s+(\d+)\s+days?', time_str_lower)
    if last_days_match:
        days = int(last_days_match.group(1))
        return now - timedelta(days=days), now
    
    # Check for "X hours ago" pattern
    hours_ago_match = re.search(r'(\d+)\s+hours?\s+ago', time_str_lower)
    if hours_ago_match:
        hours = int(hours_ago_match.group(1))
        start = now - timedelta(hours=hours)
        return start, now
    
    # Check for "X days ago" pattern
    days_ago_match = re.search(r'(\d+)\s+days?\s+ago', time_str_lower)
    if days_ago_match:
        days = int(days_ago_match.group(1))
        start = now - timedelta(days=days)
        return start, now
    
    return None, None


def filter_results_by_time(results: list, start_time: datetime | None, end_time: datetime | None) -> list:
    """Filter search results by timestamp.
    
    Args:
        results: List of search result dictionaries
        start_time: Minimum timestamp (inclusive)
        end_time: Maximum timestamp (inclusive)
        
    Returns:
        Filtered list of results
    """
    if start_time is None and end_time is None:
        return results
    
    filtered = []
    for result in results:
        timestamp = result.get("timestamp", 0)
        if not timestamp:
            continue
        
        try:
            result_time = datetime.fromtimestamp(timestamp)
            
            # Check if result is within time range
            if start_time and result_time < start_time:
                continue
            if end_time and result_time > end_time:
                continue
            
            filtered.append(result)
        except (ValueError, OSError, TypeError):
            # Invalid timestamp, skip
            continue
    
    return filtered


def test_queries_and_generate_report(
    time_filter: str | None = None,
    embedding_model: str | None = None,
    table_name: str | None = None
):
    """Run all semantic queries and generate HTML report.
    
    Args:
        time_filter: Optional time filter string (e.g., "past 2 hours", "yesterday", "last week").
                    Defaults to "past 24 hours" if not specified.
        embedding_model: Optional embedding model name (overrides env var)
        table_name: Optional table name (overrides env var and auto-generation)
    """
    print("=" * 60)
    print("Semantic Query Batch Testing")
    print("=" * 60)
    print()
    
    # Default to no time filter (search all data) if not specified
    # User can explicitly set --time-filter "past 24 hours" if they want time filtering
    if time_filter is None:
        time_filter = ""  # No filter by default - search all data
        print("⏰ No time filter specified - searching all indexed data")
        print("   (Use --time-filter \"past 24 hours\" to filter recent data, or --time-filter \"past 3 days\" for longer window)")
        print()
    elif time_filter == "":
        print("⏰ No time filter specified - searching all indexed data")
        print()
    
    # Parse time filter
    start_time = None
    end_time = None
    time_filter_str = ""
    if time_filter and time_filter.strip():
        start_time, end_time = parse_time_filter(time_filter)
        if start_time is None and end_time is None:
            print(f"⚠️  Warning: Could not parse time filter '{time_filter}', ignoring it")
        else:
            time_filter_str = f" (filtered: {time_filter})"
            print(f"⏰ Time filter: {time_filter}")
            print(f"   From: {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'beginning'}")
            print(f"   To:   {end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else 'now'}")
            print()
    
    # Initialize scene indexer
    from scene_indexer import get_table_name_for_model
    
    # Get model - prioritize argument, then .env, then default
    # Force use of .env value if it exists (already loaded at top of file)
    resolved_model = embedding_model or os.getenv("SCENE_INDEX_MODEL", "text-embedding-3-small")
    
    # Normalize model names (handle short names)
    model_aliases = {
        'e5-large-v2': 'intfloat/e5-large-v2',
        'bge-base-en': 'BAAI/bge-base-en',
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'minilm-l6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    }
    
    # Check if it's a short name and expand it
    if resolved_model in model_aliases:
        resolved_model = model_aliases[resolved_model]
        print(f"📝 Expanded model alias to: {resolved_model}")
    # Also check if it's a partial match (e.g., "e5-large-v2" -> "intfloat/e5-large-v2")
    elif resolved_model == "e5-large-v2" or resolved_model.endswith("/e5-large-v2"):
        if not resolved_model.startswith("intfloat/"):
            resolved_model = "intfloat/e5-large-v2"
            print(f"📝 Expanded model name to: {resolved_model}")
    
    # Always auto-generate table name based on model to ensure dimension match
    # This overrides .env/shell env if it has a mismatched table name (like run.sh and run_amber.sh do)
    expected_table = get_table_name_for_model(resolved_model)
    
    # Get table - prioritize argument, then check if .env/shell has a value
    resolved_table = table_name or os.getenv("SCENE_INDEX_TABLE")
    
    # Always use the model-specific table (override any mismatched values)
    if resolved_table and resolved_table != expected_table:
        print(f"⚠️  Warning: SCENE_INDEX_TABLE={resolved_table} doesn't match model {resolved_model}")
        print(f"   Auto-correcting to use: {expected_table}")
        print()
        resolved_table = expected_table
    elif not resolved_table:
        resolved_table = expected_table
        print(f"📊 Using model-specific table: {resolved_table}")
    else:
        # Table matches expected - use it
        print(f"📊 Using table: {resolved_table}")
    
    print(f"📊 Embedding model: {resolved_model}")
    print()
    
    # Temporarily unset SCENE_INDEX_TABLE in environment so SceneIndexer uses the parameter
    # instead of the environment variable (which might be wrong)
    old_table_env = os.environ.pop("SCENE_INDEX_TABLE", None)
    
    try:
        indexer = SceneIndexer(
            table_name=resolved_table,
            embedding_model=resolved_model
        )
    finally:
        # Restore the environment variable if it was set (for other code that might need it)
        if old_table_env:
            os.environ["SCENE_INDEX_TABLE"] = old_table_env
    
    if not indexer.enabled:
        print("❌ Scene indexer is not enabled. Please check LanceDB configuration.")
        return
    
    print(f"✓ Scene indexer initialized")
    print(f"  Model: {indexer.embedding_model_name} ({'OpenAI API' if indexer.use_openai_api else 'Local'})")
    print(f"  Table: {indexer.table_name}")
    print(f"  Dimension: {indexer.embedding_dim}")
    print(f"✓ Testing {len(SEMANTIC_QUERIES)} semantic queries{time_filter_str}")
    print()
    
    # Run all queries
    all_results = []
    for query, category in SEMANTIC_QUERIES:
        print(f"Testing: '{query}' ({category})...", end=" ")
        
        # Perform search (get more results to account for time filtering)
        results = indexer.query_scenes(query, limit=20, min_similarity=0.0)
        
        # Apply time filter if specified
        if start_time is not None or end_time is not None:
            results = filter_results_by_time(results, start_time, end_time)
        
        # Calculate relevance scores
        scored_results = []
        for result in results:
            score_info = calculate_relevance_score(result, query)
            scored_result = {
                **result,
                "relevance_score": score_info["total_score"],
                "scoring_breakdown": score_info
            }
            scored_results.append(scored_result)
        
        # Sort by relevance
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        all_results.append({
            "query": query,
            "category": category,
            "results": scored_results[:3],  # Top 3 results
            "total_found": len(results)
        })
        
        print(f"Found {len(results)} results")
        
        # Warn if no results found with time filter
        if len(results) == 0 and start_time is not None:
            print("   ⚠️  No results in time range - try a longer window or remove time filter")
    
    print()
    print("=" * 60)
    
    # Summary before generating report
    total_with_results = sum(1 for r in all_results if r['total_found'] > 0)
    if total_with_results == 0 and start_time is not None:
        print("⚠️  No results found for any query in the specified time range.")
        print("   Suggestions:")
        print("   - Try a longer time window: --time-filter \"past 48 hours\"")
        print("   - Try a different time period: --time-filter \"past 3 days\"")
        print("   - Search all data: --time-filter \"\"")
        print()
    
    print("Generating HTML report...")
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports/html")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"semantic_query_test_report_{timestamp}.html"
    report_path = reports_dir / report_filename
    
    # Generate HTML report (pass report_path for image path resolution and metadata)
    html_content = generate_html_report(
        all_results, 
        str(report_path), 
        time_filter=time_filter,
        embedding_model=resolved_model,
        table_name=resolved_table
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Report saved to: {report_path}")
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total queries tested: {len(SEMANTIC_QUERIES)}")
    print(f"  Queries with results: {sum(1 for r in all_results if r['total_found'] > 0)}")
    print(f"  Queries without results: {sum(1 for r in all_results if r['total_found'] == 0)}")
    print()
    print(f"Open {report_path} in your browser to view results with images!")
    print("=" * 60)
    
    return report_path

def generate_html_report(all_results, report_path=None, time_filter=None, embedding_model=None, table_name=None):
    """Generate HTML report with embedded images.
    
    Args:
        all_results: List of query results
        report_path: Optional path to report file (for image path resolution)
        time_filter: Optional time filter string to display in report
        embedding_model: Optional embedding model name to display in report
        table_name: Optional table name to display in report
    """
    # Get time filter info for report
    time_filter_info = ""
    if time_filter:
        start_time, end_time = parse_time_filter(time_filter)
        if start_time or end_time:
            time_filter_info = f"<p><strong>Time Filter:</strong> {time_filter} "
            if start_time:
                time_filter_info += f"(from {start_time.strftime('%Y-%m-%d %H:%M:%S')}) "
            if end_time:
                time_filter_info += f"(to {end_time.strftime('%Y-%m-%d %H:%M:%S')})"
            time_filter_info += "</p>"
    
    # Get model/table info for report
    model_info = ""
    if embedding_model or table_name:
        model_info = "<p><strong>Configuration:</strong> "
        if embedding_model:
            model_info += f"Model: {embedding_model} "
        if table_name:
            model_info += f"| Table: {table_name}"
        model_info += "</p>"
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Query Test Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            background: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
        }
        .query-section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .query-header {
            background: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .query-text {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .query-category {
            font-size: 14px;
            opacity: 0.9;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 5px;
        }
        .stat {
            font-size: 14px;
        }
        .stat-value {
            font-weight: bold;
            color: #2196F3;
        }
        .results-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .result-card {
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            transition: transform 0.2s;
        }
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .result-rank {
            background: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }
        .result-score {
            font-size: 12px;
            color: #666;
        }
        .result-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 5px;
            margin: 10px 0;
            border: 2px solid #ddd;
        }
        .result-text {
            font-size: 13px;
            color: #555;
            margin: 10px 0;
            line-height: 1.5;
        }
        .result-meta {
            font-size: 11px;
            color: #888;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }
        .similarity-info {
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }
        .timestamp {
            text-align: right;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <h1>🎯 Semantic Query Test Results</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """ + (model_info if model_info else "") + """
    """ + (time_filter_info if time_filter_info else "") + """
"""
    
    # Group by category
    categories = {}
    for result in all_results:
        category = result["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    # Generate sections for each category
    for category, queries in categories.items():
        html += f'    <h2>{category}</h2>\n'
        
        for query_result in queries:
            query = query_result["query"]
            results = query_result["results"]
            total_found = query_result["total_found"]
            
            html += f"""    <div class="query-section">
        <div class="query-header">
            <div class="query-text">"{query}"</div>
            <div class="query-category">{category}</div>
        </div>
        <div class="stats">
            <div class="stat">
                <span class="stat-value">{total_found}</span> total results found
            </div>
            <div class="stat">
                <span class="stat-value">{len(results)}</span> results shown
            </div>
        </div>
"""
            
            if results:
                html += '        <div class="results-container">\n'
                
                for idx, result in enumerate(results, 1):
                    similarity = result.get("similarity", 0.0)
                    distance = result.get("distance")
                    relevance = result.get("relevance_score", 0.0)
                    summary = result.get("summary_text", "No summary")
                    frame_paths = result.get("frame_paths", [])
                    timestamp = result.get("timestamp", 0)
                    
                    # Format timestamp
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        time_str = str(timestamp)
                    
                    # Get first frame image and embed as base64
                    image_html = ""
                    if frame_paths:
                        frame_path = frame_paths[0] if isinstance(frame_paths, list) else frame_paths.split(",")[0]
                        # Convert to absolute path
                        if not os.path.isabs(frame_path):
                            frame_path = os.path.join(os.getcwd(), frame_path)
                        
                        # Embed image as base64 data URI for portability
                        if os.path.exists(frame_path):
                            try:
                                with open(frame_path, 'rb') as img_file:
                                    img_data = img_file.read()
                                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                                    # Determine image type from extension
                                    ext = os.path.splitext(frame_path)[1].lower()
                                    mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png' if ext == '.png' else 'image/jpeg'
                                    image_html = f'<img src="data:{mime_type};base64,{img_base64}" alt="Frame" class="result-image" onerror="this.style.display=\'none\'">'
                            except Exception as e:
                                # Fallback to relative path if base64 encoding fails
                                if report_path:
                                    rel_path = os.path.relpath(frame_path, os.path.dirname(os.path.abspath(report_path)))
                                else:
                                    rel_path = frame_path
                                image_html = f'<img src="{rel_path}" alt="Frame" class="result-image" onerror="this.style.display=\'none\'">'
                    
                    # Similarity info
                    sim_info = ""
                    if distance is not None:
                        raw_sim = 1.0 - distance
                        sim_info = f'<div class="similarity-info">Distance: {distance:.3f} | Raw Similarity: {raw_sim:.3f} | Capped: {similarity:.3f}</div>'
                    
                    html += f"""            <div class="result-card">
                <div class="result-header">
                    <span class="result-rank">#{idx}</span>
                    <span class="result-score">Relevance: {relevance:.3f}</span>
                </div>
                {image_html}
                <div class="result-text">{summary[:200]}{'...' if len(summary) > 200 else ''}</div>
                {sim_info}
                <div class="result-meta">
                    Scene ID: {result.get('scene_id', 'N/A')}<br>
                    Level: {result.get('level', 'N/A')}<br>
                    Time: {time_str}
                </div>
            </div>
"""
                
                html += '        </div>\n'
            else:
                html += '        <div class="no-results">❌ No results found for this query</div>\n'
            
            html += '    </div>\n'
    
    html += """</body>
</html>"""
    
    return html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test semantic queries and generate HTML report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all queries with default model (searches all data by default)
  python test_semantic_queries.py
  
  # Test with specific embedding model (searches all data by default)
  python test_semantic_queries.py --embedding-model "BAAI/bge-base-en"
  python test_semantic_queries.py --embedding-model "text-embedding-3-small"
  
  # Test all 4 models and generate comparison reports (searches all data by default)
  python test_semantic_queries.py --all-models
  
  # Add time filter to search recent data only
  python test_semantic_queries.py --time-filter "past 15 minutes"
  python test_semantic_queries.py --time-filter "past 30 minutes"
  python test_semantic_queries.py --time-filter "past 0.25 hours"  # Same as 15 minutes
  python test_semantic_queries.py --time-filter "past 24 hours"
  python test_semantic_queries.py --time-filter "past 2 hours"
  python test_semantic_queries.py --time-filter "yesterday"
  python test_semantic_queries.py --embedding-model "BAAI/bge-base-en" --time-filter "past 3 days"
  
  # Test with specific table (searches all data by default)
  python test_semantic_queries.py --scene-table "scene_embeddings_bge"
  
  # Explicitly search all time (same as default now)
  python test_semantic_queries.py --time-filter ""
  
Supported embedding models:
  - BAAI/bge-base-en (768 dimensions)
  - intfloat/e5-large-v2 (1024 dimensions)
  - sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
  - text-embedding-3-small (1536 dimensions, requires OPENAI_API_KEY)
  
Supported time filters:
  - "past X minutes" (e.g., "past 15 minutes", "past 30 minutes")
  - "past X hours" (e.g., "past 2 hours", "past 12 hours", "past 0.25 hours" = 15 minutes)
  - "past X days" (e.g., "past 3 days", "past 7 days")
  - "last X minutes" (e.g., "last 15 minutes", "last 30 minutes")
  - "last X hours" (e.g., "last 6 hours")
  - "last X days" (e.g., "last 2 days")
  - "yesterday"
  - "today"
  - "past week"
  - "last week"
  - "past month"
  - "last month"
        """
    )
    parser.add_argument(
        "--time-filter",
        type=str,
        default=None,
        help="Filter results by time period (e.g., 'past 2 hours', 'yesterday', 'last week'). Defaults to no filter (search all data) if not specified."
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
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all 4 embedding models and generate comparison reports"
    )
    
    args = parser.parse_args()
    
    # Define all available models
    ALL_MODELS = [
        ("BAAI/bge-base-en", "BGE Base EN (768 dim)"),
        ("intfloat/e5-large-v2", "E5 Large v2 (1024 dim)"),
        ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM L6 v2 (384 dim)"),
        ("text-embedding-3-small", "OpenAI Small (1536 dim)"),
    ]
    
    if args.all_models:
        # Cycle through all models and generate reports
        print("=" * 70)
        print("Testing All Embedding Models")
        print("=" * 70)
        print()
        
        all_reports = []
        for model_name, model_desc in ALL_MODELS:
            print(f"\n{'='*70}")
            print(f"Testing: {model_desc}")
            print(f"Model: {model_name}")
            print(f"{'='*70}\n")
            
            try:
                report_path = test_queries_and_generate_report(
                    time_filter=args.time_filter,
                    embedding_model=model_name,
                    table_name=None  # Auto-generate table name
                )
                if report_path:
                    all_reports.append((model_name, model_desc, report_path))
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 70)
        print("All Models Test Summary")
        print("=" * 70)
        for model_name, model_desc, report_path in all_reports:
            print(f"✓ {model_desc}: {report_path}")
        print("=" * 70)
    else:
        # Single model test
        test_queries_and_generate_report(
            time_filter=args.time_filter,
            embedding_model=args.embedding_model,
            table_name=args.scene_table
        )
