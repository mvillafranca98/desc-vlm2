#!/usr/bin/env python3
"""
Batch test semantic queries and generate HTML report with images.
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
from scene_indexer import SceneIndexer
from search_scenes import calculate_relevance_score

# All semantic test queries - Based on The Office U.S. TV show content
SEMANTIC_QUERIES = [
    # Office Setting & Environment
    ("office workspace with desks and computers", "Office Setting"),
    ("corporate office environment", "Office Setting"),
    ("indoor workspace with office furniture", "Office Setting"),
    ("business office setting", "Office Setting"),
    ("professional workplace interior", "Office Setting"),
    ("office cubicles and workstations", "Office Setting"),
    ("conference room or meeting space", "Office Setting"),
    ("reception area or front desk", "Office Setting"),
    
    # People & Characters (semantic variations)
    ("person sitting at desk", "People & Characters"),
    ("individual working at computer", "People & Characters"),
    ("office worker at workstation", "People & Characters"),
    ("person in business casual attire", "People & Characters"),
    ("professional dressed individual", "People & Characters"),
    ("person wearing business clothing", "People & Characters"),
    ("colleague or coworker", "People & Characters"),
    ("multiple people in office", "People & Characters"),
    
    # Office Activities & Behaviors
    ("people having a conversation", "Office Activities"),
    ("individuals engaged in discussion", "Office Activities"),
    ("people talking or speaking", "Office Activities"),
    ("person looking at camera", "Office Activities"),
    ("individual making facial expression", "Office Activities"),
    ("people in a meeting", "Office Activities"),
    ("group discussion or conversation", "Office Activities"),
    ("person typing or using computer", "Office Activities"),
    ("individual on phone call", "Office Activities"),
    ("person writing or taking notes", "Office Activities"),
    ("people laughing or smiling", "Office Activities"),
    ("person showing emotion or reaction", "Office Activities"),
    
    # Office Furniture & Objects
    ("desk with computer monitor", "Office Objects"),
    ("office chair and workstation", "Office Objects"),
    ("office desk with items", "Office Objects"),
    ("computer screen or monitor", "Office Objects"),
    ("office supplies or stationery", "Office Objects"),
    ("telephone or phone on desk", "Office Objects"),
    ("office equipment or electronics", "Office Objects"),
    ("whiteboard or bulletin board", "Office Objects"),
    
    # Emotions & Expressions (Office-specific)
    ("person with awkward expression", "Emotions & Expressions"),
    ("individual showing confusion", "Emotions & Expressions"),
    ("person with surprised look", "Emotions & Expressions"),
    ("individual displaying frustration", "Emotions & Expressions"),
    ("person with amused expression", "Emotions & Expressions"),
    ("individual showing disbelief", "Emotions & Expressions"),
    ("person with serious or concerned look", "Emotions & Expressions"),
    ("person making eye contact with camera", "Emotions & Expressions"),
    
    # Office Interactions & Scenarios
    ("awkward office moment", "Office Interactions"),
    ("humorous workplace situation", "Office Interactions"),
    ("office prank or joke", "Office Interactions"),
    ("tense or uncomfortable conversation", "Office Interactions"),
    ("casual office interaction", "Office Interactions"),
    ("professional meeting or presentation", "Office Interactions"),
    ("people collaborating or working together", "Office Interactions"),
    ("office social interaction", "Office Interactions"),
    
    # Complex Multi-Concept Queries
    ("person sitting at desk talking to someone", "Complex Scenarios"),
    ("office worker having conversation in workspace", "Complex Scenarios"),
    ("multiple people in office having discussion", "Complex Scenarios"),
    ("individual at computer in office setting", "Complex Scenarios"),
    ("person in business attire in office environment", "Complex Scenarios"),
    ("office scene with people and furniture", "Complex Scenarios"),
    ("workplace setting with employees and desks", "Complex Scenarios"),
    ("professional office environment with people working", "Complex Scenarios"),
    
    # Abstract Concepts
    ("workplace atmosphere", "Abstract Concepts"),
    ("office culture or environment", "Abstract Concepts"),
    ("professional setting", "Abstract Concepts"),
    ("corporate workspace", "Abstract Concepts"),
    ("business environment", "Abstract Concepts"),
]

def test_queries_and_generate_report():
    """Run all semantic queries and generate HTML report."""
    print("=" * 60)
    print("Semantic Query Batch Testing")
    print("=" * 60)
    print()
    
    # Initialize scene indexer
    indexer = SceneIndexer(
        table_name=os.getenv("SCENE_INDEX_TABLE", "scene_embeddings"),
        embedding_model=os.getenv("SCENE_INDEX_MODEL", "intfloat/e5-large-v2")
    )
    
    if not indexer.enabled:
        print("❌ Scene indexer is not enabled. Please check LanceDB configuration.")
        return
    
    print(f"✓ Scene indexer initialized")
    print(f"✓ Testing {len(SEMANTIC_QUERIES)} semantic queries")
    print()
    
    # Run all queries
    all_results = []
    for query, category in SEMANTIC_QUERIES:
        print(f"Testing: '{query}' ({category})...", end=" ")
        
        # Perform search
        results = indexer.query_scenes(query, limit=5, min_similarity=0.0)
        
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
    
    print()
    print("=" * 60)
    print("Generating HTML report...")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"semantic_query_test_report_{timestamp}.html"
    
    # Generate HTML report (pass report_path for image path resolution)
    html_content = generate_html_report(all_results, report_path)
    
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

def generate_html_report(all_results, report_path=None):
    """Generate HTML report with embedded images."""
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
    test_queries_and_generate_report()

