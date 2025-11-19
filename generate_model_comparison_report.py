#!/usr/bin/env python3
"""
Generate a comprehensive model comparison report including:
- Semantic query results from HTML reports
- VLM API statistics
- LLM summarization statistics
"""

import os
import re
import base64
from datetime import datetime
from pathlib import Path
from html.parser import HTMLParser
from typing import Dict, List, Any, Optional

# Model statistics provided by user
MODEL_STATS = {
    "BAAI/bge-base-en": {
        "html_report": "reports/html/semantic_query_test_report_20251118_153953.html",
        "vlm": {
            "total_requests": 48,
            "successful": 48,
            "failed": 0,
            "success_rate": 100.0,
            "image_processing_avg_ms": 15.17,
            "api_request_avg_ms": 773.25,
            "total_time_avg_ms": 788.43,
            "api_request_min_ms": 386.66,
            "api_request_max_ms": 2384.48,
        },
        "llm": {
            "total_updates": 34,
            "successful": 21,
            "failed": 0,
            "processing_time_avg_ms": 99377.09,
            "processing_time_min_ms": 33421.29,
            "processing_time_max_ms": 188347.40,
        }
    },
    "text-embedding-3-small": {
        "html_report": "reports/html/semantic_query_test_report_20251118_160154.html",
        "vlm": {
            "total_requests": 65,
            "successful": 65,
            "failed": 0,
            "success_rate": 100.0,
            "image_processing_avg_ms": 12.89,
            "api_request_avg_ms": 617.25,
            "total_time_avg_ms": 630.15,
            "api_request_min_ms": 383.54,
            "api_request_max_ms": 4182.81,
        },
        "llm": {
            "total_updates": 24,
            "successful": 16,
            "failed": 0,
            "processing_time_avg_ms": 32462.65,
            "processing_time_min_ms": 12015.53,
            "processing_time_max_ms": 57818.74,
        }
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "html_report": "reports/html/semantic_query_test_report_20251118_161220.html",
        "vlm": {
            "total_requests": 60,
            "successful": 60,
            "failed": 0,
            "success_rate": 100.0,
            "image_processing_avg_ms": 13.30,
            "api_request_avg_ms": 881.87,
            "total_time_avg_ms": 895.18,
            "api_request_min_ms": 389.96,
            "api_request_max_ms": 8345.52,
        },
        "llm": {
            "total_updates": 33,
            "successful": 19,
            "failed": 0,
            "processing_time_avg_ms": 64264.49,
            "processing_time_min_ms": 15928.11,
            "processing_time_max_ms": 152491.44,
        }
    },
    "intfloat/e5-large-v2": {
        "html_report": "reports/html/semantic_query_test_report_20251118_162655.html",
        "vlm": {
            "total_requests": 45,
            "successful": 45,
            "failed": 0,
            "success_rate": 100.0,
            "image_processing_avg_ms": 13.84,
            "api_request_avg_ms": 781.16,
            "total_time_avg_ms": 795.02,
            "api_request_min_ms": 422.38,
            "api_request_max_ms": 2364.16,
        },
        "llm": {
            "total_updates": 34,
            "successful": 20,
            "failed": 0,
            "processing_time_avg_ms": 105421.61,
            "processing_time_min_ms": 14199.56,
            "processing_time_max_ms": 170880.48,
        }
    }
}

# Model dimensions
MODEL_DIMENSIONS = {
    "BAAI/bge-base-en": 768,
    "text-embedding-3-small": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "intfloat/e5-large-v2": 1024,
}


class HTMLQueryExtractor(HTMLParser):
    """Extract query results from HTML report."""
    
    def __init__(self):
        super().__init__()
        self.queries = []
        self.current_query = None
        self.current_category = None
        self.in_query_header = False
        self.in_stats = False
        self.in_result_card = False
        self.current_result = None
        self.current_text = ""
        self.query_results = []
        
    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self.current_category = None
        elif tag == "div":
            attrs_dict = dict(attrs)
            if "class" in attrs_dict:
                classes = attrs_dict["class"].split()
                if "query-header" in classes:
                    self.in_query_header = True
                elif "query-text" in classes:
                    self.current_text = ""
                elif "stats" in classes:
                    self.in_stats = True
                elif "result-card" in classes:
                    self.in_result_card = True
                    self.current_result = {}
        elif tag == "img" and self.in_result_card:
            attrs_dict = dict(attrs)
            if "src" in attrs_dict:
                self.current_result["image_src"] = attrs_dict["src"]
    
    def handle_endtag(self, tag):
        if tag == "h2" and self.current_category:
            pass
        elif tag == "div":
            if self.in_query_header:
                self.in_query_header = False
            elif self.in_stats:
                self.in_stats = False
            elif self.in_result_card:
                if self.current_result:
                    self.query_results.append(self.current_result)
                self.in_result_card = False
                self.current_result = None
    
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_query_header:
            if '"' in data:
                # Extract query text
                match = re.search(r'"([^"]+)"', data)
                if match:
                    self.current_query = match.group(1)
            elif self.current_category is None:
                # Category might be in the same section
                pass
        
        if self.in_result_card and self.current_result:
            if "summary" not in self.current_result:
                self.current_result["summary"] = data
            elif len(data) > len(self.current_result.get("summary", "")):
                self.current_result["summary"] = data


def extract_query_results_from_html(html_path: str) -> List[Dict[str, Any]]:
    """Extract query results from HTML report file."""
    if not os.path.exists(html_path):
        print(f"⚠️  Warning: HTML report not found: {html_path}")
        return []
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Simple extraction using regex (more reliable than HTMLParser for this case)
        queries = []
        
        # Find all query sections
        query_pattern = r'<div class="query-header">.*?<div class="query-text">"([^"]+)"</div>.*?<div class="query-category">([^<]+)</div>'
        query_matches = re.finditer(query_pattern, html_content, re.DOTALL)
        
        for match in query_matches:
            query_text = match.group(1)
            category = match.group(2).strip()
            
            # Find stats for this query
            stats_pattern = r'<span class="stat-value">(\d+)</span> total results found'
            stats_match = re.search(stats_pattern, html_content[match.end():match.end()+500])
            total_found = int(stats_match.group(1)) if stats_match else 0
            
            queries.append({
                "query": query_text,
                "category": category,
                "total_found": total_found
            })
        
        return queries
    except Exception as e:
        print(f"⚠️  Error extracting from {html_path}: {e}")
        return []


def generate_comparison_html(model_stats: Dict[str, Any]) -> str:
    """Generate comprehensive comparison HTML report."""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Model Comparison Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 4px solid #4CAF50;
            padding-bottom: 15px;
            text-align: center;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h3 {
            color: #333;
            margin-top: 30px;
            padding: 10px;
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }
        .model-section {
            background: white;
            margin: 30px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .model-header {
            background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .model-name {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .model-dimension {
            font-size: 14px;
            opacity: 0.9;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .stat-title {
            font-size: 16px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 5px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .stat-row:last-child {
            border-bottom: none;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .stat-value {
            font-weight: bold;
            color: #333;
            font-size: 14px;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .comparison-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        .comparison-table tr:hover {
            background: #f5f5f5;
        }
        .query-results {
            margin-top: 20px;
        }
        .query-item {
            background: #f8f9fa;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
        }
        .query-text {
            font-weight: bold;
            color: #333;
        }
        .query-category {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .query-count {
            color: #4CAF50;
            font-weight: bold;
        }
        .timestamp {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }
        .highlight {
            background: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        .best {
            background: #d4edda;
            color: #155724;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
        .worst {
            background: #f8d7da;
            color: #721c24;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>📊 Embedding Model Comparison Report</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""
    
    # Generate comparison table
    html += """
    <h2>📈 Overall Performance Comparison</h2>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Dimensions</th>
                <th>VLM Avg (ms)</th>
                <th>LLM Avg (ms)</th>
                <th>VLM Success</th>
                <th>LLM Success</th>
                <th>Total Queries</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Calculate best/worst for highlighting
    vlm_times = [stats["vlm"]["api_request_avg_ms"] for stats in model_stats.values()]
    llm_times = [stats["llm"]["processing_time_avg_ms"] for stats in model_stats.values()]
    best_vlm = min(vlm_times)
    best_llm = min(llm_times)
    worst_vlm = max(vlm_times)
    worst_llm = max(llm_times)
    
    for model_name, stats in model_stats.items():
        dimension = MODEL_DIMENSIONS.get(model_name, "N/A")
        vlm_avg = stats["vlm"]["api_request_avg_ms"]
        llm_avg = stats["llm"]["processing_time_avg_ms"]
        vlm_success = stats["vlm"]["success_rate"]
        llm_success = (stats["llm"]["successful"] / stats["llm"]["total_updates"] * 100) if stats["llm"]["total_updates"] > 0 else 0
        
        # Extract query count from HTML if available
        queries = extract_query_results_from_html(stats["html_report"])
        total_queries = len(queries)
        
        vlm_class = "best" if vlm_avg == best_vlm else "worst" if vlm_avg == worst_vlm else ""
        llm_class = "best" if llm_avg == best_llm else "worst" if llm_avg == worst_llm else ""
        
        html += f"""
            <tr>
                <td><strong>{model_name}</strong></td>
                <td>{dimension}</td>
                <td class="{vlm_class}">{vlm_avg:.2f}</td>
                <td class="{llm_class}">{llm_avg:.2f}</td>
                <td>{vlm_success:.1f}%</td>
                <td>{llm_success:.1f}%</td>
                <td>{total_queries}</td>
            </tr>
"""
    
    html += """
        </tbody>
    </table>
"""
    
    # Generate detailed sections for each model
    for model_name, stats in model_stats.items():
        dimension = MODEL_DIMENSIONS.get(model_name, "N/A")
        
        html += f"""
    <div class="model-section">
        <div class="model-header">
            <div class="model-name">{model_name}</div>
            <div class="model-dimension">Vector Dimension: {dimension}</div>
        </div>
        
        <h3>🎥 VLM API Statistics</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">Request Statistics</div>
                <div class="stat-row">
                    <span class="stat-label">Total Requests:</span>
                    <span class="stat-value">{stats['vlm']['total_requests']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Successful:</span>
                    <span class="stat-value">{stats['vlm']['successful']} ({stats['vlm']['success_rate']:.1f}%)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Failed:</span>
                    <span class="stat-value">{stats['vlm']['failed']}</span>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Timing Metrics (ms)</div>
                <div class="stat-row">
                    <span class="stat-label">Image Processing:</span>
                    <span class="stat-value">{stats['vlm']['image_processing_avg_ms']:.2f} (avg)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">API Request:</span>
                    <span class="stat-value">{stats['vlm']['api_request_avg_ms']:.2f} (avg)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Total Time:</span>
                    <span class="stat-value">{stats['vlm']['total_time_avg_ms']:.2f} (avg)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">API Range:</span>
                    <span class="stat-value">{stats['vlm']['api_request_min_ms']:.2f} - {stats['vlm']['api_request_max_ms']:.2f}</span>
                </div>
            </div>
        </div>
        
        <h3>🤖 LLM Summarization Statistics</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">Update Statistics</div>
                <div class="stat-row">
                    <span class="stat-label">Total Updates:</span>
                    <span class="stat-value">{stats['llm']['total_updates']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Successful:</span>
                    <span class="stat-value">{stats['llm']['successful']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Failed:</span>
                    <span class="stat-value">{stats['llm']['failed']}</span>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Processing Time (ms)</div>
                <div class="stat-row">
                    <span class="stat-label">Average:</span>
                    <span class="stat-value">{stats['llm']['processing_time_avg_ms']:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Range:</span>
                    <span class="stat-value">{stats['llm']['processing_time_min_ms']:.2f} - {stats['llm']['processing_time_max_ms']:.2f}</span>
                </div>
            </div>
        </div>
        
        <h3>🔍 Semantic Query Results</h3>
        <div class="query-results">
"""
        
        # Extract and display query results
        queries = extract_query_results_from_html(stats["html_report"])
        if queries:
            for query in queries:
                html += f"""
            <div class="query-item">
                <div class="query-text">"{query['query']}"</div>
                <div class="query-category">Category: {query['category']}</div>
                <div class="query-count">Results Found: {query['total_found']}</div>
            </div>
"""
        else:
            html += """
            <div class="query-item">
                <div class="query-text">No query results available</div>
            </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += """
    </body>
</html>
"""
    
    return html


def main():
    """Generate the comparison report."""
    print("=" * 70)
    print("Generating Embedding Model Comparison Report")
    print("=" * 70)
    print()
    
    # Generate HTML report
    html_content = generate_comparison_html(MODEL_STATS)
    
    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"reports/html/model_comparison_report_{timestamp}.html"
    os.makedirs(os.path.dirname(html_filename), exist_ok=True)
    
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {html_filename}")
    print()
    print("📊 Report includes:")
    print("   - Overall performance comparison table")
    print("   - VLM API statistics for each model")
    print("   - LLM summarization statistics for each model")
    print("   - Semantic query results summary")
    print()
    print("💡 To view the report:")
    print(f"   python serve_report.py")
    print(f"   # Or open directly: {html_filename}")
    print()
    print("📄 To convert to PDF:")
    print(f"   python convert_report_to_pdf.py")


if __name__ == "__main__":
    main()


