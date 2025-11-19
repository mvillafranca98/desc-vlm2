#!/usr/bin/env python3
"""
Simple HTTP server to serve the HTML report.
This allows the browser to load images properly.
"""

import http.server
import socketserver
import webbrowser
import sys
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow local file access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        super().end_headers()

def main():
    # Find the most recent HTML report in reports/html directory
    reports_html_dir = Path('reports/html')
    reports_html_dir.mkdir(parents=True, exist_ok=True)
    
    html_files = sorted(reports_html_dir.glob('semantic_query_test_report_*.html'), reverse=True)
    
    if not html_files:
        print("❌ No HTML report found. Please run test_semantic_queries.py first.")
        sys.exit(1)
    
    report_file = html_files[0]
    print("=" * 60)
    print("Starting HTTP Server for HTML Report")
    print("=" * 60)
    print(f"📄 Serving: {report_file}")
    print(f"🌐 Server: http://localhost:{PORT}")
    print()
    print("The report will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()
    
    # Change to project root directory (not the report directory)
    # This allows the server to serve files from the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Start server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        # Open browser with relative path from project root
        relative_path = report_file.relative_to(project_root)
        webbrowser.open(f'http://localhost:{PORT}/{relative_path}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            sys.exit(0)

if __name__ == "__main__":
    main()


