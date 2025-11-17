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
    # Find the most recent HTML report
    html_files = sorted(Path('.').glob('semantic_query_test_report_*.html'), reverse=True)
    
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
    
    # Change to the directory containing the report
    os.chdir(os.path.dirname(os.path.abspath(report_file)))
    
    # Start server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}/{os.path.basename(report_file)}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            sys.exit(0)

if __name__ == "__main__":
    main()


