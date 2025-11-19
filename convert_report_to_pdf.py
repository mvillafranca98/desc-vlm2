#!/usr/bin/env python3
"""
Convert HTML semantic query test report to PDF.
"""

import os
import sys
from pathlib import Path

def convert_with_playwright(html_path, pdf_path):
    """Convert HTML to PDF using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        
        print(f"Converting {html_path} to PDF...")
        print("This may take a few minutes for large files...")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Load HTML file
            file_url = f"file://{os.path.abspath(html_path)}"
            print(f"Loading: {file_url}")
            page.goto(file_url, wait_until="networkidle")
            
            # Wait for images to load
            print("Waiting for images to load...")
            page.wait_for_timeout(2000)  # Wait 2 seconds for images
            
            # Generate PDF
            print(f"Generating PDF: {pdf_path}")
            page.pdf(
                path=pdf_path,
                format="A4",
                print_background=True,
                margin={
                    "top": "20mm",
                    "right": "15mm",
                    "bottom": "20mm",
                    "left": "15mm"
                }
            )
            
            browser.close()
            print(f"✓ PDF created successfully: {pdf_path}")
            return True
            
    except ImportError:
        print("❌ Playwright not installed.")
        print("Installing playwright...")
        os.system("pip install playwright")
        os.system("playwright install chromium")
        print("Please run this script again.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def convert_with_weasyprint(html_path, pdf_path):
    """Convert HTML to PDF using WeasyPrint (alternative method)."""
    try:
        from weasyprint import HTML
        
        print(f"Converting {html_path} to PDF...")
        print("This may take a few minutes for large files...")
        
        HTML(filename=html_path).write_pdf(pdf_path)
        print(f"✓ PDF created successfully: {pdf_path}")
        return True
        
    except ImportError:
        print("❌ WeasyPrint not installed.")
        print("Installing weasyprint...")
        os.system("pip install weasyprint")
        print("Please run this script again.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    # Find the most recent HTML report in reports/html directory
    reports_html_dir = Path('reports/html')
    reports_html_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for model comparison report first, then semantic query reports
    comparison_files = sorted(reports_html_dir.glob('model_comparison_report_*.html'), reverse=True)
    query_files = sorted(reports_html_dir.glob('semantic_query_test_report_*.html'), reverse=True)
    
    html_file = None
    if comparison_files:
        html_file = comparison_files[0]
        print("📊 Found model comparison report")
    elif query_files:
        html_file = query_files[0]
        print("📊 Found semantic query test report")
    else:
        print("❌ No HTML report found. Please run test_semantic_queries.py or generate_model_comparison_report.py first.")
        sys.exit(1)
    
    # Save PDF to reports/pdf directory
    reports_pdf_dir = Path('reports/pdf')
    reports_pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_file = reports_pdf_dir / html_file.with_suffix('.pdf').name
    
    print("=" * 60)
    print("HTML to PDF Converter")
    print("=" * 60)
    print(f"📄 Input:  {html_file}")
    print(f"📄 Output: {pdf_file}")
    print()
    
    # Try Playwright first (better for complex HTML with embedded images)
    if convert_with_playwright(html_file, pdf_file):
        print()
        print("=" * 60)
        print(f"✅ Success! PDF saved to: {pdf_file}")
        print(f"   File size: {os.path.getsize(pdf_file) / (1024*1024):.1f} MB")
        print("=" * 60)
        return
    
    # Fallback to WeasyPrint
    print("\nTrying WeasyPrint as alternative...")
    if convert_with_weasyprint(html_file, pdf_file):
        print()
        print("=" * 60)
        print(f"✅ Success! PDF saved to: {pdf_file}")
        print(f"   File size: {os.path.getsize(pdf_file) / (1024*1024):.1f} MB")
        print("=" * 60)
        return
    
    print()
    print("=" * 60)
    print("❌ Could not convert to PDF.")
    print()
    print("Alternative: Use your browser's Print to PDF feature:")
    print(f"1. Open: http://localhost:8000/reports/html/{html_file.name}")
    print("2. Press Cmd+P (Mac) or Ctrl+P (Windows/Linux)")
    print("3. Select 'Save as PDF'")
    print("4. Click 'Save'")
    print("=" * 60)

if __name__ == "__main__":
    main()


