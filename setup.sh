#!/bin/bash

echo "================================================"
echo "Real-Time Video Summarization System - Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create reference_faces directory
echo ""
echo "Creating reference_faces directory..."
mkdir -p reference_faces

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set up Langfuse credentials (optional):"
echo "   export LANGFUSE_PUBLIC_KEY='your_key'"
echo "   export LANGFUSE_SECRET_KEY='your_key'"
echo ""
echo "3. Add reference faces to reference_faces/ directory (optional)"
echo ""
echo "4. Run the application:"
echo "   python real_time_summarizer.py"
echo ""

