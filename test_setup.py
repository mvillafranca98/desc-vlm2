#!/usr/bin/env python3
"""
Test script to verify installation and dependencies
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    tests = []
    
    # Core dependencies
    try:
        import cv2
        tests.append(("OpenCV", "✓", cv2.__version__))
    except ImportError as e:
        tests.append(("OpenCV", "✗", str(e)))
    
    try:
        import numpy
        tests.append(("NumPy", "✓", numpy.__version__))
    except ImportError as e:
        tests.append(("NumPy", "✗", str(e)))
    
    try:
        import PIL
        tests.append(("Pillow", "✓", PIL.__version__))
    except ImportError as e:
        tests.append(("Pillow", "✗", str(e)))
    
    # AI/ML
    try:
        import openai
        tests.append(("OpenAI", "✓", openai.__version__))
    except ImportError as e:
        tests.append(("OpenAI", "✗", str(e)))
    
    try:
        import transformers
        tests.append(("Transformers", "✓", transformers.__version__))
    except ImportError as e:
        tests.append(("Transformers", "✗", str(e)))
    
    try:
        import torch
        tests.append(("PyTorch", "✓", torch.__version__))
    except ImportError as e:
        tests.append(("PyTorch", "✗", str(e)))
    
    # Face recognition
    try:
        import face_recognition
        tests.append(("Face Recognition", "✓", "OK"))
    except ImportError as e:
        tests.append(("Face Recognition", "✗", str(e)))
    
    # Langfuse
    try:
        import langfuse
        tests.append(("Langfuse", "✓", langfuse.__version__))
    except ImportError as e:
        tests.append(("Langfuse", "✗", str(e)))
    
    # Print results
    print("\n" + "="*60)
    print("Dependency Check Results")
    print("="*60)
    
    for name, status, version in tests:
        print(f"{name:20s} {status:3s} {version}")
    
    print("="*60)
    
    # Check if critical dependencies are missing
    failed = [name for name, status, _ in tests if status == "✗"]
    if failed:
        print(f"\n⚠️  Missing dependencies: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed successfully!")
        return True


def test_camera():
    """Test camera access."""
    print("\nTesting camera access...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("✓ Camera 0 is accessible")
                return True
            else:
                print("✗ Camera 0 opened but cannot read frames")
                return False
        else:
            print("✗ Cannot open camera 0")
            print("  Try: python real_time_summarizer.py --camera 1")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")
    
    tests = []
    
    try:
        from vlm_client_module import VLMClientWrapper
        tests.append(("vlm_client_module", "✓"))
    except ImportError as e:
        tests.append(("vlm_client_module", f"✗ {e}"))
    
    try:
        from llm_summarizer import LLMSummarizer
        tests.append(("llm_summarizer", "✓"))
    except ImportError as e:
        tests.append(("llm_summarizer", f"✗ {e}"))
    
    try:
        from face_recognition_module import FaceRecognizer
        tests.append(("face_recognition_module", "✓"))
    except ImportError as e:
        tests.append(("face_recognition_module", f"✗ {e}"))
    
    try:
        from langfuse_tracker import LangfuseTracker
        tests.append(("langfuse_tracker", "✓"))
    except ImportError as e:
        tests.append(("langfuse_tracker", f"✗ {e}"))
    
    print("="*60)
    print("Custom Modules Check")
    print("="*60)
    
    for name, status in tests:
        print(f"{name:30s} {status}")
    
    print("="*60)
    
    failed = [name for name, status in tests if "✗" in status]
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All custom modules loaded successfully!")
        return True


def test_environment():
    """Test environment variables."""
    print("\nTesting environment...")
    
    langfuse_host = os.getenv("LANGFUSE_HOST")
    langfuse_pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_sk = os.getenv("LANGFUSE_SECRET_KEY")
    
    print("="*60)
    print("Environment Variables")
    print("="*60)
    
    if langfuse_host:
        print(f"LANGFUSE_HOST: ✓ {langfuse_host}")
    else:
        print("LANGFUSE_HOST: ⚠️  Not set (optional)")
    
    if langfuse_pk:
        print(f"LANGFUSE_PUBLIC_KEY: ✓ {langfuse_pk[:20]}...")
    else:
        print("LANGFUSE_PUBLIC_KEY: ⚠️  Not set (optional)")
    
    if langfuse_sk:
        print("LANGFUSE_SECRET_KEY: ✓ (hidden)")
    else:
        print("LANGFUSE_SECRET_KEY: ⚠️  Not set (optional)")
    
    print("="*60)
    
    if not all([langfuse_host, langfuse_pk, langfuse_sk]):
        print("\n⚠️  Langfuse tracking will be disabled")
        print("To enable, set environment variables:")
        print("  export LANGFUSE_HOST='https://cloud.langfuse.com'")
        print("  export LANGFUSE_PUBLIC_KEY='your_key'")
        print("  export LANGFUSE_SECRET_KEY='your_key'")
    else:
        print("\n✓ Langfuse environment configured")
    
    return True


def test_torch_device():
    """Test PyTorch device availability."""
    print("\nTesting PyTorch device...")
    
    try:
        import torch
        
        print("="*60)
        print("PyTorch Device Check")
        print("="*60)
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA not available")
        
        if torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) available")
        else:
            print("✗ MPS not available")
        
        print(f"ℹ️  Default device: CPU")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch device test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Real-Time Video Summarization System - Setup Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_imports()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Camera", test_camera()))
    results.append(("Environment", test_environment()))
    results.append(("PyTorch Device", test_torch_device()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    print("="*60)
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the application.")
        print("\nNext step:")
        print("  python real_time_summarizer.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check camera permissions")
        print("  3. Set Langfuse environment variables (optional)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

