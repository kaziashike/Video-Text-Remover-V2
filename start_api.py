#!/usr/bin/env python3
"""
API startup script for Video Subtitle Remover
This script will check dependencies and start the API with proper error handling
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'python-multipart',
        'opencv-python',
        'torch',
        'paddleocr'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'paddleocr':
                importlib.util.find_spec('paddleocr')
            else:
                importlib.util.find_spec(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    # Check for paddlepaddle separately
    paddle_available = False
    try:
        importlib.util.find_spec('paddle')
        paddle_available = True
    except ImportError:
        pass
    
    if not paddle_available:
        missing_packages.append('paddlepaddle-gpu')
    
    return missing_packages

def check_models():
    """Check if required model files exist"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, 'backend')
    models_dir = os.path.join(backend_dir, 'models')
    
    required_models = [
        os.path.join(models_dir, 'big-lama'),
        os.path.join(models_dir, 'sttn', 'infer_model.pth'),
        os.path.join(models_dir, 'video', 'ProPainter.pth'),
        os.path.join(models_dir, 'V4', 'ch_det')
    ]
    
    missing_models = []
    
    for model_path in required_models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    return missing_models

def start_api():
    """Start the API server"""
    try:
        # Try to import the app
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app import app
        
        # Start the server
        import uvicorn
        print("Starting Video Subtitle Remover API...")
        print("Access the API at: http://localhost:8000")
        print("API Documentation at: http://localhost:8000/docs")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
        
    except Exception as e:
        print(f"Error starting API: {e}")
        print("Make sure all dependencies are installed and models are available")
        return False
    
    return True

def main():
    print("Video Subtitle Remover API Startup Script")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    missing_packages = check_dependencies()
    if missing_packages:
        print("Missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages with:")
        print("pip install " + " ".join(missing_packages))
        return 1
    
    print("✓ All dependencies found")
    
    # Check models
    print("Checking models...")
    missing_models = check_models()
    if missing_models:
        print("Missing models:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease download the required models.")
        return 1
    
    print("✓ All models found")
    
    # Start API
    print("\nStarting API...")
    if not start_api():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())