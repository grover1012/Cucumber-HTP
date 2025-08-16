#!/usr/bin/env python3
"""
Test script to check YOLO12 model availability and loading
"""

from ultralytics import YOLO
import sys

def test_yolo12_models():
    """Test different YOLO12 model names to see which ones work."""
    
    # Test different YOLO12 model names
    model_names = [
        "yolo12n-seg.pt",
        "yolo12s-seg.pt", 
        "yolo12m-seg.pt",
        "yolo12l-seg.pt",
        "yolo12x-seg.pt",
        "yolo12n",
        "yolo12s",
        "yolo12m", 
        "yolo12l",
        "yolo12x"
    ]
    
    print("Testing YOLO12 model availability...")
    print("=" * 50)
    
    working_models = []
    
    for model_name in model_names:
        try:
            print(f"Testing: {model_name}")
            model = YOLO(model_name)
            print(f"✅ SUCCESS: {model_name} loaded successfully")
            working_models.append(model_name)
            
            # Get model info
            if hasattr(model, 'model'):
                if hasattr(model.model, 'names'):
                    print(f"   Classes: {len(model.model.names)}")
                if hasattr(model.model, 'nc'):
                    print(f"   Number of classes: {model.model.nc}")
            
        except Exception as e:
            print(f"❌ FAILED: {model_name} - {e}")
        
        print("-" * 30)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Working models: {len(working_models)}")
    for model in working_models:
        print(f"  ✅ {model}")
    
    if working_models:
        print(f"\n🎉 Use one of these working models for training!")
        return working_models[0]  # Return first working model
    else:
        print("\n❌ No YOLO12 models working. Check Ultralytics version.")
        return None

if __name__ == "__main__":
    print("Ultralytics YOLO12 Model Test")
    print("=" * 50)
    
    # Check Ultralytics version
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except:
        print("Could not determine Ultralytics version")
    
    # Test models
    working_model = test_yolo12_models()
    
    if working_model:
        print(f"\n🚀 Recommended model to use: {working_model}")
        print(f"Update your config to use: {working_model}")
    else:
        print("\n❌ No working models found. Please check:")
        print("1. Ultralytics version (should be 8.0.0+)")
        print("2. Internet connection for model download")
        print("3. Python environment")
