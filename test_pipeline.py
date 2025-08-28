#!/usr/bin/env python3
"""
Test script for the Cucumber HTP Pipeline
This script verifies that all components can be imported and basic functionality works.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core components
        from src.edge_aug import EdgeBoost, edgeboost
        print("  ✅ EdgeBoost modules imported")
        
        from src.edge_losses import EdgeLoss, edge_loss_simple
        print("  ✅ EdgeLoss modules imported")
        
        from src.pose import rotate_to_major_axis
        print("  ✅ Pose modules imported")
        
        from src.scale_ruler import RulerDetector, find_ppcm_simple
        print("  ✅ Scale ruler modules imported")
        
        from src.traits import CucumberTraitExtractor
        print("  ✅ Trait extraction modules imported")
        
        from src.ocr_label import AccessionLabelReader
        print("  ✅ OCR modules imported")
        
        from src.train_seg import train, create_edge_training_config
        print("  ✅ Training modules imported")
        
        from src.infer_seg import CucumberPhenotypingPipeline
        print("  ✅ Inference modules imported")
        
        from src.run_pipeline import main as run_pipeline
        print("  ✅ Pipeline driver imported")
        
        print("🎉 All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        import numpy as np
        import cv2
        
        # Test EdgeBoost
        from src.edge_aug import edgeboost
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced_img = edgeboost(test_img)
        print("  ✅ EdgeBoost functionality test passed")
        
        # Test EdgeLoss
        from src.edge_losses import EdgeLoss
        edge_loss_fn = EdgeLoss(lambda_edge=0.3)
        print("  ✅ EdgeLoss initialization test passed")
        
        # Test trait extractor
        from src.traits import CucumberTraitExtractor
        extractor = CucumberTraitExtractor()
        print("  ✅ Trait extractor initialization test passed")
        
        # Test pose normalization
        from src.pose import rotate_to_major_axis
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[40:60, 30:70] = 255  # Create a rectangular mask
        rotated_img, rotated_mask, angle = rotate_to_major_axis(test_img, test_mask)
        print("  ✅ Pose normalization test passed")
        
        # Test ruler detector
        from src.scale_ruler import RulerDetector
        detector = RulerDetector()
        print("  ✅ Ruler detector initialization test passed")
        
        # Test OCR reader
        from src.ocr_label import AccessionLabelReader
        reader = AccessionLabelReader()
        print("  ✅ OCR reader initialization test passed")
        
        print("🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_pipeline_config():
    """Test pipeline configuration creation."""
    print("\n⚙️ Testing pipeline configuration...")
    
    try:
        from src.train_seg import create_edge_training_config
        
        config = create_edge_training_config(
            data_yaml="data/data.yaml",
            model="yolov8m-seg.pt",
            epochs=150,
            imgsz=1024,
            batch=8
        )
        
        required_keys = ['data_yaml', 'model', 'epochs', 'imgsz', 'batch', 'edge_lambda']
        for key in required_keys:
            if key not in config:
                print(f"  ❌ Missing config key: {key}")
                return False
        
        print("  ✅ Pipeline configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🥒 Cucumber HTP Pipeline - Component Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Pipeline Configuration", test_pipeline_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\n💡 Next steps:")
        print("  1. Prepare your dataset in YOLO segmentation format")
        print("  2. Train your model: python src/run_pipeline.py --stage train")
        print("  3. Run inference: python src/run_pipeline.py --stage infer")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
