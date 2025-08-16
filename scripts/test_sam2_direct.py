#!/usr/bin/env python3
"""
Direct SAM2 Test Script
Test SAM2 model loading and segmentation without other dependencies
"""

import os
import sys
import numpy as np
import cv2

# Add SAM2 to path
sam2_path = os.path.join(os.getcwd(), "sam2")
sys.path.insert(0, sam2_path)

def test_sam2_loading():
    """Test SAM2 model loading directly."""
    print("🧪 Testing SAM2 Model Loading")
    print("=" * 50)
    
    try:
        # Import SAM2 modules
        print("📦 Importing SAM2 modules...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 modules imported successfully")
        
        # Check model file
        model_path = "models/sam2.1_hiera_tiny.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file found: {model_path}")
        
        # Set Hydra config path
        os.environ['HYDRA_CONFIG_PATH'] = os.path.join(os.getcwd(), 'sam2')
        print(f"🔧 Set HYDRA_CONFIG_PATH: {os.environ['HYDRA_CONFIG_PATH']}")
        
        # Load SAM2 model
        print("🔄 Loading SAM2 model...")
        config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
        
        sam2_model = build_sam2(config_name, checkpoint=model_path, device="cpu")
        print("✅ SAM2 model built successfully")
        print("✅ Model already on CPU")
        
        # Create predictor
        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ SAM2 predictor created successfully")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error loading SAM2: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_sam2_segmentation(predictor):
    """Test SAM2 segmentation on a simple image."""
    print("\n🎯 Testing SAM2 Segmentation")
    print("=" * 50)
    
    try:
        # Create a simple test image (100x100 white square on black background)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        print(f"📸 Created test image: {test_image.shape}")
        
        # Set image in predictor
        print("🔄 Setting image in predictor...")
        predictor.set_image(test_image)
        print("✅ Image set successfully")
        
        # Test point prompt (center of white square)
        point_coords = np.array([[50, 50]])  # Center point
        point_labels = np.array([1])  # Positive point
        
        print(f"🎯 Testing point prompt at {point_coords}")
        
        # Generate mask
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        
        print(f"✅ Mask generated successfully!")
        print(f"   Mask shape: {masks[0].shape}")
        print(f"   Mask score: {scores[0]:.4f}")
        print(f"   Mask sum: {np.sum(masks[0])}")
        
        # Test box prompt
        print("\n📦 Testing box prompt...")
        box = np.array([20, 20, 80, 80])  # [x1, y1, x2, y2]
        
        masks_box, scores_box, logits_box = predictor.predict(
            box=box,
            multimask_output=False
        )
        
        print(f"✅ Box mask generated successfully!")
        print(f"   Mask shape: {masks_box[0].shape}")
        print(f"   Mask score: {scores_box[0]:.4f}")
        print(f"   Mask sum: {np.sum(masks_box[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in segmentation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 SAM2 Direct Test")
    print("=" * 60)
    
    # Test SAM2 loading
    predictor = test_sam2_loading()
    
    if predictor is None:
        print("\n❌ SAM2 loading failed. Cannot proceed with segmentation test.")
        return
    
    # Test segmentation
    success = test_sam2_segmentation(predictor)
    
    if success:
        print("\n🎉 SAM2 is working perfectly!")
        print("✅ Model loading: SUCCESS")
        print("✅ Segmentation: SUCCESS")
        print("\n🚀 Ready to integrate with your YOLO12 pipeline!")
    else:
        print("\n❌ SAM2 segmentation test failed.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
