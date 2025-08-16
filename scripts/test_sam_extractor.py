#!/usr/bin/env python3
"""
Test script for SAM-enhanced trait extractor
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.enhanced_trait_extractor import EnhancedTraitExtractor


def main():
    """Test the SAM-enhanced trait extractor."""
    
    # Model and image paths
    model_path = "models/yolo12/cucumber_traits/weights/best.pt"
    test_image = "data/annotations/test/images/AM030_YF_2021_jpg.rf.541ac98d17535ed79e21ac842779f108.jpg"
    output_dir = "results/sam_test"
    
    print("🧪 Testing SAM-Enhanced Trait Extractor")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"✅ Model: {model_path}")
    print(f"✅ Test image: {test_image}")
    print(f"✅ Output directory: {output_dir}")
    
    try:
        # Initialize extractor
        print("\n🔄 Initializing enhanced trait extractor with SAM2...")
        extractor = EnhancedTraitExtractor(model_path)
        
        # Process image
        print("\n🔄 Processing image with SAM...")
        results = extractor.process_image(test_image, output_dir)
        
        # Print results
        print("\n🎉 Processing completed!")
        print(f"Segmentation Model: {results['segmentation_model']}")
        
        if results['cucumber_traits']:
            print("\n🥒 Cucumber Traits Extracted:")
            for key, value in results['cucumber_traits'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        
        # List output files
        if os.path.exists(output_dir):
            print("\n📋 Output files:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size} bytes)")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
