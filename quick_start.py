#!/usr/bin/env python3
"""
Quick Start Script for Cucumber Trait Extraction

This script provides a quick demonstration of the cucumber trait extraction pipeline.
It creates a sample image, runs the analysis, and shows the results.
"""

import os
import sys
import numpy as np
import cv2
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_cucumber_image():
    """Create a realistic sample cucumber image for demonstration."""
    # Create a high-quality sample image
    image = np.ones((800, 1200, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some texture to background
    noise = np.random.randint(0, 20, (800, 1200, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Draw cucumber (green with texture)
    cv2.ellipse(image, (600, 400), (300, 60), 0, 0, 360, (0, 180, 0), -1)
    # Add cucumber texture
    for i in range(0, 600, 20):
        cv2.line(image, (300 + i, 340), (300 + i, 460), (0, 160, 0), 1)
    
    # Draw ruler (blue with markings)
    cv2.rectangle(image, (50, 50), (1150, 120), (255, 0, 0), 3)
    # Add ruler markings every 50 pixels
    for i in range(0, 1100, 50):
        cv2.line(image, (100 + i, 50), (100 + i, 120), (255, 0, 0), 2)
        # Add numbers
        cv2.putText(image, str(i//50), (95 + i, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw label (red with text)
    cv2.rectangle(image, (50, 650), (250, 750), (0, 0, 255), -1)
    cv2.putText(image, "CU001", (80, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Draw color chart (multicolor patches)
    colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128)]
    for i, color in enumerate(colors):
        x = 900 + (i % 4) * 60
        y = 650 + (i // 4) * 50
        cv2.rectangle(image, (x, y), (x + 50, y + 40), color, -1)
        cv2.rectangle(image, (x, y), (x + 50, y + 40), (0, 0, 0), 1)
    
    return image

def main():
    """Main quick start function."""
    print("=== Cucumber Trait Extraction - Quick Start ===\n")
    
    # Create directories
    os.makedirs("data/raw_images", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create sample image
    print("1. Creating sample cucumber image...")
    image = create_sample_cucumber_image()
    
    # Save sample image
    sample_path = "data/raw_images/sample_cucumber.jpg"
    cv2.imwrite(sample_path, image)
    print(f"   Sample image saved to: {sample_path}")
    
    # Create detection masks
    print("\n2. Creating detection masks...")
    
    height, width = image.shape[:2]
    
    # Cucumber mask
    cucumber_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(cucumber_mask, (600, 400), (300, 60), 0, 0, 360, 255, -1)
    
    # Ruler mask
    ruler_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(ruler_mask, (50, 50), (1150, 120), 255, -1)
    
    # Label mask
    label_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(label_mask, (50, 650), (250, 750), 255, -1)
    
    # Color chart mask
    color_chart_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(color_chart_mask, (900, 650), (1140, 750), 255, -1)
    
    print("   Detection masks created")
    
    # Import utility functions
    try:
        from utils.calibration import calibrate_with_known_object
        from utils.trait_extraction import extract_all_traits, validate_measurements
        from utils.ocr_utils import extract_accession_id
        
        print("\n3. Running trait extraction pipeline...")
        
        # Calibration
        print("   - Performing ruler calibration...")
        ruler_calibration = calibrate_with_known_object(ruler_mask, image, 150.0)
        pixel_to_mm_ratio = ruler_calibration['pixel_to_mm_ratio']
        print(f"     Pixel-to-mm ratio: {pixel_to_mm_ratio:.4f}")
        
        # Trait extraction
        print("   - Extracting cucumber traits...")
        traits = extract_all_traits(cucumber_mask, image, pixel_to_mm_ratio)
        
        print("     Extracted traits:")
        if 'length' in traits:
            print(f"       Length: {traits['length']:.1f} mm")
        if 'width' in traits:
            print(f"       Width: {traits['width']:.1f} mm")
        if 'aspect_ratio' in traits:
            print(f"       Aspect Ratio: {traits['aspect_ratio']:.2f}")
        if 'area_mm2' in traits:
            print(f"       Area: {traits['area_mm2']:.1f} mm²")
        
        # Validation
        print("   - Validating measurements...")
        validation = validate_measurements(traits)
        valid_count = sum(validation.values())
        total_count = len(validation)
        print(f"     Validation: {valid_count}/{total_count} checks passed")
        
        # OCR
        print("   - Extracting accession ID...")
        accession_result = extract_accession_id(image, label_mask)
        print(f"     Accession ID: {accession_result['accession_id']}")
        print(f"     Confidence: {accession_result['confidence']:.3f}")
        
        # Save results
        results = {
            'calibration': ruler_calibration,
            'cucumber_traits': traits,
            'validation': validation,
            'accession_id': accession_result
        }
        
        results_path = "results/quick_start_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n   Results saved to: {results_path}")
        
        print("\n4. Pipeline completed successfully!")
        
    except ImportError as e:
        print(f"\n3. Error importing modules: {e}")
        print("   Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("   And that Tesseract OCR is installed on your system.")
    
    # Final instructions
    print("\n=== Next Steps ===")
    print("\nTo use with your own images:")
    print("1. Place cucumber images in data/raw_images/")
    print("2. Create YOLO format annotations")
    print("3. Train model: python scripts/train_yolo.py --config configs/training_config.yaml")
    print("4. Run inference: python scripts/extract_traits.py --model models/best.pt --image-dir data/raw_images")
    
    print("\nFor more information, see:")
    print("- docs/workflow_guide.md")
    print("- examples/simple_example.py")
    print("- README.md")
    
    print("\n=== Quick Start Complete ===")

if __name__ == "__main__":
    main()
