#!/usr/bin/env python3
"""
Simple example script demonstrating cucumber trait extraction.
This script shows how to use the pipeline with a single image.
"""

import os
import sys
import numpy as np
import cv2

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.image_utils import load_image, save_image
from utils.trait_extraction import extract_all_traits, validate_measurements
from utils.calibration import calibrate_with_known_object
from utils.ocr_utils import extract_accession_id


def create_sample_image():
    """Create a sample cucumber image for demonstration."""
    # Create a blank image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw a cucumber (green ellipse)
    cv2.ellipse(image, (400, 300), (200, 50), 0, 0, 360, (0, 255, 0), -1)
    
    # Draw a ruler (blue rectangle with markings)
    cv2.rectangle(image, (50, 50), (750, 100), (255, 0, 0), 2)
    for i in range(0, 700, 50):
        cv2.line(image, (100 + i, 50), (100 + i, 100), (255, 0, 0), 2)
    
    # Draw a label (red rectangle)
    cv2.rectangle(image, (50, 500), (200, 550), (0, 0, 255), -1)
    cv2.putText(image, "CU001", (70, 530), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw a color chart (yellow rectangle)
    cv2.rectangle(image, (600, 500), (750, 550), (255, 255, 0), -1)
    
    return image


def main():
    """Main function demonstrating the pipeline."""
    print("=== Cucumber Trait Extraction Example ===\n")
    
    # Create sample image
    print("1. Creating sample cucumber image...")
    image = create_sample_image()
    
    # Save sample image
    os.makedirs("../data/raw_images", exist_ok=True)
    sample_path = "../data/raw_images/sample_cucumber.jpg"
    save_image(image, sample_path)
    print(f"   Sample image saved to: {sample_path}")
    
    # Create masks for demonstration
    print("\n2. Creating detection masks...")
    
    # Cucumber mask
    cucumber_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.ellipse(cucumber_mask, (400, 300), (200, 50), 0, 0, 360, 255, -1)
    
    # Ruler mask
    ruler_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(ruler_mask, (50, 50), (750, 100), 255, -1)
    
    # Label mask
    label_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(label_mask, (50, 500), (200, 550), 255, -1)
    
    print("   Masks created for cucumber, ruler, and label")
    
    # Calibration
    print("\n3. Performing calibration...")
    
    # Assume ruler is 15cm (150mm) long
    ruler_calibration = calibrate_with_known_object(ruler_mask, image, 150.0)
    pixel_to_mm_ratio = ruler_calibration['pixel_to_mm_ratio']
    
    print(f"   Pixel-to-mm ratio: {pixel_to_mm_ratio:.4f}")
    print(f"   Ruler length (pixels): {ruler_calibration['ruler_length_px']:.1f}")
    
    # Trait extraction
    print("\n4. Extracting cucumber traits...")
    
    traits = extract_all_traits(cucumber_mask, image, pixel_to_mm_ratio)
    
    print("   Extracted traits:")
    if 'length' in traits:
        print(f"     Length: {traits['length']:.1f} mm")
    if 'width' in traits:
        print(f"     Width: {traits['width']:.1f} mm")
    if 'aspect_ratio' in traits:
        print(f"     Aspect Ratio: {traits['aspect_ratio']:.2f}")
    if 'area_mm2' in traits:
        print(f"     Area: {traits['area_mm2']:.1f} mm²")
    if 'volume_cm3' in traits:
        print(f"     Volume: {traits['volume_cm3']:.2f} cm³")
    
    # Validation
    print("\n5. Validating measurements...")
    
    validation = validate_measurements(traits)
    
    print("   Validation results:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"     {key}: {status}")
    
    # OCR extraction
    print("\n6. Extracting accession ID...")
    
    accession_result = extract_accession_id(image, label_mask)
    
    print(f"   Accession ID: {accession_result['accession_id']}")
    print(f"   Confidence: {accession_result['confidence']:.3f}")
    print(f"   Is Valid: {accession_result['is_valid']}")
    
    # Create results summary
    print("\n7. Creating results summary...")
    
    results = {
        'calibration': ruler_calibration,
        'cucumber_traits': traits,
        'validation': validation,
        'accession_id': accession_result
    }
    
    # Save results
    os.makedirs("../results", exist_ok=True)
    results_path = "../results/example_results.json"
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   Results saved to: {results_path}")
    
    # Final summary
    print("\n=== EXAMPLE COMPLETED ===")
    print("\nThis example demonstrates:")
    print("• Image creation and loading")
    print("• Mask creation for object detection")
    print("• Ruler-based calibration")
    print("• Trait extraction from segmentation masks")
    print("• Measurement validation")
    print("• OCR for accession ID extraction")
    print("• Results saving and reporting")
    
    print("\nTo use with real images:")
    print("1. Place your cucumber images in data/raw_images/")
    print("2. Train a YOLO model using scripts/train_yolo.py")
    print("3. Run inference using scripts/extract_traits.py")
    print("4. Or use the CucumberTraitExtractor class directly")


if __name__ == "__main__":
    main()
