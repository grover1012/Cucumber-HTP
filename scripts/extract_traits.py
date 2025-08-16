#!/usr/bin/env python3
"""
Main script for extracting cucumber traits from images.
Uses trained YOLO model to detect objects and extract phenotypic measurements.
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.trait_extractor import CucumberTraitExtractor


def main():
    """Main function for trait extraction."""
    parser = argparse.ArgumentParser(
        description="Extract phenotypic traits from cucumber images using YOLO"
    )
    
    parser.add_argument(
        "--model", 
        required=True, 
        help="Path to trained YOLO model (.pt file)"
    )
    
    parser.add_argument(
        "--image", 
        help="Path to single image for processing"
    )
    
    parser.add_argument(
        "--image-dir", 
        help="Directory containing images for batch processing"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    
    parser.add_argument(
        "--batch", 
        action="store_true",
        help="Process images in batch mode"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    if args.image and args.image_dir:
        parser.error("Cannot specify both --image and --image-dir")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Initialize trait extractor
    try:
        extractor = CucumberTraitExtractor(args.model, args.confidence)
        print(f"Trait extractor initialized with model: {args.model}")
    except Exception as e:
        print(f"Error initializing trait extractor: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image:
        # Process single image
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        print(f"Processing single image: {args.image}")
        
        try:
            results = extractor.process_image(args.image, args.output_dir)
            
            # Print summary
            print("\n=== PROCESSING RESULTS ===")
            print(f"Image: {args.image}")
            
            if 'detections' in results:
                detections = results['detections']
                print(f"\nDetections:")
                for class_name, detection in detections.items():
                    if detection:
                        print(f"  {class_name}: {detection['confidence']:.3f}")
                    else:
                        print(f"  {class_name}: Not detected")
            
            if 'cucumber_traits' in results and results['cucumber_traits']:
                traits = results['cucumber_traits']
                print(f"\nCucumber Traits:")
                if 'length' in traits:
                    print(f"  Length: {traits['length']:.2f} mm")
                if 'width' in traits:
                    print(f"  Width: {traits['width']:.2f} mm")
                if 'aspect_ratio' in traits:
                    print(f"  Aspect Ratio: {traits['aspect_ratio']:.2f}")
                if 'area_mm2' in traits:
                    print(f"  Area: {traits['area_mm2']:.2f} mm²")
                if 'volume_cm3' in traits:
                    print(f"  Volume: {traits['volume_cm3']:.2f} cm³")
            
            if 'accession_id' in results and results['accession_id'].get('accession_id'):
                accession = results['accession_id']
                print(f"\nAccession ID: {accession['accession_id']}")
                print(f"Confidence: {accession['confidence']:.3f}")
                if accession.get('was_corrected'):
                    print(f"Corrected from: {accession.get('original', 'Unknown')}")
            
            if 'calibration' in results:
                cal = results['calibration']
                if 'ruler' in cal:
                    ruler_cal = cal['ruler']
                    print(f"\nCalibration:")
                    print(f"  Pixel-to-mm ratio: {ruler_cal.get('pixel_to_mm_ratio', 0):.4f}")
                    print(f"  Confidence: {ruler_cal.get('confidence', 0):.3f}")
            
            print(f"\nResults saved to: {args.output_dir}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            sys.exit(1)
    
    elif args.image_dir:
        # Process images in batch
        if not os.path.exists(args.image_dir):
            print(f"Error: Image directory not found: {args.image_dir}")
            sys.exit(1)
        
        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(args.image_dir).glob(f"*{ext}"))
            image_files.extend(Path(args.image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in directory: {args.image_dir}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images for batch processing")
        
        # Process images
        try:
            batch_results = extractor.batch_process(
                [str(f) for f in image_files], 
                args.output_dir
            )
            
            # Create summary report
            summary = extractor.create_summary_report(batch_results)
            
            # Save summary report
            summary_path = os.path.join(args.output_dir, "batch_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print(f"\nBatch processing completed!")
            print(f"Summary report saved to: {summary_path}")
            print(f"All results saved to: {args.output_dir}")
            
            # Print summary to console
            print("\n" + summary)
            
        except Exception as e:
            print(f"Error during batch processing: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
