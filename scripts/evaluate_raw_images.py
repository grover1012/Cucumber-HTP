#!/usr/bin/env python3
"""Evaluate raw cucumber images using EnhancedTraitExtractor.

This script loads images from a directory (default: data/raw_images),
processes each image with EnhancedTraitExtractor, saves per-image
results (JSON metrics and visualization overlays) to a timestamped
subdirectory under models/evaluation/, and logs detection and
segmentation success rates for whole cucumbers and slices.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.enhanced_trait_extractor import EnhancedTraitExtractor  # type: ignore
from utils.image_utils import load_image  # type: ignore


def get_image_files(directory: Path):
    """Return list of image files in directory."""
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    files = []
    for ext in exts:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return files


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw images with EnhancedTraitExtractor")
    parser.add_argument('--model', required=True, help='Path to YOLO12 model (.pt file)')
    parser.add_argument('--image-dir', default='data/raw_images', help='Directory containing raw images')
    parser.add_argument('--output-root', default='models/evaluation', help='Root directory for evaluation outputs')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for detections')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    # Prepare timestamped evaluation directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = Path(args.output_root) / timestamp
    os.makedirs(eval_dir, exist_ok=True)

    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"No images found in directory: {image_dir}")
        return

    # Initialize extractor
    extractor = EnhancedTraitExtractor(args.model, args.confidence)

    per_image_log = []
    summary = {
        'total_images': len(image_files),
        'cucumber_detected': 0,
        'slice_detected': 0,
        'segmentation_success': 0,
    }

    for img_path in image_files:
        print(f"Processing {img_path}...")

        # Process image and save outputs
        try:
            results = extractor.process_image(str(img_path), str(eval_dir))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

        # Determine detections
        cucumber_detected = results['detections'].get('cucumber') is not None
        segmentation_success = bool(
            cucumber_detected and results['detections']['cucumber'].get('mask') is not None
        )

        # Detect slices separately (not returned by process_image)
        try:
            image = load_image(str(img_path))
            detections = extractor.detect_objects(image)
            slice_detected = any(d['class_name'] == 'slice' for d in detections)
        except Exception as e:
            print(f"Warning: could not run slice detection for {img_path}: {e}")
            slice_detected = False

        if cucumber_detected:
            summary['cucumber_detected'] += 1
        if slice_detected:
            summary['slice_detected'] += 1
        if segmentation_success:
            summary['segmentation_success'] += 1

        per_image_log.append({
            'image': str(img_path),
            'cucumber_detected': cucumber_detected,
            'slice_detected': slice_detected,
            'segmentation_success': segmentation_success,
        })

    total = summary['total_images']
    summary['cucumber_detection_rate'] = (
        summary['cucumber_detected'] / total if total else 0
    )
    summary['slice_detection_rate'] = (
        summary['slice_detected'] / total if total else 0
    )
    summary['segmentation_success_rate'] = (
        summary['segmentation_success'] / total if total else 0
    )

    # Save log and summary
    log_path = eval_dir / 'evaluation_log.json'
    with open(log_path, 'w') as f:
        json.dump({'per_image': per_image_log, 'summary': summary}, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {eval_dir}")


if __name__ == '__main__':
    main()
