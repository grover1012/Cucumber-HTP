#!/usr/bin/env python3
"""Generate SAM2 masks for cucumber images.

This utility scans a directory of images, detects cucumbers using the
YOLO12 model and generates segmentation masks using the SAM2 prompt
templates defined in ``EnhancedTraitExtractor``.

Masks are saved to the specified output directory (defaults to
``data/sam2_masks``).
"""

import argparse
import os
from pathlib import Path

import cv2

from src.inference.enhanced_trait_extractor import EnhancedTraitExtractor


def process_directory(model_path: str, image_dir: str, output_dir: str) -> None:
    extractor = EnhancedTraitExtractor(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        detections = extractor.detect_objects(image)
        cucumbers = [d for d in detections if d.get("class_name") == "cucumber"]
        for idx, det in enumerate(cucumbers):
            mask = extractor.generate_segmentation_mask(image, det["bbox"])
            out_path = os.path.join(
                output_dir, f"{Path(img_name).stem}_{idx}.png"
            )
            cv2.imwrite(out_path, mask)
            print(f"Saved mask to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SAM2 masks for cucumbers")
    parser.add_argument(
        "--model", required=True, help="Path to trained YOLO12 model weights"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir",
        default="data/sam2_masks",
        help="Directory to store generated masks",
    )
    args = parser.parse_args()

    process_directory(args.model, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()

