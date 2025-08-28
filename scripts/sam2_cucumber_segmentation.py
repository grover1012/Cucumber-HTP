#!/usr/bin/env python3
"""
SAM2 Integration for Cucumber Segmentation
Combines YOLO detection with SAM2 segmentation
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO

# Try to import SAM2
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("✅ SAM2 available")
except ImportError:
    try:
        from segment_anything_hq import sam_model_registry, SamPredictor
        SAM_AVAILABLE = True
        print("✅ SAM-HQ available")
    except ImportError:
        SAM_AVAILABLE = False
        print("❌ SAM2 not available")

class SAM2CucumberSegmenter:
    def __init__(self, yolo_model_path, sam_checkpoint_path):
        """Initialize the segmenter."""
        self.yolo_model = YOLO(yolo_model_path)
        
        if SAM_AVAILABLE and sam_checkpoint_path:
            try:
                print(f"🔄 Loading SAM2 from {sam_checkpoint_path}")
                sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
                self.sam_predictor = SamPredictor(sam)
                print("✅ SAM2 loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load SAM2: {e}")
                self.sam_predictor = None
        else:
            self.sam_predictor = None
    
    def segment_cucumbers(self, image_path, output_dir, conf_threshold=0.3):
        """Segment cucumbers using YOLO + SAM2."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"🔍 Processing image: {Path(image_path).name}")
        
        # Run YOLO detection
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        detections = results[0]
        
        if not detections.boxes:
            print("❌ No detections found")
            return None
        
        # Process each detection
        segmentation_results = []
        
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            
            # Only process cucumbers
            if class_id == 4:  # cucumber class
                print(f"  Processing cucumber {i+1}: conf={confidence:.3f}")
                
                # Get SAM2 segmentation mask
                mask = self._get_sam2_mask(image_rgb, bbox)
                
                if mask is not None:
                    # Save mask
                    mask_filename = f"{Path(image_path).stem}_cucumber_{i+1}_mask.png"
                    mask_path = Path(output_dir) / mask_filename
                    cv2.imwrite(str(mask_path), mask)
                    
                    # Store results
                    segmentation_results.append({
                        'id': i,
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'mask_path': str(mask_path),
                        'mask_area': np.sum(mask > 0)
                    })
                    
                    print(f"    ✅ Mask saved: {mask_filename}")
                else:
                    print(f"    ⚠️ Failed to generate mask")
        
        return segmentation_results
    
    def _get_sam2_mask(self, image, bbox):
        """Get segmentation mask from SAM2."""
        if self.sam_predictor is None:
            return None
        
        try:
            # Set image in SAM2
            self.sam_predictor.set_image(image)
            
            # Convert bbox to SAM2 format
            x1, y1, x2, y2 = bbox
            sam_bbox = np.array([x1, y1, x2, y2])
            
            # Get masks from SAM2
            masks, scores, logits = self.sam_predictor.predict(
                box=sam_bbox,
                multimask_output=True
            )
            
            # Choose best mask by score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            
            # Convert to uint8
            mask = (best_mask * 255).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            print(f"⚠️ SAM2 failed for bbox {bbox}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='SAM2 Cucumber Segmentation')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--sam-checkpoint', required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--image-path', required=True, help='Path to image to segment')
    parser.add_argument('--output-dir', required=True, help='Output directory for masks')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize segmenter
    segmenter = SAM2CucumberSegmenter(args.yolo_model, args.sam_checkpoint)
    
    # Run segmentation
    results = segmenter.segment_cucumbers(args.image_path, args.output_dir, args.conf)
    
    if results:
        print(f"✅ Segmentation completed: {len(results)} cucumbers processed")

if __name__ == "__main__":
    main()
