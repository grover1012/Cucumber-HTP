#!/usr/bin/env python3
"""
SAM-Enhanced Trait Extractor for Cucumber Analysis
Combines YOLO12 detection with SAM segmentation for superior results
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from pathlib import Path
import torch

# Import utility modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image, save_image, create_visualization
from utils.trait_extraction import extract_all_traits, validate_measurements
from utils.calibration import (
    detect_ruler_markings, calibrate_with_known_object, detect_color_chart,
    normalize_colors, calculate_illumination_correction, apply_illumination_correction,
    create_calibration_report
)
from utils.ocr_utils import (
    extract_accession_id, validate_accession_id, correct_common_ocr_errors,
    create_ocr_report
)


class SAMTraitExtractor:
    """
    SAM-enhanced trait extractor combining YOLO12 detection with SAM segmentation.
    """
    
    def __init__(self, yolo_model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the SAM-enhanced trait extractor.
        
        Args:
            yolo_model_path: Path to trained YOLO12 model
            confidence_threshold: Minimum confidence for detections
        """
        self.yolo_model_path = yolo_model_path
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.sam_model = None
        
        # Updated to match your 12-class dataset
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        self.class_colors = [
            (255, 0, 0),    # Blue for big_ruler
            (0, 0, 255),    # Red for blue_dot
            (128, 0, 128),  # Purple for cavity
            (255, 255, 0),  # Cyan for color_chart
            (0, 255, 0),    # Green for cucumber
            (0, 255, 0),    # Green for green_dot
            (128, 128, 0),  # Olive for hollow
            (0, 0, 255),    # Red for label
            (255, 165, 0),  # Orange for objects
            (255, 0, 0),    # Blue for red_dot
            (0, 255, 255),  # Yellow for ruler
            (255, 192, 203) # Pink for slice
        ]
        
        # Load models
        self._load_yolo_model()
        self._load_sam_model()
    
    def _load_yolo_model(self):
        """Load the YOLO12 model."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"✅ YOLO12 model loaded successfully from {self.yolo_model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO12 model: {e}")
            self.yolo_model = None
    
    def _load_sam_model(self):
        """Load the SAM model."""
        try:
            import segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            
            # Download SAM model if not exists
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            sam_checkpoint_path = f"models/{sam_checkpoint}"
            
            if not os.path.exists(sam_checkpoint_path):
                print(f"📥 Downloading SAM model to {sam_checkpoint_path}...")
                os.makedirs("models", exist_ok=True)
                # You can download manually from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
                print("Please download SAM model manually from:")
                print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                print(f"and place it in: {sam_checkpoint_path}")
                return
            
            # Load SAM model
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            sam.to(device="cpu")  # Use CPU for compatibility
            self.sam_predictor = SamPredictor(sam)
            print(f"✅ SAM model loaded successfully from {sam_checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error loading SAM model: {e}")
            print("SAM will be disabled, using bounding box fallback")
            self.sam_predictor = None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image using YOLO12.
        
        Args:
            image: Input image
            
        Returns:
            List of detection dictionaries
        """
        if self.yolo_model is None:
            raise ValueError("YOLO12 model not loaded")
        
        # Run inference
        results = self.yolo_model(image, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Safely get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'mask': None  # Will be filled by SAM
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def generate_sam_mask(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Generate high-quality mask using SAM.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Binary segmentation mask
        """
        if self.sam_predictor is None:
            # Fallback to bounding box mask
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 255
            return mask
        
        try:
            # Set image in SAM predictor
            self.sam_predictor.set_image(image)
            
            # Convert bbox to SAM format [x, y, width, height]
            x1, y1, x2, y2 = bbox
            sam_bbox = np.array([x1, y1, x2, y2])
            
            # Generate mask
            masks, scores, logits = self.sam_predictor.predict(
                box=sam_bbox,
                multimask_output=False
            )
            
            # Return best mask
            best_mask = masks[0].astype(np.uint8) * 255
            return best_mask
            
        except Exception as e:
            print(f"⚠️ SAM mask generation failed: {e}, using bounding box fallback")
            # Fallback to bounding box mask
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = 255
            return mask
    
    def extract_cucumber_traits(self, image: np.ndarray,
                               cucumber_detection: Dict,
                               calibration: Dict) -> Dict:
        """
        Extract phenotypic traits from cucumber detection using SAM segmentation.
        
        Args:
            image: Input image
            cucumber_detection: Cucumber detection result
            calibration: Calibration results
            
        Returns:
            Dictionary with extracted traits
        """
        if not cucumber_detection or 'bbox' not in cucumber_detection:
            return {}
        
        # Generate high-quality mask using SAM
        bbox = cucumber_detection['bbox']
        mask = self.generate_sam_mask(image, bbox)
        
        # Update detection with SAM mask
        cucumber_detection['mask'] = mask
        
        pixel_to_mm_ratio = calibration.get('ruler', {}).get('pixel_to_mm_ratio', 1.0)
        
        # Extract traits using the high-quality SAM mask
        traits = extract_all_traits(mask, image, pixel_to_mm_ratio)
        
        # Add bounding box measurements as backup
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1
        length_px = max(width_px, height_px)
        width_px = min(width_px, height_px)
        
        # Convert to real units
        length_mm = length_px * pixel_to_mm_ratio
        width_mm = width_px * pixel_to_mm_ratio
        aspect_ratio = length_mm / width_mm if width_mm > 0 else 0
        
        # Update traits with measurements
        traits.update({
            'length_mm': length_mm,
            'width_mm': width_mm,
            'aspect_ratio': aspect_ratio,
            'length_px': length_px,
            'width_px': width_px,
            'bbox_area_px': width_px * height_px,
            'bbox_area_mm2': (width_px * height_px) * (pixel_to_mm_ratio ** 2),
            'sam_mask_quality': 'high' if self.sam_predictor else 'bounding_box'
        })
        
        # Validate measurements
        validation = validate_measurements(traits)
        traits['validation'] = validation
        
        return traits
    
    def process_image(self, image_path: str, 
                     output_dir: str = None) -> Dict:
        """
        Process a single image and extract all traits using SAM-enhanced segmentation.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            
        Returns:
            Dictionary with all extracted information
        """
        # Load image
        image = load_image(image_path)
        
        # Detect objects with YOLO12
        detections = self.detect_objects(image)
        
        # Separate detections by class
        cucumber_detections = [d for d in detections if d['class_name'] in ['cucumber', '4']]
        ruler_detections = [d for d in detections if d['class_name'] in ['ruler', '10', 'big_ruler', '0']]
        label_detections = [d for d in detections if d['class_name'] in ['label', '7']]
        color_chart_detections = [d for d in detections if d['class_name'] in ['color_chart', '3']]
        
        # Get best detection for each class (highest confidence)
        best_cucumber = max(cucumber_detections, key=lambda x: x['confidence']) if cucumber_detections else None
        best_ruler = max(ruler_detections, key=lambda x: x['confidence']) if ruler_detections else None
        best_label = max(label_detections, key=lambda x: x['confidence']) if label_detections else None
        best_color_chart = max(color_chart_detections, key=lambda x: x['confidence']) if color_chart_detections else None
        
        # Simple calibration (you can enhance this)
        calibration = {
            'ruler': {'pixel_to_mm_ratio': 1.0, 'confidence': 0.0},
            'color_chart': {'reference_colors': np.array([]), 'confidence': 0.0},
            'illumination': {'correction_factor': 1.0}
        }
        
        # Extract cucumber traits with SAM
        cucumber_traits = {}
        if best_cucumber:
            cucumber_traits = self.extract_cucumber_traits(image, best_cucumber, calibration)
        
        # Create results dictionary
        results = {
            'image_path': image_path,
            'detections': {
                'cucumber': best_cucumber,
                'ruler': best_ruler,
                'label': best_label,
                'color_chart': best_color_chart
            },
            'cucumber_traits': cucumber_traits,
            'calibration': calibration,
            'sam_enabled': self.sam_predictor is not None
        }
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results as JSON
            results_file = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save visualization
            if best_cucumber and 'mask' in best_cucumber:
                vis_image = self.create_visualization(image, detections, best_cucumber['mask'])
                vis_file = os.path.join(output_dir, f"{Path(image_path).stem}_visualization.jpg")
                cv2.imwrite(vis_file, vis_image)
        
        return results
    
    def create_visualization(self, image: np.ndarray, detections: List[Dict], 
                           cucumber_mask: np.ndarray) -> np.ndarray:
        """
        Create visualization of detections and segmentation.
        
        Args:
            image: Input image
            detections: List of detections
            cucumber_mask: Cucumber segmentation mask
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.class_colors[detection['class_id']] if detection['class_id'] < len(self.class_colors) else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay cucumber mask
        if cucumber_mask is not None:
            mask_colored = np.zeros_like(image)
            mask_colored[cucumber_mask > 0] = [0, 255, 0]  # Green for cucumber
            
            # Blend mask with image
            alpha = 0.3
            vis_image = cv2.addWeighted(vis_image, 1-alpha, mask_colored, alpha, 0)
        
        return vis_image


def main():
    """Test the SAM-enhanced trait extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM-enhanced cucumber trait extraction")
    parser.add_argument("--model", required=True, help="Path to YOLO12 model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SAMTraitExtractor(args.model)
    
    # Process image
    results = extractor.process_image(args.image, args.output_dir)
    
    # Print results
    print("\n=== SAM-Enhanced Trait Extraction Results ===")
    print(f"Image: {args.image}")
    print(f"SAM Enabled: {results['sam_enabled']}")
    
    if results['cucumber_traits']:
        print("\n🥒 Cucumber Traits:")
        for key, value in results['cucumber_traits'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
