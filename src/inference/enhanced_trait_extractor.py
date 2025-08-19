#!/usr/bin/env python3
"""
Enhanced Trait Extractor for Cucumber Analysis
Works with both original SAM and SAM2 (when available)
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


class EnhancedTraitExtractor:
    """
    Enhanced trait extractor with SAM segmentation capabilities.
    Automatically detects and uses the best available segmentation model.
    """
    
    def __init__(self, yolo_model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the enhanced trait extractor.
        
        Args:
            yolo_model_path: Path to trained YOLO12 model
            confidence_threshold: Minimum confidence for detections
        """
        self.yolo_model_path = yolo_model_path
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.sam_predictor = None
        self.sam_version = "none"
        
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

        # Normalized prompt templates for SAM2 segmentation
        # Coordinates are given relative to the bounding box (range 0-1)
        self.prompt_templates = {
            "cucumber": np.array([
                [0.5, 0.5],   # Center
                [0.2, 0.2],   # Top-left
                [0.8, 0.2],   # Top-right
                [0.2, 0.8],   # Bottom-left
                [0.8, 0.8],   # Bottom-right
            ])
        }
        
        # Load models
        self._load_yolo_model()
        self._load_segmentation_model()
    
    def _load_yolo_model(self):
        """Load the YOLO12 model."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"✅ YOLO12 model loaded successfully from {self.yolo_model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO12 model: {e}")
            self.yolo_model = None
    
    def _load_segmentation_model(self):
        """Load the best available segmentation model (SAM or SAM2)."""
        # Try SAM2 first (requires Python 3.10+)
        if self._try_load_sam2():
            self.sam_version = "sam2"
            return
        
        # Try original SAM
        if self._try_load_sam():
            self.sam_version = "sam"
            return
        
        # Fallback to no segmentation
        print("⚠️ No segmentation model available, using bounding box fallback")
        self.sam_version = "none"
    
    def _try_load_sam2(self):
        """Try to load SAM2 model."""
        try:
            # Check if SAM2 directory exists
            if os.path.exists("sam2"):
                print("🔍 SAM2 directory found, attempting to load...")
                
                # Add SAM2 to path
                sam2_path = os.path.join(os.getcwd(), "sam2")
                sys.path.insert(0, sam2_path)
                
                # Try to import SAM2
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                # Check if SAM2 model exists
                sam2_checkpoint = "sam2.1_hiera_tiny.pt"  # Smallest model for speed
                sam2_checkpoint_path = f"models/{sam2_checkpoint}"
                
                if not os.path.exists(sam2_checkpoint_path):
                    print(f"📥 Downloading SAM2 model to {sam2_checkpoint_path}...")
                    os.makedirs("models", exist_ok=True)
                    print("Please download SAM2 model manually from:")
                    print("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
                    print(f"and place it in: {sam2_checkpoint_path}")
                    return False
                
                # Load SAM2 model with configuration
                config_name = "configs/sam2.1/sam2.1_hiera_t.yaml"
                
                sam2_model = build_sam2(config_name, checkpoint=sam2_checkpoint_path, device="cpu")
                print("✅ Model already on CPU")
                self.sam_predictor = SAM2ImagePredictor(sam2_model)
                print(f"✅ SAM2 model loaded successfully from {sam2_checkpoint_path}")
                return True
                
        except Exception as e:
            print(f"⚠️ SAM2 loading failed: {e}")
            return False
        
        return False
    
    def _try_load_sam(self):
        """Try to load original SAM model."""
        try:
            import segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            
            # Download SAM model if not exists
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            sam_checkpoint_path = f"models/{sam_checkpoint}"
            
            if not os.path.exists(sam_checkpoint_path):
                print(f"📥 Downloading SAM model to {sam_checkpoint_path}...")
                os.makedirs("models", exist_ok=True)
                print("Please download SAM model manually from:")
                print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                print(f"and place it in: {sam_checkpoint_path}")
                return False
            
            # Load SAM model
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            sam.to(device="cpu")  # Use CPU for compatibility
            self.sam_predictor = SamPredictor(sam)
            print(f"✅ SAM model loaded successfully from {sam_checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ SAM loading failed: {e}")
            return False
    
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
                        'mask': None  # Will be filled by segmentation model
                    }
                    
                    detections.append(detection)
        
        return detections

    def _get_prompt_points(self, bbox: List[float], template_key: str = "cucumber") -> Optional[np.ndarray]:
        """Calculate absolute prompt points using a predefined template."""
        template = self.prompt_templates.get(template_key)
        if template is None:
            return None
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        points = template.copy()
        points[:, 0] = x1 + points[:, 0] * w
        points[:, 1] = y1 + points[:, 1] * h
        return points
    
    def generate_segmentation_mask(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Generate high-quality mask using available segmentation model.
        
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
            # Set image in predictor
            self.sam_predictor.set_image(image)
            
            # Convert bbox to SAM format [x, y, width, height]
            x1, y1, x2, y2 = bbox
            sam_bbox = np.array([x1, y1, x2, y2])
            
            # Generate mask
            if self.sam_version == "sam2":
                # Use predefined prompt template; fall back to center point
                points = self._get_prompt_points(bbox)
                if points is None or len(points) == 0:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    points = np.array([[center_x, center_y]])
                point_labels = np.ones(len(points), dtype=int)
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=False
                )
            else:
                # Original SAM prediction using box prompt
                masks, scores, logits = self.sam_predictor.predict(
                    box=sam_bbox,
                    multimask_output=False
                )
            
            # Return best mask
            best_mask = masks[0].astype(np.uint8) * 255
            return best_mask
            
        except Exception as e:
            print(f"⚠️ Segmentation mask generation failed: {e}, using bounding box fallback")
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
        Extract phenotypic traits from cucumber detection using segmentation.
        
        Args:
            image: Input image
            cucumber_detection: Cucumber detection result
            calibration: Calibration results
            
        Returns:
            Dictionary with extracted traits
        """
        if not cucumber_detection or 'bbox' not in cucumber_detection:
            return {}
        
        # Generate high-quality mask using segmentation model
        bbox = cucumber_detection['bbox']
        mask = self.generate_segmentation_mask(image, bbox)
        
        # Update detection with mask
        cucumber_detection['mask'] = mask
        
        pixel_to_mm_ratio = calibration.get('ruler', {}).get('pixel_to_mm_ratio', 1.0)
        
        # Extract traits using the high-quality mask
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
            'segmentation_model': self.sam_version,
            'mask_quality': 'high' if self.sam_predictor else 'bounding_box'
        })
        
        # Validate measurements
        validation = validate_measurements(traits)
        traits['validation'] = validation
        
        return traits
    
    def process_image(self, image_path: str, 
                     output_dir: str = None) -> Dict:
        """
        Process a single image and extract all traits using segmentation.
        
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
        
        # Extract cucumber traits with segmentation
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
            'segmentation_model': self.sam_version,
            'model_info': {
                'yolo_model': self.yolo_model_path,
                'segmentation_model': self.sam_version,
                'total_detections': len(detections)
            }
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
    """Test the enhanced trait extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced cucumber trait extraction with segmentation")
    parser.add_argument("--model", required=True, help="Path to YOLO12 model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="results/enhanced", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = EnhancedTraitExtractor(args.model)
    
    # Process image
    results = extractor.process_image(args.image, args.output_dir)
    
    # Print results
    print("\n=== Enhanced Trait Extraction Results ===")
    print(f"Image: {args.image}")
    print(f"Segmentation Model: {results['segmentation_model']}")
    
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
