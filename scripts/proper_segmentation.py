#!/usr/bin/env python3
"""
Proper Segmentation with SAM2
Uses YOLO for detection and SAM2 for accurate segmentation masks
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

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
        print("❌ SAM2 not available - will use fallback")

class ProperSegmentation:
    def __init__(self, yolo_model_path, sam_checkpoint_path=None):
        """Initialize with YOLO model and optional SAM2 checkpoint."""
        self.yolo_model = YOLO(yolo_model_path)
        
        # Class names and colors
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        self.colors = {
            'cucumber': (0, 255, 0),      # Green
            'slice': (255, 165, 0),        # Orange
            'ruler': (255, 0, 0),         # Red
            'color_chart': (0, 0, 255),   # Blue
            'big_ruler': (255, 0, 255),   # Magenta
            'cavity': (0, 255, 255),      # Cyan
            'hollow': (128, 128, 128),    # Gray
            'label': (255, 255, 0),       # Yellow
            'objects': (128, 0, 128),     # Purple
            'blue_dot': (255, 0, 128),    # Pink
            'green_dot': (0, 128, 0),     # Dark Green
            'red_dot': (128, 0, 0)        # Dark Red
        }
        
        # Initialize SAM2 if available
        self.sam_predictor = None
        if SAM_AVAILABLE and sam_checkpoint_path:
            try:
                print(f"🔄 Loading SAM2 from {sam_checkpoint_path}")
                sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
                self.sam_predictor = SamPredictor(sam)
                print("✅ SAM2 loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load SAM2: {e}")
                self.sam_predictor = None
    
    def create_fallback_mask(self, image_shape, bbox, expand_ratio=0.05):
        """Create a better fallback mask using morphological operations."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Expand bbox slightly
        bw, bh = x2 - x1, y2 - y1
        x1e = max(0, int(x1 - expand_ratio * bw))
        y1e = max(0, int(y1 - expand_ratio * bh))
        x2e = min(w - 1, int(x2 + expand_ratio * bw))
        y2e = min(h - 1, int(y2 + expand_ratio * bh))
        
        # Create base mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1e:y2e, x1e:x2e] = 255
        
        # Apply morphological operations for smoother edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def get_sam_mask(self, image, bbox):
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
    
    def segment_image(self, image_path, output_dir, conf_threshold=0.3):
        """Segment objects in the image using YOLO + SAM2."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        print(f"🔍 Processing image: {Path(image_path).name}")
        print(f"📏 Image dimensions: {w} x {h} pixels")
        
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
            class_name = self.class_names[class_id]
            
            print(f"  Processing {class_name} (ID {i}): conf={confidence:.3f}")
            
            # Get segmentation mask
            mask = self.get_sam_mask(image_rgb, bbox)
            
            if mask is None:
                print(f"    ⚠️ Using fallback mask for {class_name}")
                mask = self.create_fallback_mask(image.shape, bbox)
            
            # Save individual mask
            mask_filename = f"{Path(image_path).stem}_obj{i:02d}_{class_name}_mask.png"
            mask_path = Path(output_dir) / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            # Store results
            segmentation_results.append({
                'id': i,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox.tolist(),
                'mask_path': str(mask_path),
                'mask_area': np.sum(mask > 0),
                'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            })
            
            print(f"    ✅ Mask saved: {mask_filename}")
        
        # Create visualization
        self._create_segmentation_visualization(image_rgb, segmentation_results, image_path, output_dir)
        
        # Save results JSON
        results_path = Path(output_dir) / f"{Path(image_path).stem}_segmentation_results.json"
        with open(results_path, 'w') as f:
            json.dump(segmentation_results, f, indent=2)
        
        print(f"✅ Segmentation complete! Results saved to {output_dir}")
        return segmentation_results
    
    def _create_segmentation_visualization(self, image_rgb, results, image_path, output_dir):
        """Create visualization showing original image, detections, and masks."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Segmentation Results: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0,0].imshow(image_rgb)
        axes[0,0].set_title('1. Original Image', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        # Plot 2: Detections with bounding boxes
        axes[0,1].imshow(image_rgb)
        axes[0,1].set_title('2. YOLO Detections', fontsize=14, fontweight='bold')
        axes[0,1].axis('off')
        
        for result in results:
            bbox = result['bbox']
            class_name = result['class_name']
            confidence = result['confidence']
            color = self.colors.get(class_name, (255, 255, 255))
            color_norm = tuple(c/255 for c in color)
            
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color_norm, facecolor='none')
            axes[0,1].add_patch(rect)
            
            label = f"{class_name}\n{confidence:.3f}"
            axes[0,1].text(x1, y1-5, label, fontsize=9, color=color_norm,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 3: Segmentation masks overlay
        axes[1,0].imshow(image_rgb)
        axes[1,0].set_title('3. Segmentation Masks Overlay', fontsize=14, fontweight='bold')
        axes[1,0].axis('off')
        
        # Create combined mask overlay
        combined_mask = np.zeros_like(image_rgb)
        for result in results:
            mask_path = result['mask_path']
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                class_name = result['class_name']
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Apply mask with color
                mask_bool = mask > 0
                combined_mask[mask_bool] = color
        
        # Blend with original image
        alpha = 0.6
        blended = cv2.addWeighted(image_rgb, 1-alpha, combined_mask, alpha, 0)
        axes[1,0].imshow(blended)
        
        # Plot 4: Individual masks grid
        axes[1,1].set_title('4. Individual Masks', fontsize=14, fontweight='bold')
        axes[1,1].axis('off')
        
        # Show first few masks in a grid
        n_masks = min(6, len(results))
        if n_masks > 0:
            # Create a grid layout
            cols = 3
            rows = (n_masks + cols - 1) // cols
            
            for i in range(n_masks):
                result = results[i]
                mask_path = result['mask_path']
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    # Create subplot for this mask
                    ax = plt.subplot2grid((rows, cols), (i // cols, i % cols), fig=fig)
                    ax.imshow(mask, cmap='gray')
                    ax.set_title(f"{result['class_name']} {i}", fontsize=10)
                    ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = Path(output_dir) / f"{Path(image_path).stem}_segmentation_visualization.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Proper Segmentation with YOLO + SAM2')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--image-path', required=True, help='Path to image to segment')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--sam-checkpoint', help='Path to SAM2 checkpoint (optional)')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize segmenter
    segmenter = ProperSegmentation(args.yolo_model, args.sam_checkpoint)
    
    # Run segmentation
    results = segmenter.segment_image(args.image_path, args.output_dir, args.conf)

if __name__ == "__main__":
    main()
