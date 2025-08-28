#!/usr/bin/env python3
"""
Fix Detection Issues
Clean up over-detection and misclassification problems
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DetectionFixer:
    def __init__(self, yolo_model_path):
        """Initialize the detection fixer."""
        self.yolo_model = YOLO(yolo_model_path)
        
        # Class names from your YOLO model
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Color scheme
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
    
    def clean_detections(self, detections, image_shape, iou_threshold=0.3):
        """Clean up detections by removing duplicates and fixing misclassifications."""
        if not detections.boxes:
            return []
        
        # Convert detections to list for processing
        detection_list = []
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            detection_list.append({
                'id': i,
                'bbox': bbox,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'aspect_ratio': (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) if (bbox[2] - bbox[0]) > 0 else 0
            })
        
        # Step 1: Fix obvious misclassifications
        fixed_detections = self._fix_misclassifications(detection_list)
        
        # Step 2: Merge overlapping cucumber detections
        merged_detections = self._merge_overlapping_cucumbers(fixed_detections, iou_threshold)
        
        # Step 3: Remove low-confidence duplicates
        final_detections = self._remove_duplicates(merged_detections, iou_threshold)
        
        return final_detections
    
    def _fix_misclassifications(self, detections):
        """Fix obvious misclassifications based on object characteristics."""
        fixed = []
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            aspect_ratio = det['aspect_ratio']
            area = det['area']
            
            # Fix big_ruler misclassifications over cucumbers
            if class_name == 'big_ruler' and confidence < 0.1:
                # Check if this looks more like a cucumber
                if aspect_ratio > 1.5 and area > 100000:  # Large, elongated object
                    det['class_name'] = 'cucumber'
                    det['confidence'] = max(confidence * 2, 0.3)  # Boost confidence
                    print(f"  🔧 Fixed: big_ruler → cucumber (conf: {confidence:.3f} → {det['confidence']:.3f})")
            
            # Fix cavity misclassifications that look like cucumbers
            elif class_name == 'cavity' and aspect_ratio > 2.0 and area > 50000:
                det['class_name'] = 'cucumber'
                det['confidence'] = max(confidence * 3, 0.4)
                print(f"  🔧 Fixed: cavity → cucumber (conf: {confidence:.3f} → {det['confidence']:.3f})")
            
            # Fix objects misclassifications that look like cucumbers
            elif class_name == 'objects' and aspect_ratio > 2.5 and area > 80000:
                det['class_name'] = 'cucumber'
                det['confidence'] = max(confidence * 2, 0.35)
                print(f"  🔧 Fixed: objects → cucumber (conf: {confidence:.3f} → {det['confidence']:.3f})")
            
            fixed.append(det)
        
        return fixed
    
    def _merge_overlapping_cucumbers(self, detections, iou_threshold):
        """Merge overlapping cucumber detections into single objects."""
        cucumber_dets = [d for d in detections if d['class_name'] in ['cucumber', 'slice']]
        other_dets = [d for d in detections if d['class_name'] not in ['cucumber', 'slice']]
        
        if len(cucumber_dets) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        cucumber_dets.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged_cucumbers = []
        used_indices = set()
        
        for i, det1 in enumerate(cucumber_dets):
            if i in used_indices:
                continue
            
            current_group = [det1]
            used_indices.add(i)
            
            # Find overlapping cucumbers
            for j, det2 in enumerate(cucumber_dets[i+1:], i+1):
                if j in used_indices:
                    continue
                
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    current_group.append(det2)
                    used_indices.add(j)
                    print(f"  🔗 Merging overlapping cucumbers: {det1['confidence']:.3f} + {det2['confidence']:.3f}")
            
            # Merge the group
            if len(current_group) > 1:
                merged = self._merge_detection_group(current_group)
                merged_cucumbers.append(merged)
            else:
                merged_cucumbers.append(det1)
        
        return merged_cucumbers + other_dets
    
    def _merge_detection_group(self, group):
        """Merge a group of overlapping detections."""
        # Use the highest confidence detection as base
        base = max(group, key=lambda x: x['confidence'])
        
        # Calculate merged bounding box
        x1 = min(d['bbox'][0] for d in group)
        y1 = min(d['bbox'][1] for d in group)
        x2 = max(d['bbox'][2] for d in group)
        y2 = max(d['bbox'][3] for d in group)
        
        # Calculate merged confidence (weighted average)
        total_weight = sum(d['confidence'] for d in group)
        merged_confidence = sum(d['confidence'] * d['confidence'] for d in group) / total_weight
        
        merged = base.copy()
        merged['bbox'] = np.array([x1, y1, x2, y2])
        merged['confidence'] = merged_confidence
        merged['area'] = (x2 - x1) * (y2 - y1)
        merged['aspect_ratio'] = (y2 - y1) / (x2 - x1) if (x2 - x1) > 0 else 0
        
        return merged
    
    def _remove_duplicates(self, detections, iou_threshold):
        """Remove remaining duplicate detections."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        for det in detections:
            is_duplicate = False
            
            for existing in final_detections:
                iou = self._calculate_iou(det['bbox'], existing['bbox'])
                if iou > iou_threshold and det['class_name'] == existing['class_name']:
                    is_duplicate = True
                    print(f"  🗑️ Removing duplicate: {det['class_name']} (conf: {det['confidence']:.3f})")
                    break
            
            if not is_duplicate:
                final_detections.append(det)
        
        return final_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_and_fix_image(self, image_path, output_dir):
        """Analyze image, fix detections, and create comparison visualization."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"🔍 Analyzing and fixing image: {Path(image_path).name}")
        print(f"📏 Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
        
        # Run YOLO detection with low confidence to see all problematic detections
        results = self.yolo_model(image, conf=0.01, verbose=False)
        original_detections = results[0]
        
        print(f"\n📊 Original Detection Results:")
        print("=" * 50)
        self._print_detection_summary(original_detections)
        
        # Clean up detections
        print(f"\n🔧 Fixing Detection Issues...")
        print("=" * 50)
        
        fixed_detections = self.clean_detections(original_detections, image.shape)
        
        print(f"\n📊 Fixed Detection Results:")
        print("=" * 50)
        self._print_detection_summary_fixed(fixed_detections)
        
        # Create comparison visualization
        self._create_comparison_visualization(image, original_detections, fixed_detections, image_path, output_dir)
        
        return fixed_detections
    
    def _print_detection_summary(self, detections):
        """Print summary of original detections."""
        if not detections.boxes:
            print("  No detections found")
            return
        
        class_counts = {}
        for detection in detections.boxes:
            class_id = int(detection.cls[0])
            class_name = self.class_names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
    def _print_detection_summary_fixed(self, detections):
        """Print summary of fixed detections."""
        if not detections:
            print("  No detections found")
            return
        
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
    def _create_comparison_visualization(self, image, original_detections, fixed_detections, image_path, output_dir):
        """Create visualization comparing original vs fixed detections."""
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with comparison
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Detection Fix Comparison: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('1. Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Plot 2: Original detections (problematic)
        axes[1].imshow(image_rgb)
        axes[1].set_title('2. Original Detections (Problems)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Draw original detection boxes
        if original_detections.boxes:
            for detection in original_detections.boxes:
                bbox = detection.xyxy[0].cpu().numpy()
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                class_name = self.class_names[class_id]
                
                x1, y1, x2, y2 = bbox
                color = self.colors.get(class_name, (255, 255, 255))
                color_normalized = tuple(c/255 for c in color)
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color_normalized, facecolor='none')
                axes[1].add_patch(rect)
                
                label = f"{class_name}\n{confidence:.3f}"
                axes[1].text(x1, y1-5, label, fontsize=8, color=color_normalized,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 3: Fixed detections
        axes[2].imshow(image_rgb)
        axes[2].set_title('3. Fixed Detections (Clean)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Draw fixed detection boxes
        if fixed_detections:
            for det in fixed_detections:
                bbox = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                x1, y1, x2, y2 = bbox
                color = self.colors.get(class_name, (255, 255, 255))
                color_normalized = tuple(c/255 for c in color)
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color_normalized, facecolor='none')
                axes[2].add_patch(rect)
                
                label = f"{class_name}\n{confidence:.3f}"
                axes[2].text(x1, y1-5, label, fontsize=10, color=color_normalized,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comparison visualization
        output_path = Path(output_dir) / f"{Path(image_path).stem}_fixed_comparison.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comparison visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fix Detection Issues')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--image-path', required=True, help='Path to image to fix')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = DetectionFixer(args.yolo_model)
    
    # Analyze and fix the image
    fixed_detections = fixer.analyze_and_fix_image(args.image_path, args.output_dir)

if __name__ == "__main__":
    main()
