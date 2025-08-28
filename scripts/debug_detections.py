#!/usr/bin/env python3
"""
Debug Detections
Show exactly what's being detected and what's wrong
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DetectionDebugger:
    def __init__(self, yolo_model_path):
        """Initialize the detection debugger."""
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
    
    def debug_image(self, image_path, output_dir):
        """Debug the image with detailed detection analysis."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"🔍 Debugging image: {Path(image_path).name}")
        print(f"📏 Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
        
        # Run YOLO detection with very low confidence
        print(f"\n🔍 Running YOLO with conf=0.01 to see ALL detections...")
        results = self.yolo_model(image, conf=0.01, verbose=False)
        detections = results[0]
        
        if not detections.boxes:
            print("❌ No detections found!")
            return
        
        # Analyze each detection in detail
        print(f"\n📊 DETAILED DETECTION ANALYSIS:")
        print("=" * 80)
        
        detection_details = []
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = height / width if width > 0 else 0
            
            detection_details.append({
                'id': i,
                'bbox': bbox,
                'class_name': class_name,
                'confidence': confidence,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'width': width,
                'height': height,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2
            })
            
            print(f"Detection {i:2d}: {class_name:12s} | Conf: {confidence:6.3f} | "
                  f"Area: {area:8.0f} | AR: {aspect_ratio:5.2f} | "
                  f"Pos: ({x1:5.0f}, {y1:5.0f}) to ({x2:5.0f}, {y2:5.0f})")
        
        # Group detections by class
        print(f"\n📈 DETECTION SUMMARY BY CLASS:")
        print("=" * 80)
        
        class_groups = {}
        for det in detection_details:
            class_name = det['class_name']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        for class_name, dets in sorted(class_groups.items()):
            print(f"\n{class_name.upper()}: {len(dets)} detections")
            print("-" * 40)
            
            # Sort by confidence
            dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            for det in dets:
                print(f"  ID {det['id']:2d}: Conf {det['confidence']:6.3f} | "
                      f"Area {det['area']:8.0f} | AR {det['aspect_ratio']:5.2f}")
        
        # Analyze cucumber detections specifically
        cucumber_dets = [d for d in detection_details if d['class_name'] == 'cucumber']
        if cucumber_dets:
            print(f"\n🥒 CUCUMBER DETECTION ANALYSIS:")
            print("=" * 80)
            print(f"Total cucumber detections: {len(cucumber_dets)}")
            
            # Check for overlapping cucumbers
            overlapping_groups = self._find_overlapping_groups(cucumber_dets, iou_threshold=0.3)
            
            print(f"Overlapping cucumber groups: {len(overlapping_groups)}")
            for i, group in enumerate(overlapping_groups):
                print(f"  Group {i+1}: {len(group)} overlapping detections")
                for det in group:
                    print(f"    ID {det['id']}: Conf {det['confidence']:.3f}, Area {det['area']:.0f}")
        
        # Create detailed visualization
        self._create_debug_visualization(image, detection_details, image_path, output_dir)
        
        return detection_details
    
    def _find_overlapping_groups(self, detections, iou_threshold=0.3):
        """Find groups of overlapping detections."""
        if len(detections) <= 1:
            return []
        
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            current_group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    current_group.append(det2)
                    used.add(j)
            
            if len(current_group) > 1:
                groups.append(current_group)
        
        return groups
    
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
    
    def _create_debug_visualization(self, image, detection_details, image_path, output_dir):
        """Create detailed debug visualization."""
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Detection Debug: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0,0].imshow(image_rgb)
        axes[0,0].set_title('1. Original Image', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        # Plot 2: All detections with confidence
        axes[0,1].imshow(image_rgb)
        axes[0,1].set_title('2. All Detections (with confidence)', fontsize=14, fontweight='bold')
        axes[0,1].axis('off')
        
        # Draw all detection boxes with confidence
        for det in detection_details:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_name, (255, 255, 255))
            color_normalized = tuple(c/255 for c in color)
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color_normalized, facecolor='none')
            axes[0,1].add_patch(rect)
            
            # Add label with confidence
            label = f"{class_name}\n{confidence:.3f}"
            axes[0,1].text(x1, y1-5, label, fontsize=8, color=color_normalized,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 3: Only cucumber detections
        axes[1,0].imshow(image_rgb)
        axes[1,0].set_title('3. Cucumber Detections Only', fontsize=14, fontweight='bold')
        axes[1,0].axis('off')
        
        cucumber_dets = [d for d in detection_details if d['class_name'] == 'cucumber']
        for det in cucumber_dets:
            bbox = det['bbox']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0)  # Green for cucumbers
            color_normalized = (0, 1, 0)
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=color_normalized, facecolor='none')
            axes[1,0].add_patch(rect)
            
            # Add label with ID and confidence
            label = f"Cucumber {det['id']}\n{confidence:.3f}"
            axes[1,0].text(x1, y1-5, label, fontsize=10, color=color_normalized,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 4: Problematic detections (rulers, cavity, etc.)
        axes[1,1].imshow(image_rgb)
        axes[1,1].set_title('4. Problematic Detections (rulers, cavity, etc.)', fontsize=14, fontweight='bold')
        axes[1,1].axis('off')
        
        problematic_classes = ['big_ruler', 'cavity', 'objects', 'hollow']
        for det in detection_details:
            if det['class_name'] in problematic_classes:
                bbox = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                x1, y1, x2, y2 = bbox
                color = self.colors.get(class_name, (255, 255, 255))
                color_normalized = tuple(c/255 for c in color)
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color_normalized, facecolor='none')
                axes[1,1].add_patch(rect)
                
                # Add label with ID and confidence
                label = f"{class_name} {det['id']}\n{confidence:.3f}"
                axes[1,1].text(x1, y1-5, label, fontsize=10, color=color_normalized,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the debug visualization
        output_path = Path(output_dir) / f"{Path(image_path).stem}_debug.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Debug visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Debug Detections')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--image-path', required=True, help='Path to image to debug')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize debugger
    debugger = DetectionDebugger(args.yolo_model)
    
    # Debug the image
    detection_details = debugger.debug_image(args.image_path, args.output_dir)

if __name__ == "__main__":
    main()
