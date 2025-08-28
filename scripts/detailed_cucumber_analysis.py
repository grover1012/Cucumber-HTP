#!/usr/bin/env python3
"""
Detailed Cucumber Analysis
Identify all potential cucumbers and analyze why some might be missed
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DetailedCucumberAnalyzer:
    def __init__(self, yolo_model_path):
        """Initialize the detailed analyzer."""
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
            'objects': (255, 255, 0),      # Yellow (potential cucumbers)
            'cavity': (0, 255, 255),      # Cyan
            'hollow': (128, 128, 128),    # Gray
            'other': (255, 0, 255)        # Magenta
        }
    
    def analyze_with_very_low_confidence(self, image_path, confidence_threshold=0.01):
        """Analyze with very low confidence to see all potential detections."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"🔍 Analyzing image: {Path(image_path).name}")
        print(f"📏 Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
        print(f"🎯 Using very low confidence threshold: {confidence_threshold}")
        
        # Run YOLO detection with very low confidence
        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
        detections = results[0]
        
        # Categorize detections
        cucumber_detections = []
        slice_detections = []
        potential_cucumbers = []
        other_detections = []
        
        print(f"\n📊 ALL DETECTIONS (confidence ≥ {confidence_threshold}):")
        print("=" * 60)
        
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Calculate detection area
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            area_percentage = (area / (image.shape[0] * image.shape[1])) * 100
            
            # Calculate aspect ratio (length/width)
            aspect_ratio = height / width if width > 0 else 0
            
            detection_info = {
                'id': i,
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'width': width,
                'height': height,
                'area': area,
                'area_percentage': area_percentage,
                'aspect_ratio': aspect_ratio
            }
            
            # Categorize based on class and characteristics
            if class_name == 'cucumber':
                cucumber_detections.append(detection_info)
                category = "CUCUMBER"
            elif class_name == 'slice':
                slice_detections.append(detection_info)
                category = "SLICE"
            elif class_name in ['objects', 'cavity', 'hollow'] and aspect_ratio > 2.0:
                # These might be misclassified cucumbers
                potential_cucumbers.append(detection_info)
                category = "POTENTIAL CUCUMBER"
            else:
                other_detections.append(detection_info)
                category = "OTHER"
            
            print(f"  Detection {i+1}: {class_name} (conf: {confidence:.3f}) - {category}")
            print(f"    BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print(f"    Size: {width:.1f} x {height:.1f} pixels")
            print(f"    Area: {area:.0f} pixels ({area_percentage:.2f}% of image)")
            print(f"    Aspect Ratio: {aspect_ratio:.2f}")
            print()
        
        # Summary
        print(f"📈 DETECTION SUMMARY:")
        print(f"  Total detections: {len(detections.boxes)}")
        print(f"  Cucumber detections: {len(cucumber_detections)}")
        print(f"  Slice detections: {len(slice_detections)}")
        print(f"  Potential cucumbers: {len(potential_cucumbers)}")
        print(f"  Other detections: {len(other_detections)}")
        
        # Analyze potential cucumbers
        if potential_cucumbers:
            print(f"\n🔍 POTENTIAL CUCUMBER ANALYSIS:")
            print("=" * 40)
            for det in potential_cucumbers:
                print(f"  {det['class']} (conf: {det['confidence']:.3f})")
                print(f"    Aspect ratio: {det['aspect_ratio']:.2f} (typical cucumber: 3-8)")
                print(f"    Area: {det['area_percentage']:.2f}% of image")
                print(f"    This might be a misclassified cucumber!")
                print()
        
        # Create comprehensive visualization
        self.create_detailed_visualization(image, detections, image_path)
        
        return {
            'cucumbers': cucumber_detections,
            'slices': slice_detections,
            'potential_cucumbers': potential_cucumbers,
            'others': other_detections
        }
    
    def create_detailed_visualization(self, image, detections, image_path):
        """Create detailed visualization showing all detections."""
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(f'Detailed Cucumber Analysis: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot 2: All detections with class labels
        axes[0, 1].imshow(image_rgb)
        axes[0, 1].set_title('2. All Detections (Classified)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Plot 3: Only cucumber/slice detections
        axes[0, 2].imshow(image_rgb)
        axes[0, 2].set_title('3. Cucumber/Slice Only', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Plot 4: Potential cucumbers (misclassified)
        axes[1, 0].imshow(image_rgb)
        axes[1, 0].set_title('4. Potential Cucumbers (Misclassified)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Plot 5: Detection confidence heatmap
        axes[1, 1].imshow(image_rgb)
        axes[1, 1].set_title('5. Confidence Levels', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Plot 6: Detection statistics
        axes[1, 2].axis('off')
        
        # Draw all detection boxes with appropriate colors
        for detection in detections.boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            x1, y1, x2, y2 = bbox
            
            # Determine color and category
            if class_name == 'cucumber':
                color = self.colors['cucumber']
                category = 'cucumber'
            elif class_name == 'slice':
                color = self.colors['slice']
                category = 'slice'
            elif class_name in ['objects', 'cavity', 'hollow']:
                color = self.colors['objects']
                category = 'potential'
            else:
                color = self.colors['other']
                category = 'other'
            
            color_normalized = tuple(c/255 for c in color)
            
            # Plot 2: All detections
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color_normalized, facecolor='none')
            axes[0, 1].add_patch(rect)
            axes[0, 1].text(x1, y1-5, f"{class_name}\n{confidence:.3f}", 
                           fontsize=8, color=color_normalized,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot 3: Only cucumbers/slices
            if category in ['cucumber', 'slice']:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color_normalized, facecolor='none')
                axes[0, 2].add_patch(rect)
                axes[0, 2].text(x1, y1-5, f"{class_name}\n{confidence:.3f}", 
                               fontsize=10, color=color_normalized,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot 4: Potential cucumbers
            if category == 'potential':
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color_normalized, facecolor='none')
                axes[1, 0].add_patch(rect)
                axes[1, 0].text(x1, y1-5, f"{class_name}\n{confidence:.3f}", 
                               fontsize=10, color=color_normalized,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot 5: Confidence levels
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[1, 1].add_patch(rect)
            
            # Color code by confidence
            if confidence > 0.8:
                color_code = 'green'
            elif confidence > 0.5:
                color_code = 'orange'
            else:
                color_code = 'red'
            
            axes[1, 1].text(x1, y1-5, f"{confidence:.3f}", 
                           fontsize=8, color=color_code, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 6: Detection summary table
        if len(detections.boxes) > 0:
            # Create detection summary table
            table_data = []
            for detection in detections.boxes:
                bbox = detection.xyxy[0].cpu().numpy()
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                class_name = self.class_names[class_id]
                
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 0
                
                table_data.append([
                    class_name,
                    f"{confidence:.3f}",
                    f"{width:.0f} x {height:.0f}",
                    f"{aspect_ratio:.2f}"
                ])
            
            table = axes[1, 2].table(
                cellText=table_data,
                colLabels=['Class', 'Confidence', 'Size (px)', 'Aspect Ratio'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(4):
                    if i == 0:  # Header row
                        table[(i, j)].set_facecolor('#4CAF50')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    else:
                        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            axes[1, 2].set_title('6. Detection Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the detailed visualization
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_detailed_analysis.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Detailed analysis visualization saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Detailed Cucumber Analysis')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--image-path', required=True, help='Path to image to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DetailedCucumberAnalyzer(args.yolo_model)
    
    # Run detailed analysis
    results = analyzer.analyze_with_very_low_confidence(args.image_path)

if __name__ == "__main__":
    main()
