#!/usr/bin/env python3
"""
Diagnose Detection Issues
Analyze why objects are being missed or incorrectly detected
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DetectionDiagnostic:
    def __init__(self, yolo_model_path):
        """Initialize the diagnostic tool."""
        self.yolo_model = YOLO(yolo_model_path)
        
        # Class names from your YOLO model
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Color scheme for visualization
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
    
    def analyze_image(self, image_path, confidence_threshold=0.1):
        """Analyze detection performance on a specific image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"🔍 Analyzing image: {Path(image_path).name}")
        print(f"📏 Image dimensions: {image.shape[1]} x {image.shape[0]} pixels")
        
        # Run YOLO detection with different confidence thresholds
        print(f"\n🎯 Running YOLO detection with confidence threshold: {confidence_threshold}")
        
        results = self.yolo_model(image, verbose=False)
        detections = results[0]
        
        # Analyze all detections
        all_detections = []
        cucumber_detections = []
        other_detections = []
        
        print(f"\n📊 Detection Results:")
        print("=" * 50)
        
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
            
            detection_info = {
                'id': i,
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'width': width,
                'height': height,
                'area': area,
                'area_percentage': area_percentage
            }
            
            all_detections.append(detection_info)
            
            if class_name in ['cucumber', 'slice']:
                cucumber_detections.append(detection_info)
            else:
                other_detections.append(detection_info)
            
            print(f"  Detection {i+1}: {class_name} (conf: {confidence:.3f})")
            print(f"    BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print(f"    Size: {width:.1f} x {height:.1f} pixels")
            print(f"    Area: {area:.0f} pixels ({area_percentage:.2f}% of image)")
            print()
        
        print(f"📈 Summary:")
        print(f"  Total detections: {len(all_detections)}")
        print(f"  Cucumber/slice detections: {len(cucumber_detections)}")
        print(f"  Other object detections: {len(other_detections)}")
        
        # Create comprehensive visualization
        self.create_diagnostic_visualization(image, all_detections, image_path)
        
        return all_detections
    
    def create_diagnostic_visualization(self, image, detections, image_path):
        """Create detailed diagnostic visualization."""
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'Detection Diagnostic: {Path(image_path).name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot 2: All detections with confidence scores
        axes[0, 1].imshow(image_rgb)
        axes[0, 1].set_title('2. All Detections (with confidence)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Draw all detection boxes
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color for this class
            color = self.colors.get(class_name, (255, 255, 255))
            color_normalized = tuple(c/255 for c in color)
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color_normalized, facecolor='none')
            axes[0, 1].add_patch(rect)
            
            # Add label with confidence
            label = f"{class_name}\n{confidence:.3f}"
            axes[0, 1].text(x1, y1-5, label, fontsize=8, color=color_normalized, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 3: Only cucumber/slice detections
        axes[1, 0].imshow(image_rgb)
        axes[1, 0].set_title('3. Cucumber/Slice Detections Only', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        cucumber_detections = [d for d in detections if d['class'] in ['cucumber', 'slice']]
        for det in cucumber_detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            class_name = det['class']
            confidence = det['confidence']
            
            color = self.colors.get(class_name, (255, 255, 255))
            color_normalized = tuple(c/255 for c in color)
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=color_normalized, facecolor='none')
            axes[1, 0].add_patch(rect)
            
            label = f"{class_name}\n{confidence:.3f}"
            axes[1, 0].text(x1, y1-5, label, fontsize=10, color=color_normalized, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 4: Detection statistics table
        axes[1, 1].axis('off')
        
        if detections:
            # Create detection summary table
            table_data = []
            for det in detections:
                table_data.append([
                    det['class'],
                    f"{det['confidence']:.3f}",
                    f"{det['width']:.0f} x {det['height']:.0f}",
                    f"{det['area_percentage']:.2f}%"
                ])
            
            table = axes[1, 1].table(
                cellText=table_data,
                colLabels=['Class', 'Confidence', 'Size (px)', 'Area %'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(4):
                    if i == 0:  # Header row
                        table[(i, j)].set_facecolor('#4CAF50')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    else:
                        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            axes[1, 1].set_title('4. Detection Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the diagnostic visualization
        output_path = Path(image_path).parent / f"{Path(image_path).stem}_diagnostic.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Diagnostic visualization saved: {output_path}")
    
    def run_detection_test(self, image_path, confidence_levels=[0.1, 0.3, 0.5, 0.7]):
        """Test detection at different confidence levels."""
        print(f"🧪 Testing detection at different confidence levels")
        print("=" * 60)
        
        for conf_threshold in confidence_levels:
            print(f"\n🎯 Confidence threshold: {conf_threshold}")
            print("-" * 40)
            
            # Run detection with this threshold
            results = self.yolo_model(image_path, conf=conf_threshold, verbose=False)
            detections = results[0]
            
            cucumber_count = 0
            other_count = 0
            
            for detection in detections.boxes:
                class_id = int(detection.cls[0])
                class_name = self.class_names[class_id]
                
                if class_name in ['cucumber', 'slice']:
                    cucumber_count += 1
                else:
                    other_count += 1
            
            print(f"  Cucumber/slice detections: {cucumber_count}")
            print(f"  Other object detections: {other_count}")
            print(f"  Total detections: {len(detections.boxes)}")

def main():
    parser = argparse.ArgumentParser(description='Detection Diagnostic Tool')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--image-path', required=True, help='Path to image to analyze')
    parser.add_argument('--confidence-test', action='store_true', help='Test different confidence levels')
    
    args = parser.parse_args()
    
    # Initialize diagnostic tool
    diagnostic = DetectionDiagnostic(args.yolo_model)
    
    # Analyze the image
    detections = diagnostic.analyze_image(args.image_path)
    
    # Run confidence level tests if requested
    if args.confidence_test:
        diagnostic.run_detection_test(args.image_path)

if __name__ == "__main__":
    main()
