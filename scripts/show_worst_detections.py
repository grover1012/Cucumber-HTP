#!/usr/bin/env python3
"""
Show Worst Detections
Displays the worst performing images with their actual detections
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from ultralytics import YOLO
from collections import defaultdict

class WorstDetectionViewer:
    def __init__(self, model_path, analysis_file):
        """Initialize the worst detection viewer."""
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.analysis_file = analysis_file
        
        # Class names and colors
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
        
        # Load analysis data
        with open(analysis_file, 'r') as f:
            self.analyses = json.load(f)
    
    def show_worst_detections(self, num_images=5):
        """Show the worst performing images with their detections."""
        print(f"🔍 SHOWING WORST {num_images} DETECTIONS")
        print("=" * 60)
        
        # Sort by quality score (worst first)
        worst_images = sorted(self.analyses, key=lambda x: x['quality_score'])[:num_images]
        
        for i, analysis in enumerate(worst_images):
            print(f"\n🖼️  Image {i+1}/{num_images}: {Path(analysis['image_path']).name}")
            print(f"   Quality Score: {analysis['quality_score']:.1f}%")
            print(f"   Expected Detections: {analysis['total_detections']}")
            
            # Show low confidence objects
            if analysis['low_confidence_objects']:
                print("   ⚠️ Low Confidence Objects:")
                for obj in analysis['low_confidence_objects']:
                    print(f"     - {obj['class']}: {obj['confidence']:.3f}")
            
            # Show potential issues
            if analysis['potential_misses']:
                print("   ❌ Issues:")
                for issue in analysis['potential_misses']:
                    print(f"     - {issue}")
            
            print()
            
            # Run detection and show results
            self._show_detection_results(analysis)
            
            # Wait for user input
            print("👀 Press SPACE to continue, Q to quit, R to restart")
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("👋 Stopped by user")
                break
            elif key == ord('r') or key == ord('R'):
                print("🔄 Restarting...")
                i = 0
                continue
            
            # Close current window
            cv2.destroyAllWindows()
        
        cv2.destroyAllWindows()
        print("✅ Visualization complete!")
    
    def _show_detection_results(self, analysis):
        """Show detection results for a specific image."""
        image_path = analysis['image_path']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Run detection with very low confidence to see everything
        results = self.model(str(image_path), conf=0.1)
        
        if not results or not results[0].boxes:
            print(f"  ⚠️ No objects detected even at low confidence")
            return
        
        # Create display image
        display_image = image.copy()
        height, width = image.shape[:2]
        
        # Add title
        title = f"Quality: {analysis['quality_score']:.1f}% | Expected: {analysis['total_detections']} | Actual: {len(results[0].boxes)}"
        cv2.putText(display_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detections
        boxes = results[0].boxes
        actual_detections = defaultdict(int)
        
        for i, box in enumerate(boxes):
            # Get detection info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            # Get class name and color
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            color = self.class_colors[class_id] if class_id < len(self.class_colors) else (0, 255, 0)
            
            # Count detections
            actual_detections[class_name] += 1
            
            # Draw bounding box
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.3f}"
            cv2.putText(display_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show comparison
        print(f"  📊 Detection Comparison:")
        print(f"     Expected: {dict(analysis['class_distribution'])}")
        print(f"     Actual:   {dict(actual_detections)}")
        
        # Show confidence distribution
        conf_levels = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf >= 0.8:
                conf_levels['very_high'] += 1
            elif conf >= 0.6:
                conf_levels['high'] += 1
            elif conf >= 0.4:
                conf_levels['medium'] += 1
            elif conf >= 0.2:
                conf_levels['low'] += 1
            else:
                conf_levels['very_low'] += 1
        
        print(f"     Confidence: Very High: {conf_levels['very_high']}, "
              f"High: {conf_levels['high']}, "
              f"Medium: {conf_levels['medium']}, "
              f"Low: {conf_levels['low']}, "
              f"Very Low: {conf_levels['very_low']}")
        
        # Display the image
        window_name = f"Worst Detection: {Path(image_path).name}"
        cv2.imshow(window_name, display_image)
        
        # Resize window if too large
        cv2.resizeWindow(window_name, min(1200, width), min(800, height))

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Show worst performing detections")
    parser.add_argument("--model", default="models/yolo12/cucumber_traits/weights/best.pt",
                       help="Path to trained YOLO12 model")
    parser.add_argument("--analysis-file", default="results/enhanced_auto_labels_new/analysis/detailed_analyses.json",
                       help="Path to detailed analyses JSON file")
    parser.add_argument("--num-images", type=int, default=5,
                       help="Number of worst images to show")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_file):
        print(f"❌ Analysis file not found: {args.analysis_file}")
        print("Please run the enhanced auto-labeler first.")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Initialize viewer
    viewer = WorstDetectionViewer(args.model, args.analysis_file)
    
    # Show worst detections
    viewer.show_worst_detections(args.num_images)

if __name__ == "__main__":
    main()
