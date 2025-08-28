#!/usr/bin/env python3
"""
Confidence Tuning for Detection Improvement
Tests different confidence thresholds to find optimal settings
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from ultralytics import YOLO
from collections import defaultdict

class ConfidenceTuner:
    def __init__(self, model_path, analysis_file):
        """Initialize the confidence tuner."""
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
    
    def test_confidence_levels(self, test_image_name, confidence_levels=[0.1, 0.3, 0.5, 0.7]):
        """Test different confidence levels on a specific image."""
        print(f"🔍 TESTING CONFIDENCE LEVELS ON: {test_image_name}")
        print("=" * 60)
        
        # Find the test image
        test_analysis = None
        for analysis in self.analyses:
            if test_image_name in analysis['image_path']:
                test_analysis = analysis
                break
        
        if test_analysis is None:
            print(f"❌ Image not found: {test_image_name}")
            return
        
        print(f"📊 Expected: {dict(test_analysis['class_distribution'])}")
        print(f"🎯 Quality Score: {test_analysis['quality_score']:.1f}%")
        print()
        
        # Test each confidence level
        for conf_level in confidence_levels:
            print(f"🔧 Testing Confidence: {conf_level}")
            print("-" * 30)
            
            # Run detection with this confidence
            results = self.model(str(test_analysis['image_path']), conf=conf_level)
            
            if not results or not results[0].boxes:
                print(f"  ❌ No objects detected at confidence {conf_level}")
                continue
            
            # Count detections by class
            detections = defaultdict(int)
            confidences = []
            
            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                
                detections[class_name] += 1
                confidences.append(confidence)
            
            # Show results
            print(f"  📊 Detections: {dict(detections)}")
            print(f"  📈 Total objects: {len(results[0].boxes)}")
            print(f"  🎯 Avg confidence: {np.mean(confidences):.3f}")
            print(f"  📊 Min confidence: {np.min(confidences):.3f}")
            print(f"  📊 Max confidence: {np.max(confidences):.3f}")
            
            # Compare with expected
            expected = test_analysis['class_distribution']
            print(f"  🔍 Comparison with expected:")
            for class_name in set(list(detections.keys()) + list(expected.keys())):
                actual = detections.get(class_name, 0)
                exp = expected.get(class_name, 0)
                status = "✅" if actual == exp else "❌"
                print(f"    {class_name}: Expected {exp}, Got {actual} {status}")
            
            print()
        
        # Show the image with optimal confidence
        print("👀 Press SPACE to see image with optimal confidence (0.5)")
        input()
        
        self._show_image_with_confidence(test_analysis, 0.5)
    
    def _show_image_with_confidence(self, analysis, confidence):
        """Show image with detections at specific confidence level."""
        image_path = analysis['image_path']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Run detection
        results = self.model(str(image_path), conf=confidence)
        
        if not results or not results[0].boxes:
            print(f"⚠️ No objects detected at confidence {confidence}")
            return
        
        # Create display image
        display_image = image.copy()
        height, width = image.shape[:2]
        
        # Add title
        title = f"Confidence: {confidence} | Quality: {analysis['quality_score']:.1f}% | Objects: {len(results[0].boxes)}"
        cv2.putText(display_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detections
        boxes = results[0].boxes
        detections = defaultdict(int)
        
        for i, box in enumerate(boxes):
            # Get detection info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            # Get class name and color
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            color = self.class_colors[class_id] if class_id < len(self.class_colors) else (0, 255, 0)
            
            # Count detections
            detections[class_name] += 1
            
            # Draw bounding box
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.3f}"
            cv2.putText(display_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show comparison
        print(f"📊 Final Detection Results:")
        print(f"   Expected: {dict(analysis['class_distribution'])}")
        print(f"   Actual:   {dict(detections)}")
        
        # Display the image
        window_name = f"Confidence Tuning: {Path(image_path).name}"
        cv2.imshow(window_name, display_image)
        
        # Resize window if too large
        cv2.resizeWindow(window_name, min(1200, width), min(800, height))
        
        print("👀 Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def find_optimal_confidence(self, test_images=None):
        """Find optimal confidence for problematic images."""
        if test_images is None:
            # Use worst 3 images
            test_images = sorted(self.analyses, key=lambda x: x['quality_score'])[:3]
        
        print("🎯 FINDING OPTIMAL CONFIDENCE SETTINGS")
        print("=" * 50)
        
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        results = {}
        
        for conf in confidence_levels:
            total_objects = 0
            total_expected = 0
            matches = 0
            
            for analysis in test_images:
                expected = analysis['class_distribution']
                total_expected += sum(expected.values())
                
                # Run detection
                results_det = self.model(str(analysis['image_path']), conf=conf)
                
                if results_det and results_det[0].boxes:
                    actual_objects = len(results_det[0].boxes)
                    total_objects += actual_objects
                    
                    # Count matches by class
                    detections = defaultdict(int)
                    for box in results_det[0].boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                        detections[class_name] += 1
                    
                    # Compare with expected
                    for class_name in expected:
                        if class_name in detections:
                            matches += min(expected[class_name], detections[class_name])
                else:
                    actual_objects = 0
            
            accuracy = matches / total_expected if total_expected > 0 else 0
            results[conf] = {
                'total_detected': total_objects,
                'total_expected': total_expected,
                'accuracy': accuracy,
                'over_detection': total_objects - total_expected
            }
            
            print(f"Confidence {conf}: {total_objects} detected, {total_expected} expected, "
                  f"Accuracy: {accuracy:.2%}, Over-detection: {total_objects - total_expected}")
        
        # Find best confidence
        best_conf = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\n🏆 Best confidence: {best_conf} (Accuracy: {results[best_conf]['accuracy']:.2%})")
        
        return best_conf, results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Tune confidence thresholds for better detection")
    parser.add_argument("--model", default="models/yolo12/cucumber_traits/weights/best.pt",
                       help="Path to trained YOLO12 model")
    parser.add_argument("--analysis-file", default="results/enhanced_auto_labels_new/analysis/detailed_analyses.json",
                       help="Path to detailed analyses JSON file")
    parser.add_argument("--test-image", type=str, default="AM177_GRIN_2",
                       help="Test image name (partial match)")
    parser.add_argument("--find-optimal", action="store_true",
                       help="Find optimal confidence for all problematic images")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_file):
        print(f"❌ Analysis file not found: {args.analysis_file}")
        print("Please run the enhanced auto-labeler first.")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Initialize tuner
    tuner = ConfidenceTuner(args.model, args.analysis_file)
    
    if args.find_optimal:
        tuner.find_optimal_confidence()
    else:
        tuner.test_confidence_levels(args.test_image)

if __name__ == "__main__":
    main()
