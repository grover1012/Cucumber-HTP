#!/usr/bin/env python3
"""
Visualize Detection Issues
Shows problematic detections with bounding boxes and confidence scores
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from collections import defaultdict

class DetectionVisualizer:
    def __init__(self, analysis_file):
        """Initialize the detection visualizer."""
        self.analysis_file = analysis_file
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Colors for visualization (BGR format)
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
    
    def visualize_quality_issues(self, quality_threshold=80, max_images=10):
        """Visualize images with quality issues."""
        print(f"🔍 Visualizing Detection Issues (Quality < {quality_threshold}%)")
        print("=" * 60)
        
        # Find problematic images
        problematic_images = []
        for analysis in self.analyses:
            if analysis['quality_score'] < quality_threshold:
                problematic_images.append(analysis)
        
        # Sort by quality score (worst first)
        problematic_images.sort(key=lambda x: x['quality_score'])
        
        print(f"📊 Found {len(problematic_images)} problematic images")
        print(f"🎯 Showing first {min(max_images, len(problematic_images))} images")
        print()
        
        # Show each problematic image
        for i, analysis in enumerate(problematic_images[:max_images]):
            print(f"🖼️  Image {i+1}: {Path(analysis['image_path']).name}")
            print(f"   Quality Score: {analysis['quality_score']:.1f}%")
            print(f"   Total Detections: {analysis['total_detections']}")
            
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
            
            # Visualize the image
            self._visualize_single_image(analysis)
            
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
    
    def visualize_specific_image(self, image_name):
        """Visualize a specific image by name."""
        print(f"🔍 Looking for image: {image_name}")
        
        # Find the image in analyses
        target_analysis = None
        for analysis in self.analyses:
            if image_name in analysis['image_path']:
                target_analysis = analysis
                break
        
        if target_analysis is None:
            print(f"❌ Image not found: {image_name}")
            return
        
        print(f"✅ Found image: {Path(target_analysis['image_path']).name}")
        print(f"   Quality Score: {target_analysis['quality_score']:.1f}%")
        
        self._visualize_single_image(target_analysis)
        
        print("👀 Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _visualize_single_image(self, analysis):
        """Visualize a single image with its detections."""
        image_path = analysis['image_path']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Create a copy for drawing
        display_image = image.copy()
        height, width = image.shape[:2]
        
        # Add title
        title = f"Quality: {analysis['quality_score']:.1f}% | Detections: {analysis['total_detections']}"
        cv2.putText(display_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw detections (we need to get these from the model since we only have analysis)
        # For now, show the image and analysis info
        print(f"   📁 Image path: {image_path}")
        print(f"   📏 Image size: {width}x{height}")
        
        # Show class distribution
        class_dist = analysis['class_distribution']
        print(f"   🏷️  Classes detected: {', '.join([f'{k}: {v}' for k, v in class_dist.items()])}")
        
        # Show confidence distribution
        conf_dist = analysis['confidence_distribution']
        print(f"   📊 Confidence: Very High: {conf_dist.get('very_high', 0)}, "
              f"High: {conf_dist.get('high', 0)}, "
              f"Medium: {conf_dist.get('medium', 0)}, "
              f"Low: {conf_dist.get('low', 0)}, "
              f"Very Low: {conf_dist.get('very_low', 0)}")
        
        # Display the image
        window_name = f"Detection Issues: {Path(image_path).name}"
        cv2.imshow(window_name, display_image)
        
        # Resize window if too large
        cv2.resizeWindow(window_name, min(1200, width), min(800, height))
    
    def show_quality_distribution(self):
        """Show the overall quality distribution."""
        print("📊 OVERALL QUALITY DISTRIBUTION")
        print("=" * 40)
        
        quality_ranges = {
            'Excellent (90-100%)': 0,
            'Good (80-89%)': 0,
            'Fair (60-79%)': 0,
            'Poor (0-59%)': 0
        }
        
        for analysis in self.analyses:
            score = analysis['quality_score']
            if score >= 90:
                quality_ranges['Excellent (90-100%)'] += 1
            elif score >= 80:
                quality_ranges['Good (80-89%)'] += 1
            elif score >= 60:
                quality_ranges['Fair (60-79%)'] += 1
            else:
                quality_ranges['Poor (0-59%)'] += 1
        
        for range_name, count in quality_ranges.items():
            percentage = (count / len(self.analyses)) * 100
            print(f"{range_name}: {count} images ({percentage:.1f}%)")
        
        print()
        
        # Show worst images
        worst_images = sorted(self.analyses, key=lambda x: x['quality_score'])[:5]
        print("🔴 WORST 5 IMAGES:")
        for i, analysis in enumerate(worst_images):
            print(f"{i+1}. {Path(analysis['image_path']).name}: {analysis['quality_score']:.1f}%")
        
        print()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize detection quality issues")
    parser.add_argument("--analysis-file", default="results/enhanced_auto_labels_new/analysis/detailed_analyses.json",
                       help="Path to detailed analyses JSON file")
    parser.add_argument("--quality-threshold", type=int, default=80,
                       help="Quality threshold below which to show images")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to show")
    parser.add_argument("--image-name", type=str, default=None,
                       help="Show specific image by name (partial match)")
    parser.add_argument("--show-distribution", action="store_true",
                       help="Show quality distribution summary")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_file):
        print(f"❌ Analysis file not found: {args.analysis_file}")
        print("Please run the enhanced auto-labeler first.")
        return
    
    # Initialize visualizer
    visualizer = DetectionVisualizer(args.analysis_file)
    
    if args.show_distribution:
        visualizer.show_quality_distribution()
    
    if args.image_name:
        visualizer.visualize_specific_image(args.image_name)
    else:
        visualizer.visualize_quality_issues(args.quality_threshold, args.max_images)

if __name__ == "__main__":
    main()
