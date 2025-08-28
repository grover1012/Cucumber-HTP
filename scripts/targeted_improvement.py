#!/usr/bin/env python3
"""
Targeted Improvement for Problematic Images
Analyzes and improves images with quality scores below 70%
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt

class TargetedImprover:
    def __init__(self, model_path, analysis_file):
        """Initialize the targeted improver."""
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
    
    def identify_problematic_images(self, quality_threshold=70):
        """Identify images with quality below threshold."""
        problematic = [img for img in self.analyses if img['quality_score'] < quality_threshold]
        
        print(f"🔍 Found {len(problematic)} problematic images (quality < {quality_threshold}%):")
        print("=" * 80)
        
        # Group by issue type
        grin_images = []
        complex_scenes = []
        under_detected = []
        
        for img in problematic:
            img_name = Path(img['image_path']).name
            quality = img['quality_score']
            expected = sum(img['class_distribution'].values())
            detected = img.get('total_detections', 0)
            
            print(f"📸 {img_name}")
            print(f"   Quality: {quality:.1f}% | Expected: {expected} | Detected: {detected}")
            print(f"   Classes: {dict(img['class_distribution'])}")
            
            # Categorize issues
            if 'GRIN' in img_name:
                grin_images.append(img)
                print("   🎯 Issue: GRIN image (close-up view)")
            elif detected < expected * 0.7:
                under_detected.append(img)
                print("   🎯 Issue: Under-detection")
            else:
                complex_scenes.append(img)
                print("   🎯 Issue: Complex scene")
            print()
        
        return {
            'grin_images': grin_images,
            'under_detected': under_detected,
            'complex_scenes': complex_scenes,
            'all_problematic': problematic
        }
    
    def test_adaptive_confidence(self, image_path, expected_classes):
        """Test different confidence strategies for a specific image."""
        print(f"🔧 Testing adaptive confidence for: {Path(image_path).name}")
        print("-" * 60)
        
        # Strategy 1: Very low confidence for all classes
        print("📊 Strategy 1: Very low confidence (0.1)")
        results1 = self.model(str(image_path), conf=0.1)
        detections1 = self._count_detections(results1)
        print(f"   Detected: {dict(detections1)}")
        
        # Strategy 2: Class-specific low confidence
        print("📊 Strategy 2: Class-specific low confidence")
        class_conf = {
            'cucumber': 0.15, 'big_ruler': 0.2, 'ruler': 0.2,
            'color_chart': 0.2, 'label': 0.25, 'slice': 0.15,
            'cavity': 0.15, 'hollow': 0.15, 'blue_dot': 0.3,
            'red_dot': 0.3, 'green_dot': 0.3, 'objects': 0.25
        }
        
        # Use the lowest confidence for detected classes
        min_conf = min([class_conf.get(cls, 0.2) for cls in expected_classes.keys()])
        results2 = self.model(str(image_path), conf=min_conf)
        detections2 = self._count_detections(results2)
        print(f"   Detected: {dict(detections2)}")
        
        # Strategy 3: Progressive confidence reduction
        print("📊 Strategy 3: Progressive confidence reduction")
        best_results = None
        best_score = 0
        best_conf = 0.1
        
        for conf in [0.1, 0.15, 0.2, 0.25, 0.3]:
            results = self.model(str(image_path), conf=conf)
            if results and results[0].boxes:
                score = self._calculate_improvement_score(results, expected_classes)
                if score > best_score:
                    best_score = score
                    best_results = results
                    best_conf = conf
        
        if best_results:
            detections3 = self._count_detections(best_results)
            print(f"   Best at confidence {best_conf}: {dict(detections3)}")
            print(f"   Improvement score: {best_score:.2f}")
        
        print()
        return best_results, best_conf, best_score
    
    def _count_detections(self, results):
        """Count detections by class."""
        if not results or not results[0].boxes:
            return {}
        
        detections = defaultdict(int)
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            detections[class_name] += 1
        
        return detections
    
    def _calculate_improvement_score(self, results, expected_classes):
        """Calculate how much the results improve over expected."""
        if not results or not results[0].boxes:
            return 0
        
        detections = self._count_detections(results)
        total_expected = sum(expected_classes.values())
        total_detected = sum(detections.values())
        
        # Score based on coverage and accuracy
        coverage = total_detected / total_expected if total_expected > 0 else 0
        accuracy = sum([min(expected_classes.get(cls, 0), detections.get(cls, 0)) for cls in expected_classes]) / total_expected if total_expected > 0 else 0
        
        return coverage * accuracy
    
    def create_improvement_report(self, problematic_images):
        """Create a comprehensive improvement report."""
        print("📋 CREATING IMPROVEMENT REPORT")
        print("=" * 50)
        
        report = {
            'summary': {
                'total_problematic': len(problematic_images['all_problematic']),
                'grin_images': len(problematic_images['grin_images']),
                'under_detected': len(problematic_images['under_detected']),
                'complex_scenes': len(problematic_images['complex_scenes'])
            },
            'recommendations': [],
            'test_results': []
        }
        
        # Test improvement strategies on sample images
        test_images = problematic_images['all_problematic'][:5]  # Test first 5
        
        for img in test_images:
            print(f"🧪 Testing improvement strategies for: {Path(img['image_path']).name}")
            
            best_results, best_conf, best_score = self.test_adaptive_confidence(
                img['image_path'], img['class_distribution']
            )
            
            report['test_results'].append({
                'image': Path(img['image_path']).name,
                'original_quality': img['quality_score'],
                'best_confidence': best_conf,
                'improvement_score': best_score
            })
        
        # Generate recommendations
        if problematic_images['grin_images']:
            report['recommendations'].append({
                'issue': 'GRIN images (close-up views)',
                'count': len(problematic_images['grin_images']),
                'solution': 'Use very low confidence (0.1-0.15) for all classes',
                'reason': 'Close-up views have different object scales and appearances'
            })
        
        if problematic_images['under_detected']:
            report['recommendations'].append({
                'issue': 'Under-detection',
                'count': len(problematic_images['under_detected']),
                'solution': 'Use class-specific low confidence thresholds',
                'reason': 'Some classes need lower thresholds than others'
            })
        
        if problematic_images['complex_scenes']:
            report['recommendations'].append({
                'issue': 'Complex scenes',
                'count': len(problematic_images['complex_scenes']),
                'solution': 'Use progressive confidence reduction',
                'reason': 'Multiple objects need careful threshold balancing'
            })
        
        # Print recommendations
        print("\n🎯 IMPROVEMENT RECOMMENDATIONS:")
        print("=" * 50)
        for rec in report['recommendations']:
            print(f"📌 {rec['issue']} ({rec['count']} images)")
            print(f"   💡 Solution: {rec['solution']}")
            print(f"   🔍 Reason: {rec['reason']}")
            print()
        
        return report
    
    def save_improvement_report(self, report, output_dir):
        """Save the improvement report to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        report_file = os.path.join(output_dir, 'improvement_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Improvement report saved to: {report_file}")
        return report_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Targeted improvement for problematic images")
    parser.add_argument("--model", default="models/yolo12/cucumber_traits/weights/best.pt",
                       help="Path to trained YOLO12 model")
    parser.add_argument("--analysis-file", default="results/improved_labels_v2/analysis/detailed_analyses.json",
                       help="Path to detailed analyses JSON file")
    parser.add_argument("--output-dir", default="results/improvement_analysis",
                       help="Output directory for improvement reports")
    parser.add_argument("--quality-threshold", type=float, default=70.0,
                       help="Quality threshold for identifying problematic images")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_file):
        print(f"❌ Analysis file not found: {args.analysis_file}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Initialize improver
    improver = TargetedImprover(args.model, args.analysis_file)
    
    # Step 1: Identify problematic images
    print("🔍 STEP 1: IDENTIFYING PROBLEMATIC IMAGES")
    print("=" * 60)
    problematic = improver.identify_problematic_images(args.quality_threshold)
    
    # Step 2: Create improvement report
    print("\n🔧 STEP 2: CREATING IMPROVEMENT REPORT")
    print("=" * 60)
    report = improver.create_improvement_report(problematic)
    
    # Step 3: Save report
    print("\n💾 STEP 3: SAVING IMPROVEMENT REPORT")
    print("=" * 60)
    improver.save_improvement_report(report, args.output_dir)
    
    print("\n🎉 TARGETED IMPROVEMENT ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Review the improvement recommendations")
    print("2. Test the suggested confidence thresholds")
    print("3. Apply improvements to problematic images")

if __name__ == "__main__":
    main()
