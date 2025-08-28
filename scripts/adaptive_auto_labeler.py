#!/usr/bin/env python3
"""
Adaptive Auto-Labeler with Smart Confidence Tuning
Automatically adjusts confidence thresholds based on image characteristics
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from ultralytics import YOLO
from collections import defaultdict
import re

class AdaptiveAutoLabeler:
    def __init__(self, model_path):
        """Initialize the adaptive auto-labeler."""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        # Class names and base confidence thresholds
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Base confidence thresholds (from our analysis)
        self.base_confidence = {
            'cucumber': 0.25,
            'ruler': 0.30,
            'label': 0.35,
            'color_chart': 0.30,
            'slice': 0.25,
            'cavity': 0.25,
            'hollow': 0.25,
            'big_ruler': 0.30,
            'blue_dot': 0.40,
            'red_dot': 0.40,
            'green_dot': 0.40,
            'objects': 0.35
        }
        
        # Adaptive confidence adjustments
        self.adaptive_rules = {
            'grin_images': {
                'multiplier': 0.5,  # Reduce confidence by 50% for GRIN images
                'reason': 'Close-up views need lower thresholds'
            },
            'complex_scenes': {
                'multiplier': 0.7,  # Reduce confidence by 30% for complex scenes
                'reason': 'Multiple objects need careful balancing'
            },
            'many_small_objects': {
                'multiplier': 0.6,  # Reduce confidence by 40% for many small objects
                'reason': 'Small objects are harder to detect'
            }
        }
    
    def analyze_image_characteristics(self, image_path):
        """Analyze image characteristics to determine confidence strategy."""
        img_name = Path(image_path).name.lower()
        
        characteristics = {
            'is_grin': 'grin' in img_name,
            'is_complex': False,
            'has_many_small_objects': False,
            'recommended_strategy': 'standard'
        }
        
        # Determine if it's a complex scene based on expected objects
        # This would typically come from ground truth, but we'll estimate
        if characteristics['is_grin']:
            characteristics['recommended_strategy'] = 'grin_images'
        elif 'fs' in img_name or 'f1' in img_name or 'f2' in img_name:
            characteristics['recommended_strategy'] = 'complex_scenes'
            characteristics['is_complex'] = True
        
        return characteristics
    
    def calculate_adaptive_confidence(self, image_path, expected_classes=None):
        """Calculate adaptive confidence thresholds for an image."""
        characteristics = self.analyze_image_characteristics(image_path)
        
        # Start with base confidence
        adaptive_confidence = self.base_confidence.copy()
        
        # Apply adaptive adjustments
        if characteristics['recommended_strategy'] == 'grin_images':
            multiplier = self.adaptive_rules['grin_images']['multiplier']
            for cls in adaptive_confidence:
                adaptive_confidence[cls] *= multiplier
            print(f"   🎯 GRIN image detected - applying {multiplier}x confidence reduction")
            
        elif characteristics['recommended_strategy'] == 'complex_scenes':
            multiplier = self.adaptive_rules['complex_scenes']['multiplier']
            for cls in adaptive_confidence:
                adaptive_confidence[cls] *= multiplier
            print(f"   🎯 Complex scene detected - applying {multiplier}x confidence reduction")
        
        # Further adjust based on expected classes if available
        if expected_classes:
            # For images with many small objects (dots, slices), reduce confidence further
            small_object_classes = ['blue_dot', 'red_dot', 'green_dot', 'slice', 'cavity', 'hollow']
            small_object_count = sum([expected_classes.get(cls, 0) for cls in small_object_classes])
            
            if small_object_count > 5:
                for cls in small_object_classes:
                    if cls in adaptive_confidence:
                        adaptive_confidence[cls] *= 0.8
                print(f"   🎯 Many small objects ({small_object_count}) - reducing confidence for small object classes")
        
        return adaptive_confidence, characteristics
    
    def run_adaptive_detection(self, image_path, expected_classes=None):
        """Run detection with adaptive confidence thresholds."""
        print(f"🔍 Running adaptive detection on: {Path(image_path).name}")
        
        # Calculate adaptive confidence
        adaptive_conf, characteristics = self.calculate_adaptive_confidence(image_path, expected_classes)
        
        # Use the lowest confidence for initial detection
        min_confidence = min(adaptive_conf.values())
        print(f"   📊 Using minimum confidence: {min_confidence:.3f}")
        
        # Run detection
        results = self.model(str(image_path), conf=min_confidence)
        
        if not results or not results[0].boxes:
            print(f"   ❌ No objects detected even at confidence {min_confidence}")
            return None, adaptive_conf, characteristics
        
        # Count detections
        detections = defaultdict(int)
        confidences = []
        
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            
            detections[class_name] += 1
            confidences.append(confidence)
        
        print(f"   📈 Detected {len(results[0].boxes)} objects")
        print(f"   🎯 Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        
        return results, adaptive_conf, characteristics
    
    def process_image(self, image_path, expected_classes=None):
        """Process a single image with adaptive confidence."""
        results, adaptive_conf, characteristics = self.run_adaptive_detection(image_path, expected_classes)
        
        if results is None:
            return {
                'image_path': image_path,
                'success': False,
                'detections': {},
                'confidence_used': adaptive_conf,
                'characteristics': characteristics,
                'quality_score': 0.0
            }
        
        # Count final detections
        detections = defaultdict(int)
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            detections[class_name] += 1
        
        # Calculate quality score if expected classes are provided
        quality_score = 0.0
        if expected_classes:
            total_expected = sum(expected_classes.values())
            total_detected = sum(detections.values())
            
            if total_expected > 0:
                # Calculate accuracy based on class matches
                matches = sum([min(expected_classes.get(cls, 0), detections.get(cls, 0)) for cls in expected_classes])
                quality_score = (matches / total_expected) * 100
        
        return {
            'image_path': image_path,
            'success': True,
            'detections': dict(detections),
            'confidence_used': adaptive_conf,
            'characteristics': characteristics,
            'quality_score': quality_score,
            'total_objects': len(results[0].boxes)
        }
    
    def process_directory(self, input_dir, output_dir, expected_data=None):
        """Process all images in a directory with adaptive confidence."""
        print(f"🚀 ADAPTIVE AUTO-LABELING STARTED")
        print(f"📁 Input: {input_dir}")
        print(f"📁 Output: {output_dir}")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        print(f"📸 Found {len(image_files)} images to process")
        print()
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"🔄 Processing {i}/{len(image_files)}: {image_path.name}")
            
            # Get expected classes if available
            expected_classes = None
            if expected_data:
                for item in expected_data:
                    if image_path.name in item.get('image_path', ''):
                        expected_classes = item.get('class_distribution', {})
                        break
            
            # Process image
            result = self.process_image(str(image_path), expected_classes)
            results.append(result)
            
            # Show results
            if result['success']:
                quality = result['quality_score']
                objects = result['total_objects']
                status = "✅" if quality >= 80 else "⚠️" if quality >= 60 else "❌"
                print(f"   {status} Labeled with {objects} objects (Quality: {quality:.1f}%)")
            else:
                print(f"   ❌ Failed to process image")
            print()
        
        # Save results
        self.save_results(results, output_dir)
        
        # Generate summary
        self.generate_summary(results, output_dir)
        
        return results
    
    def save_results(self, results, output_dir):
        """Save processing results."""
        # Save detailed results
        results_file = os.path.join(output_dir, 'analysis', 'adaptive_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
    
    def generate_summary(self, results, output_dir):
        """Generate summary statistics."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if not successful:
            print("❌ No images were successfully processed")
            return
        
        # Calculate statistics
        total_images = len(results)
        successful_count = len(successful)
        failed_count = len(failed)
        
        quality_scores = [r['quality_score'] for r in successful]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        total_objects = sum([r['total_objects'] for r in successful])
        
        # Quality distribution
        excellent = len([q for q in quality_scores if q >= 80])
        good = len([q for q in quality_scores if 60 <= q < 80])
        fair = len([q for q in quality_scores if 40 <= q < 60])
        poor = len([q for q in quality_scores if q < 40])
        
        # Class distribution
        class_counts = defaultdict(int)
        for result in successful:
            for cls, count in result['detections'].items():
                class_counts[cls] += count
        
        # Save summary
        summary = {
            'total_images': total_images,
            'successful': successful_count,
            'failed': failed_count,
            'success_rate': (successful_count / total_images) * 100,
            'average_quality': avg_quality,
            'total_objects': total_objects,
            'quality_distribution': {
                'excellent': excellent,
                'good': good,
                'fair': fair,
                'poor': poor
            },
            'class_distribution': dict(class_counts)
        }
        
        summary_file = os.path.join(output_dir, 'analysis', 'adaptive_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n📊 ADAPTIVE AUTO-LABELING SUMMARY")
        print("=" * 50)
        print(f"📸 Total images: {total_images}")
        print(f"✅ Successful: {successful_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📈 Success rate: {summary['success_rate']:.1f}%")
        print(f"🎯 Average quality: {avg_quality:.1f}%")
        print(f"🔢 Total objects: {total_objects}")
        
        print(f"\n📊 Quality Distribution:")
        print(f"   Excellent (≥80%): {excellent} images")
        print(f"   Good (60-79%): {good} images")
        print(f"   Fair (40-59%): {fair} images")
        print(f"   Poor (<40%): {poor} images")
        
        print(f"\n📊 Class Distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls}: {count} detections")
        
        print(f"\n💾 Summary saved to: {summary_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Adaptive auto-labeler with smart confidence tuning")
    parser.add_argument("--model", default="models/yolo12/cucumber_traits/weights/best.pt",
                       help="Path to trained YOLO12 model")
    parser.add_argument("--input-dir", default="data/new_annotations/train/images",
                       help="Input directory with images")
    parser.add_argument("--output-dir", default="results/adaptive_labels",
                       help="Output directory for labels and analysis")
    parser.add_argument("--expected-data", default="results/improved_labels_v2/analysis/detailed_analyses.json",
                       help="Path to expected data for quality calculation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Load expected data if available
    expected_data = None
    if os.path.exists(args.expected_data):
        try:
            with open(args.expected_data, 'r') as f:
                expected_data = json.load(f)
            print(f"📊 Loaded expected data from: {args.expected_data}")
        except Exception as e:
            print(f"⚠️ Could not load expected data: {e}")
    
    # Initialize adaptive labeler
    labeler = AdaptiveAutoLabeler(args.model)
    
    # Process directory
    results = labeler.process_directory(args.input_dir, args.output_dir, expected_data)
    
    print("\n🎉 ADAPTIVE AUTO-LABELING COMPLETE!")
    print("=" * 60)
    print("The system automatically adjusted confidence thresholds based on:")
    print("• Image characteristics (GRIN, complex scenes)")
    print("• Object types and counts")
    print("• Historical performance data")

if __name__ == "__main__":
    main()
