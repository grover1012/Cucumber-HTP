#!/usr/bin/env python3
"""
Enhanced Auto-Labeling with Better Detection
Improved confidence thresholds, error analysis, and quality control
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from ultralytics import YOLO
from collections import defaultdict

class EnhancedAutoLabeler:
    def __init__(self, model_path, confidence_threshold=0.3):
        """Initialize enhanced auto-labeler with trained model."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class names mapping
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Class-specific confidence thresholds
        self.class_confidence_thresholds = {
            'cucumber': 0.25,      # Lower threshold for main objects
            'ruler': 0.3,          # Medium threshold for calibration objects
            'label': 0.35,         # Higher threshold for text objects
            'color_chart': 0.3,    # Medium threshold for calibration
            'slice': 0.25,         # Lower threshold for cucumber parts
            'cavity': 0.25,        # Lower threshold for cucumber parts
            'hollow': 0.25,        # Lower threshold for cucumber parts
            'big_ruler': 0.3,      # Medium threshold for calibration
            'blue_dot': 0.4,       # Higher threshold for small objects
            'red_dot': 0.4,        # Higher threshold for small objects
            'green_dot': 0.4,      # Higher threshold for small objects
            'objects': 0.35        # Medium threshold for other objects
        }
        
        # Detection statistics
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'class_detections': defaultdict(int),
            'confidence_ranges': defaultdict(int),
            'low_confidence_detections': 0,
            'potential_misses': 0
        }
    
    def get_confidence_threshold(self, class_name):
        """Get class-specific confidence threshold."""
        return self.class_confidence_thresholds.get(class_name, self.confidence_threshold)
    
    def analyze_detection_quality(self, results, image_path):
        """Analyze detection quality and identify potential issues."""
        analysis = {
            'image_path': str(image_path),
            'total_detections': 0,
            'class_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'low_confidence_objects': [],
            'potential_misses': [],
            'quality_score': 0.0
        }
        
        if not results or not results[0].boxes:
            analysis['quality_score'] = 0.0
            analysis['potential_misses'].append('No objects detected - possible model failure')
            return analysis
        
        boxes = results[0].boxes
        analysis['total_detections'] = len(boxes)
        
        # Analyze each detection
        for box in boxes:
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            
            # Update statistics
            analysis['class_distribution'][class_name] += 1
            self.stats['class_detections'][class_name] += 1
            
            # Confidence analysis
            if confidence < 0.5:
                analysis['low_confidence_objects'].append({
                    'class': class_name,
                    'confidence': confidence,
                    'threshold': self.get_confidence_threshold(class_name)
                })
                self.stats['low_confidence_detections'] += 1
            
            # Confidence range categorization
            if confidence < 0.3:
                analysis['confidence_distribution']['very_low'] += 1
            elif confidence < 0.5:
                analysis['confidence_distribution']['low'] += 1
            elif confidence < 0.7:
                analysis['confidence_distribution']['medium'] += 1
            elif confidence < 0.9:
                analysis['confidence_distribution']['high'] += 1
            else:
                analysis['confidence_distribution']['very_high'] += 1
        
        # Quality scoring
        high_conf_count = analysis['confidence_distribution']['high'] + analysis['confidence_distribution']['very_high']
        total_detections = analysis['total_detections']
        
        if total_detections > 0:
            analysis['quality_score'] = (high_conf_count / total_detections) * 100
        
        # Identify potential issues
        if analysis['total_detections'] == 0:
            analysis['potential_misses'].append('No objects detected')
        elif analysis['total_detections'] < 3:
            analysis['potential_misses'].append('Very few objects - possible under-detection')
        elif analysis['quality_score'] < 50:
            analysis['potential_misses'].append('Low confidence detections - possible false positives')
        
        return analysis
    
    def convert_to_yolo_format(self, results, image_path):
        """Convert YOLO results to YOLO format labels with quality filtering."""
        yolo_labels = []
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return yolo_labels
            
        height, width = img.shape[:2]
        
        if not results or not results[0].boxes:
            return yolo_labels
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                    
                    # Apply class-specific confidence threshold
                    threshold = self.get_confidence_threshold(class_name)
                    
                    if confidence >= threshold:
                        # Convert to YOLO format (normalized coordinates)
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        
                        # Validate coordinates
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < w <= 1 and 0 < h <= 1):
                            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        return yolo_labels
    
    def auto_label_directory(self, input_dir, output_dir, save_analysis=True):
        """Auto-label all images in a directory with enhanced quality control."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create analysis directory
        analysis_dir = output_path / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"📸 Found {len(image_files)} images to auto-label")
        print(f"🎯 Using enhanced confidence thresholds:")
        for class_name, threshold in self.class_confidence_thresholds.items():
            print(f"  {class_name}: {threshold:.2f}")
        print("=" * 60)
        
        labeled_count = 0
        total_detections = 0
        all_analyses = []
        
        for i, image_file in enumerate(image_files):
            print(f"🔄 Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Run inference with lower base confidence
                results = self.model(str(image_file), conf=0.2)
                
                # Analyze detection quality
                analysis = self.analyze_detection_quality(results, image_file)
                all_analyses.append(analysis)
                
                # Convert to YOLO format with quality filtering
                yolo_labels = self.convert_to_yolo_format(results, image_file)
                
                if yolo_labels:
                    # Save YOLO format labels
                    label_file = output_path / f"{image_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        for label in yolo_labels:
                            f.write(label + '\n')
                    
                    labeled_count += 1
                    total_detections += len(yolo_labels)
                    
                    # Print quality summary
                    quality_emoji = "✅" if analysis['quality_score'] >= 70 else "⚠️" if analysis['quality_score'] >= 50 else "❌"
                    print(f"  {quality_emoji} Labeled with {len(yolo_labels)} objects (Quality: {analysis['quality_score']:.1f}%)")
                    
                    if analysis['potential_misses']:
                        print(f"  ⚠️ Issues: {', '.join(analysis['potential_misses'])}")
                else:
                    print(f"  ⚠️ No objects passed quality filters")
                    
            except Exception as e:
                print(f"  ❌ Error processing {image_file.name}: {e}")
        
        # Update global statistics
        self.stats['total_images'] = len(image_files)
        self.stats['total_detections'] = total_detections
        
        # Save detailed analysis
        if save_analysis:
            self._save_analysis_report(all_analyses, analysis_dir)
        
        # Print final summary
        self._print_final_summary(labeled_count, total_detections, all_analyses)
        
        return labeled_count, total_detections
    
    def _save_analysis_report(self, analyses, analysis_dir):
        """Save detailed analysis report."""
        # Summary statistics
        summary = {
            'total_images': len(analyses),
            'total_detections': sum(a['total_detections'] for a in analyses),
            'average_quality_score': np.mean([a['quality_score'] for a in analyses]),
            'class_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'quality_distribution': {
                'excellent': len([a for a in analyses if a['quality_score'] >= 80]),
                'good': len([a for a in analyses if 60 <= a['quality_score'] < 80]),
                'fair': len([a for a in analyses if 40 <= a['quality_score'] < 60]),
                'poor': len([a for a in analyses if a['quality_score'] < 40])
            }
        }
        
        # Aggregate class and confidence distributions
        for analysis in analyses:
            for class_name, count in analysis['class_distribution'].items():
                summary['class_distribution'][class_name] += count
            for conf_level, count in analysis['confidence_distribution'].items():
                summary['confidence_distribution'][conf_level] += count
        
        # Save summary
        summary_file = analysis_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed analyses
        details_file = analysis_dir / "detailed_analyses.json"
        with open(details_file, 'w') as f:
            json.dump(analyses, f, indent=2, default=str)
        
        print(f"📊 Analysis reports saved to: {analysis_dir}")
    
    def _print_final_summary(self, labeled_count, total_detections, analyses):
        """Print comprehensive final summary."""
        print("\n" + "=" * 60)
        print("🎉 ENHANCED AUTO-LABELING COMPLETE!")
        print("=" * 60)
        
        print(f"📊 Results Summary:")
        print(f"  Images processed: {len(analyses)}")
        print(f"  Images labeled: {labeled_count}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average quality score: {np.mean([a['quality_score'] for a in analyses]):.1f}%")
        
        print(f"\n🔍 Quality Distribution:")
        quality_dist = {
            'excellent': len([a for a in analyses if a['quality_score'] >= 80]),
            'good': len([a for a in analyses if 60 <= a['quality_score'] < 80]),
            'fair': len([a for a in analyses if 40 <= a['quality_score'] < 60]),
            'poor': len([a for a in analyses if a['quality_score'] < 40])
        }
        
        for level, count in quality_dist.items():
            percentage = (count / len(analyses)) * 100
            print(f"  {level.capitalize()}: {count} images ({percentage:.1f}%)")
        
        print(f"\n📈 Detection Statistics:")
        for class_name, count in self.stats['class_detections'].items():
            print(f"  {class_name}: {count} detections")
        
        print(f"\n⚠️ Quality Issues Found:")
        issues = []
        for analysis in analyses:
            issues.extend(analysis['potential_misses'])
        
        if issues:
            issue_counts = defaultdict(int)
            for issue in issues:
                issue_counts[issue] += 1
            
            for issue, count in issue_counts.items():
                print(f"  {issue}: {count} occurrences")
        else:
            print("  None detected - excellent quality!")
        
        print(f"\n🎯 Recommendations:")
        if np.mean([a['quality_score'] for a in analyses]) < 70:
            print("  - Consider lowering confidence thresholds for better recall")
            print("  - Review images with low quality scores")
            print("  - Check for systematic detection failures")
        else:
            print("  - Quality is excellent! Ready for training")
            print("  - Consider manual review of low-confidence detections")
        
        print("=" * 60)

def main():
    """Main function for enhanced auto-labeling."""
    parser = argparse.ArgumentParser(description="Enhanced auto-labeling with quality control")
    parser.add_argument("--model", required=True, help="Path to trained YOLO12 model")
    parser.add_argument("--input-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", required=True, help="Output directory for labels")
    parser.add_argument("--confidence", type=float, default=0.3, help="Base confidence threshold")
    parser.add_argument("--save-analysis", action="store_true", help="Save detailed analysis reports")
    
    args = parser.parse_args()
    
    # Initialize enhanced labeler
    labeler = EnhancedAutoLabeler(args.model, args.confidence)
    
    # Start enhanced auto-labeling
    labeled_count, total_detections = labeler.auto_label_directory(
        args.input_dir, 
        args.output_dir, 
        args.save_analysis
    )

if __name__ == "__main__":
    main()
