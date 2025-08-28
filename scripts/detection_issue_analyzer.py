#!/usr/bin/env python3
"""
Detection Issue Analyzer
Identifies and helps fix specific detection problems
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class DetectionIssueAnalyzer:
    def __init__(self, model_path):
        """Initialize detection issue analyzer."""
        self.model = YOLO(model_path)
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Common detection issues and their solutions
        self.issue_patterns = {
            'missing_cucumbers': {
                'description': 'Cucumbers not detected in images',
                'causes': ['Low confidence threshold', 'Poor lighting', 'Occlusion', 'Small size'],
                'solutions': ['Lower confidence threshold', 'Improve lighting', 'Check for overlap', 'Adjust minimum size']
            },
            'false_positives': {
                'description': 'Objects detected that are not cucumbers',
                'causes': ['High confidence threshold', 'Similar objects', 'Background noise'],
                'solutions': ['Increase confidence threshold', 'Add negative examples', 'Improve training data']
            },
            'poor_bounding_boxes': {
                'description': 'Inaccurate bounding box coordinates',
                'causes': ['Model training issues', 'Image resolution', 'Object boundaries'],
                'solutions': ['Retrain model', 'Increase image resolution', 'Improve annotation quality']
            },
            'class_confusion': {
                'description': 'Objects classified incorrectly',
                'causes': ['Similar class appearances', 'Insufficient training data', 'Class imbalance'],
                'solutions': ['Add more training examples', 'Balance class distribution', 'Improve class features']
            }
        }
    
    def analyze_single_image(self, image_path, confidence_levels=[0.1, 0.3, 0.5, 0.7]):
        """Analyze detection issues at different confidence levels."""
        print(f"🔍 Analyzing: {Path(image_path).name}")
        print("=" * 50)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print("❌ Could not load image")
            return None
        
        height, width = image.shape[:2]
        print(f"📐 Image dimensions: {width}x{height}")
        
        analysis_results = {}
        
        for conf in confidence_levels:
            print(f"\n🎯 Confidence threshold: {conf}")
            
            # Run inference
            results = self.model(str(image_path), conf=conf)
            
            if not results or not results[0].boxes:
                print(f"  ❌ No objects detected at confidence {conf}")
                analysis_results[conf] = {
                    'detections': 0,
                    'objects': [],
                    'issues': ['No objects detected']
                }
                continue
            
            boxes = results[0].boxes
            detections = []
            
            for i, box in enumerate(boxes):
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                
                # Calculate object properties
                obj_width = x2 - x1
                obj_height = y2 - y1
                obj_area = obj_width * obj_height
                obj_center_x = (x1 + x2) / 2
                obj_center_y = (y1 + y2) / 2
                
                detection_info = {
                    'id': i,
                    'class_name': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'width': obj_width,
                    'height': obj_height,
                    'area': obj_area,
                    'center': [obj_center_x, obj_center_y],
                    'relative_size': obj_area / (width * height)
                }
                
                detections.append(detection_info)
            
            # Analyze detections for this confidence level
            issues = self._identify_detection_issues(detections, width, height)
            
            analysis_results[conf] = {
                'detections': len(detections),
                'objects': detections,
                'issues': issues
            }
            
            print(f"  ✅ Detected {len(detections)} objects")
            for obj in detections:
                print(f"    - {obj['class_name']}: conf={obj['confidence']:.3f}, size={obj['relative_size']:.4f}")
            
            if issues:
                print(f"  ⚠️ Issues found: {', '.join(issues)}")
        
        return analysis_results
    
    def _identify_detection_issues(self, detections, image_width, image_height):
        """Identify specific detection issues."""
        issues = []
        
        if not detections:
            issues.append('No objects detected')
            return issues
        
        # Check for missing essential objects
        essential_classes = ['cucumber', 'ruler', 'label']
        detected_classes = [d['class_name'] for d in detections]
        
        for essential in essential_classes:
            if essential not in detected_classes:
                issues.append(f'Missing {essential}')
        
        # Check for suspicious detections
        for detection in detections:
            # Very small objects
            if detection['relative_size'] < 0.001:
                issues.append(f'Very small {detection["class_name"]} (size: {detection["relative_size"]:.6f})')
            
            # Very large objects (might be false positives)
            if detection['relative_size'] > 0.5:
                issues.append(f'Very large {detection["class_name"]} (size: {detection["relative_size"]:.4f})')
            
            # Low confidence detections
            if detection['confidence'] < 0.4:
                issues.append(f'Low confidence {detection["class_name"]} (conf: {detection["confidence"]:.3f})')
        
        # Check for overlapping objects
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], i+1):
                overlap = self._calculate_overlap(det1['bbox'], det2['bbox'])
                if overlap > 0.7:  # 70% overlap
                    issues.append(f'High overlap between {det1["class_name"]} and {det2["class_name"]} ({overlap:.2f})')
        
        return issues
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def analyze_dataset_issues(self, images_dir, labels_dir, sample_size=50):
        """Analyze detection issues across the dataset."""
        print(f"🔍 Analyzing dataset issues (sample size: {sample_size})")
        print("=" * 60)
        
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        # Get sample images
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
        
        if len(image_files) > sample_size:
            import random
            image_files = random.sample(image_files, sample_size)
        
        print(f"📸 Analyzing {len(image_files)} sample images...")
        
        dataset_issues = defaultdict(int)
        class_detection_stats = defaultdict(lambda: {'count': 0, 'avg_confidence': 0.0, 'size_range': []})
        
        for i, image_file in enumerate(image_files):
            print(f"🔄 Progress: {i+1}/{len(image_files)}")
            
            # Check if corresponding label exists
            label_file = labels_path / f"{image_file.stem}.txt"
            
            if not label_file.exists():
                dataset_issues['missing_labels'] += 1
                continue
            
            # Analyze image
            analysis = self.analyze_single_image(str(image_file), confidence_levels=[0.3])
            
            if analysis and 0.3 in analysis:
                result = analysis[0.3]
                
                # Aggregate issues
                for issue in result['issues']:
                    dataset_issues[issue] += 1
                
                # Aggregate class statistics
                for obj in result['objects']:
                    class_name = obj['class_name']
                    class_detection_stats[class_name]['count'] += 1
                    class_detection_stats[class_name]['avg_confidence'] += obj['confidence']
                    class_detection_stats[class_name]['size_range'].append(obj['relative_size'])
        
        # Calculate averages
        for class_name in class_detection_stats:
            if class_detection_stats[class_name]['count'] > 0:
                class_detection_stats[class_name]['avg_confidence'] /= class_detection_stats[class_name]['count']
        
        # Print analysis results
        self._print_dataset_analysis(dataset_issues, class_detection_stats)
        
        return dataset_issues, class_detection_stats
    
    def _print_dataset_analysis(self, dataset_issues, class_detection_stats):
        """Print comprehensive dataset analysis."""
        print("\n" + "=" * 60)
        print("📊 DATASET ISSUE ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"\n⚠️ Common Issues Found:")
        if dataset_issues:
            for issue, count in sorted(dataset_issues.items(), key=lambda x: x[1], reverse=True):
                print(f"  {issue}: {count} occurrences")
        else:
            print("  No major issues detected!")
        
        print(f"\n📈 Class Detection Statistics:")
        for class_name, stats in class_detection_stats.items():
            if stats['count'] > 0:
                avg_size = np.mean(stats['size_range']) if stats['size_range'] else 0
                print(f"  {class_name}:")
                print(f"    - Count: {stats['count']}")
                print(f"    - Avg Confidence: {stats['avg_confidence']:.3f}")
                print(f"    - Avg Relative Size: {avg_size:.6f}")
        
        print(f"\n🎯 Recommendations:")
        if dataset_issues:
            print("  - Review images with missing labels")
            print("  - Check for systematic detection failures")
            print("  - Consider adjusting confidence thresholds")
            print("  - Improve training data quality")
        else:
            print("  - Dataset quality is excellent!")
            print("  - Ready for training")
        
        print("=" * 60)
    
    def generate_improvement_report(self, dataset_issues, class_detection_stats, output_dir):
        """Generate detailed improvement report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report = {
            'summary': {
                'total_issues': sum(dataset_issues.values()),
                'issue_categories': len(dataset_issues),
                'classes_analyzed': len(class_detection_stats)
            },
            'issues': dict(dataset_issues),
            'class_statistics': dict(class_detection_stats),
            'recommendations': self._generate_recommendations(dataset_issues, class_detection_stats)
        }
        
        # Save report
        report_file = output_path / "detection_improvement_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Improvement report saved to: {report_file}")
        return report_file
    
    def _generate_recommendations(self, dataset_issues, class_detection_stats):
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        # Issue-based recommendations
        if 'missing_labels' in dataset_issues:
            recommendations.append({
                'priority': 'high',
                'issue': 'Missing labels',
                'solution': 'Ensure all images have corresponding label files',
                'action': 'Run auto-labeling on images without labels'
            })
        
        if 'No objects detected' in dataset_issues:
            recommendations.append({
                'priority': 'high',
                'issue': 'No objects detected',
                'solution': 'Lower confidence threshold or improve model',
                'action': 'Reduce confidence threshold to 0.1-0.2'
            })
        
        # Class-based recommendations
        for class_name, stats in class_detection_stats.items():
            if stats['count'] > 0:
                if stats['avg_confidence'] < 0.5:
                    recommendations.append({
                        'priority': 'medium',
                        'issue': f'Low confidence for {class_name}',
                        'solution': 'Improve training data or lower threshold',
                        'action': f'Lower confidence threshold for {class_name} class'
                    })
        
        return recommendations

def main():
    """Main function for detection issue analysis."""
    parser = argparse.ArgumentParser(description="Analyze detection issues and provide improvements")
    parser.add_argument("--model", required=True, help="Path to trained YOLO12 model")
    parser.add_argument("--images-dir", help="Directory containing images for dataset analysis")
    parser.add_argument("--labels-dir", help="Directory containing labels for dataset analysis")
    parser.add_argument("--single-image", help="Path to single image for analysis")
    parser.add_argument("--output-dir", default="results/detection_analysis", help="Output directory for reports")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size for dataset analysis")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DetectionIssueAnalyzer(args.model)
    
    if args.single_image:
        # Analyze single image
        print("🔍 Single Image Analysis Mode")
        print("=" * 40)
        analysis = analyzer.analyze_single_image(args.single_image)
        
    elif args.images_dir and args.labels_dir:
        # Analyze dataset
        print("🔍 Dataset Analysis Mode")
        print("=" * 40)
        dataset_issues, class_stats = analyzer.analyze_dataset_issues(
            args.images_dir, args.labels_dir, args.sample_size
        )
        
        # Generate improvement report
        analyzer.generate_improvement_report(dataset_issues, class_stats, args.output_dir)
        
    else:
        print("❌ Please specify either --single-image or both --images-dir and --labels-dir")
        return

if __name__ == "__main__":
    main()
