#!/usr/bin/env python3
"""
Quick Detection Fix
Rapidly fix common detection issues with adjustable parameters
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from ultralytics import YOLO
import shutil
from collections import defaultdict

class QuickDetectionFix:
    def __init__(self, model_path):
        """Initialize quick detection fixer."""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        # Class names
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Default fix parameters
        self.fix_params = {
            'base_confidence': 0.15,  # Very low base confidence
            'class_thresholds': {
                'cucumber': 0.15,     # Very low for main objects
                'ruler': 0.2,         # Low for calibration
                'label': 0.25,        # Medium for text
                'color_chart': 0.2,   # Low for calibration
                'slice': 0.15,        # Very low for parts
                'cavity': 0.15,       # Very low for parts
                'hollow': 0.15,       # Very low for parts
                'big_ruler': 0.2,     # Low for calibration
                'blue_dot': 0.3,      # Medium for small objects
                'red_dot': 0.3,       # Medium for small objects
                'green_dot': 0.3,     # Medium for small objects
                'objects': 0.25       # Medium for others
            },
            'min_object_size': 0.0005,  # Minimum relative size
            'max_object_size': 0.8,     # Maximum relative size
            'max_overlap': 0.8,         # Maximum overlap allowed
            'enable_post_processing': True
        }
    
    def apply_quick_fix(self, input_dir, output_dir, fix_params=None):
        """Apply quick fixes to improve detection."""
        if fix_params:
            self.fix_params.update(fix_params)
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        output_path.mkdir(exist_ok=True)
        (output_path / "fixed_labels").mkdir(exist_ok=True)
        (output_path / "fixed_images").mkdir(exist_ok=True)
        (output_path / "fix_report").mkdir(exist_ok=True)
        
        print(f"🔧 Applying Quick Detection Fixes")
        print(f"📁 Input: {input_path}")
        print(f"📁 Output: {output_path}")
        print("=" * 50)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"📸 Found {len(image_files)} images to fix")
        
        fix_results = {
            'total_images': len(image_files),
            'fixed_images': 0,
            'total_detections_before': 0,
            'total_detections_after': 0,
            'issues_fixed': defaultdict(int),
            'class_improvements': defaultdict(int)
        }
        
        for i, image_file in enumerate(image_files):
            print(f"🔄 Fixing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Run detection with very low confidence
                results = self.model(str(image_file), conf=self.fix_params['base_confidence'])
                
                if not results or not results[0].boxes:
                    print(f"  ⚠️ No objects detected even at low confidence")
                    continue
                
                # Apply fixes
                fixed_detections = self._apply_detection_fixes(results, image_file)
                
                if fixed_detections:
                    # Save fixed labels
                    label_file = output_path / "fixed_labels" / f"{image_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        for detection in fixed_detections:
                            f.write(f"{detection['class_id']} {detection['x_center']:.6f} {detection['y_center']:.6f} {detection['width']:.6f} {detection['height']:.6f}\n")
                    
                    # Copy image
                    shutil.copy2(image_file, output_path / "fixed_images" / image_file.name)
                    
                    fix_results['fixed_images'] += 1
                    fix_results['total_detections_after'] += len(fixed_detections)
                    
                    print(f"  ✅ Fixed: {len(fixed_detections)} objects")
                else:
                    print(f"  ⚠️ No valid detections after fixes")
                    
            except Exception as e:
                print(f"  ❌ Error fixing {image_file.name}: {e}")
        
        # Generate fix report
        self._generate_fix_report(fix_results, output_path / "fix_report")
        
        return fix_results
    
    def _apply_detection_fixes(self, results, image_path):
        """Apply various detection fixes."""
        # Load image for size calculations
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        height, width = img.shape[:2]
        detections = []
        
        if not results or not results[0].boxes:
            return detections
        
        boxes = results[0].boxes
        
        for box in boxes:
            # Get detection info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            
            # Apply class-specific threshold
            threshold = self.fix_params['class_thresholds'].get(class_name, self.fix_params['base_confidence'])
            
            if confidence >= threshold:
                # Calculate object properties
                obj_width = x2 - x1
                obj_height = y2 - y1
                relative_size = (obj_width * obj_height) / (width * height)
                
                # Size filtering
                if (relative_size < self.fix_params['min_object_size'] or 
                    relative_size > self.fix_params['max_object_size']):
                    continue
                
                # Convert to YOLO format
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = obj_width / width
                h = obj_height / height
                
                # Validate coordinates
                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                    0 < w <= 1 and 0 < h <= 1):
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': w,
                        'height': h,
                        'relative_size': relative_size
                    }
                    
                    detections.append(detection)
        
        # Post-processing fixes
        if self.fix_params['enable_post_processing']:
            detections = self._apply_post_processing_fixes(detections, width, height)
        
        return detections
    
    def _apply_post_processing_fixes(self, detections, image_width, image_height):
        """Apply post-processing fixes to detections."""
        if len(detections) <= 1:
            return detections
        
        # Remove overlapping detections
        filtered_detections = []
        
        for i, det1 in enumerate(detections):
            keep_detection = True
            
            for j, det2 in enumerate(detections):
                if i == j:
                    continue
                
                # Calculate overlap
                overlap = self._calculate_overlap(
                    det1['x_center'], det1['y_center'], det1['width'], det1['height'],
                    det2['x_center'], det2['y_center'], det2['width'], det2['height']
                )
                
                if overlap > self.fix_params['max_overlap']:
                    # Keep the one with higher confidence
                    if det1['confidence'] < det2['confidence']:
                        keep_detection = False
                        break
            
            if keep_detection:
                filtered_detections.append(det1)
        
        return filtered_detections
    
    def _calculate_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Calculate overlap between two normalized bounding boxes."""
        # Convert to absolute coordinates
        x1_abs = (x1 - w1/2) * image_width
        y1_abs = (y1 - h1/2) * image_height
        x2_abs = (x1 + w1/2) * image_width
        y2_abs = (y1 + h1/2) * image_height
        
        x3_abs = (x2 - w2/2) * image_width
        y3_abs = (y2 - h2/2) * image_height
        x4_abs = (x2 + w2/2) * image_width
        y4_abs = (y2 + h2/2) * image_height
        
        # Calculate intersection
        x1_i = max(x1_abs, x3_abs)
        y1_i = max(y1_abs, y3_abs)
        x2_i = min(x2_abs, x4_abs)
        y2_i = min(y2_abs, y4_abs)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1 * image_width * image_height
        area2 = w2 * h2 * image_width * image_height
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _generate_fix_report(self, fix_results, report_dir):
        """Generate fix report."""
        report = {
            'fix_summary': fix_results,
            'fix_parameters': self.fix_params,
            'recommendations': self._generate_fix_recommendations(fix_results)
        }
        
        # Save report
        report_file = report_dir / "fix_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 50)
        print("🔧 QUICK FIX COMPLETE!")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"  Images processed: {fix_results['total_images']}")
        print(f"  Images fixed: {fix_results['fixed_images']}")
        print(f"  Total detections: {fix_results['total_detections_after']}")
        print(f"  Fix rate: {(fix_results['fixed_images']/fix_results['total_images'])*100:.1f}%")
        print(f"\n📁 Fixed data saved to:")
        print(f"  Labels: {report_dir.parent}/fixed_labels/")
        print(f"  Images: {report_dir.parent}/fixed_images/")
        print(f"  Report: {report_dir}/")
        print("=" * 50)
    
    def _generate_fix_recommendations(self, fix_results):
        """Generate recommendations based on fix results."""
        recommendations = []
        
        fix_rate = fix_results['fixed_images'] / fix_results['total_images']
        
        if fix_rate < 0.5:
            recommendations.append({
                'priority': 'high',
                'issue': 'Low fix rate',
                'solution': 'Lower confidence thresholds further',
                'action': 'Reduce base_confidence to 0.1'
            })
        
        if fix_results['total_detections_after'] < fix_results['total_images'] * 3:
            recommendations.append({
                'priority': 'medium',
                'issue': 'Few detections per image',
                'solution': 'Improve model sensitivity',
                'action': 'Lower class-specific thresholds'
            })
        
        return recommendations

def main():
    """Main function for quick detection fix."""
    parser = argparse.ArgumentParser(description="Quick detection fix for common issues")
    parser.add_argument("--model", required=True, help="Path to trained YOLO12 model")
    parser.add_argument("--input-dir", required=True, help="Directory containing images")
    parser.add_argument("--output-dir", required=True, help="Output directory for fixed data")
    parser.add_argument("--base-confidence", type=float, default=0.15, help="Base confidence threshold")
    parser.add_argument("--min-size", type=float, default=0.0005, help="Minimum relative object size")
    parser.add_argument("--max-size", type=float, default=0.8, help="Maximum relative object size")
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = QuickDetectionFix(args.model)
    
    # Apply fixes
    fix_params = {
        'base_confidence': args.base_confidence,
        'min_object_size': args.min_size,
        'max_object_size': args.max_size
    }
    
    fix_results = fixer.apply_quick_fix(args.input_dir, args.output_dir, fix_params)

if __name__ == "__main__":
    main()
