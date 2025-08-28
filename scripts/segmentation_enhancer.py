#!/usr/bin/env python3
"""
Segmentation Enhancer with Data Extraction
Generates segmentation masks and extracts measurements from detected objects
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from ultralytics import YOLO
from collections import defaultdict
import time
from datetime import datetime
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt

class SegmentationEnhancer:
    def __init__(self, model_path):
        """Initialize the segmentation enhancer."""
        self.model_path = model_path
        self.model = YOLO(model_path)
        
        # Class names and colors for visualization
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Color palette for segmentation visualization
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        # Measurement reference objects (in pixels)
        self.reference_objects = {
            'ruler': 100,  # 100 pixels = 1 cm
            'big_ruler': 200,  # 200 pixels = 1 cm
            'color_chart': 50   # 50 pixels = 1 cm
        }
        
    def generate_segmentation_mask(self, image, bbox, class_id, confidence):
        """Generate segmentation mask from bounding box."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Create binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill bounding box area
        mask[int(y1):int(y2), int(x1):int(x2)] = 255
        
        # Apply morphological operations for better shape
        if class_id in [4, 11]:  # cucumber or slice - use elliptical shape
            # Create elliptical mask
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            a, b = (x2 - x1) / 2, (y2 - y1) / 2
            
            y_coords, x_coords = np.ogrid[:height, :width]
            ellipse_mask = ((x_coords - center_x) ** 2 / a ** 2 + 
                           (y_coords - center_y) ** 2 / b ** 2) <= 1
            mask = (ellipse_mask * 255).astype(np.uint8)
        
        # Apply smoothing for natural boundaries
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = (mask > 127).astype(np.uint8) * 255
        
        return mask
    
    def extract_measurements(self, mask, class_name, reference_pixels_per_cm):
        """Extract measurements from segmentation mask."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area_pixels = cv2.contourArea(largest_contour)
        perimeter_pixels = cv2.arcLength(largest_contour, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to real-world units (cm)
        area_cm2 = area_pixels / (reference_pixels_per_cm ** 2)
        perimeter_cm = perimeter_pixels / reference_pixels_per_cm
        width_cm = w / reference_pixels_per_cm
        height_cm = h / reference_pixels_per_cm
        
        # Shape analysis
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_pixels) / hull_area if hull_area > 0 else 0
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Circularity
        circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
        
        return {
            'area_pixels': int(area_pixels),
            'area_cm2': round(area_cm2, 2),
            'perimeter_pixels': int(perimeter_pixels),
            'perimeter_cm': round(perimeter_cm, 2),
            'width_pixels': int(w),
            'width_cm': round(width_cm, 2),
            'height_pixels': int(h),
            'height_cm': round(height_cm, 2),
            'solidity': round(solidity, 3),
            'aspect_ratio': round(aspect_ratio, 3),
            'circularity': round(circularity, 3),
            'centroid_x': int(x + w/2),
            'centroid_y': int(y + h/2)
        }
    
    def find_reference_scale(self, detections, image_shape):
        """Find reference scale from ruler or color chart."""
        height, width = image_shape[:2]
        
        # Look for reference objects
        for detection in detections:
            class_name = self.class_names[int(detection.cls)]
            if class_name in self.reference_objects:
                bbox = detection.xyxy[0]
                ref_width = bbox[2] - bbox[0]
                ref_height = bbox[3] - bbox[1]
                
                # Use the larger dimension for scale
                ref_size = max(ref_width, ref_height)
                expected_cm = self.reference_objects[class_name]
                
                return ref_size / expected_cm
        
        # Default scale if no reference found
        return 100  # 100 pixels = 1 cm
    
    def process_image(self, image_path, output_dir, save_masks=True, save_analysis=True):
        """Process single image with segmentation and measurements."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        
        # Run detection
        results = self.model(image, verbose=False)
        detections = results[0]
        
        # Find reference scale
        pixels_per_cm = self.find_reference_scale(detections.boxes, image.shape)
        
        # Initialize results
        image_results = {
            'image_path': str(image_path),
            'image_size': {'width': width, 'height': height},
            'scale_factor': pixels_per_cm,
            'detections': [],
            'measurements': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Create output directories
        masks_dir = output_dir / 'masks' if save_masks else None
        if save_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each detection
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Generate segmentation mask
            mask = self.generate_segmentation_mask(image, bbox, class_id, confidence)
            
            # Save mask if requested
            if save_masks:
                mask_filename = f"{image_path.stem}_mask_{i:03d}_{class_name}.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)
            
            # Extract measurements
            measurements = self.extract_measurements(mask, class_name, pixels_per_cm)
            
            # Store detection info
            detection_info = {
                'id': i,
                'class': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'bbox': bbox.tolist(),
                'mask_path': str(mask_path) if save_masks else None,
                'measurements': measurements
            }
            
            image_results['detections'].append(detection_info)
            
            # Store measurements by class
            if class_name not in image_results['measurements']:
                image_results['measurements'][class_name] = []
            image_results['measurements'][class_name].append(measurements)
        
        # Calculate processing time
        image_results['processing_time'] = time.time() - start_time
        
        return image_results
    
    def process_directory(self, input_dir, output_dir, save_masks=True, save_analysis=True):
        """Process all images in directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        if save_masks:
            (output_path / 'masks').mkdir(exist_ok=True)
        if save_analysis:
            (output_path / 'analysis').mkdir(exist_ok=True)
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"🔍 Found {len(image_files)} images to process")
        
        # Process each image
        all_results = []
        total_start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"🔄 Processing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.process_image(image_file, output_path, save_masks, save_analysis)
                if result:
                    all_results.append(result)
                    print(f"   ✅ Processed: {len(result['detections'])} objects")
                else:
                    print(f"   ❌ Failed to process")
            except Exception as e:
                print(f"   ❌ Error processing {image_file.name}: {e}")
        
        # Generate summary
        total_time = time.time() - total_start_time
        
        summary = {
            'total_images': len(image_files),
            'processed_images': len(all_results),
            'total_time': total_time,
            'average_time_per_image': total_time / len(image_files) if image_files else 0,
            'total_objects': sum(len(r['detections']) for r in all_results),
            'class_distribution': defaultdict(int),
            'measurement_summary': {}
        }
        
        # Analyze class distribution and measurements
        for result in all_results:
            for detection in result['detections']:
                class_name = detection['class']
                summary['class_distribution'][class_name] += 1
                
                # Aggregate measurements
                if class_name not in summary['measurement_summary']:
                    summary['measurement_summary'][class_name] = {
                        'count': 0,
                        'total_area': 0,
                        'total_perimeter': 0,
                        'areas': [],
                        'perimeters': []
                    }
                
                measurements = detection['measurements']
                if measurements:
                    summary['measurement_summary'][class_name]['count'] += 1
                    summary['measurement_summary'][class_name]['total_area'] += measurements.get('area_cm2', 0)
                    summary['measurement_summary'][class_name]['total_perimeter'] += measurements.get('perimeter_cm', 0)
                    summary['measurement_summary'][class_name]['areas'].append(measurements.get('area_cm2', 0))
                    summary['measurement_summary'][class_name]['perimeters'].append(measurements.get('perimeter_cm', 0))
        
        # Calculate averages
        for class_name, data in summary['measurement_summary'].items():
            if data['count'] > 0:
                data['average_area'] = round(data['total_area'] / data['count'], 2)
                data['average_perimeter'] = round(data['total_perimeter'] / data['count'], 2)
                data['min_area'] = round(min(data['areas']), 2) if data['areas'] else 0
                data['max_area'] = round(max(data['areas']), 2) if data['areas'] else 0
        
        # Save results
        if save_analysis:
            # Save detailed results
            with open(output_path / 'analysis' / 'detailed_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Save summary
            with open(output_path / 'analysis' / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Segmentation Enhancer with Data Extraction')
    parser.add_argument('--model', required=True, help='Path to YOLO model')
    parser.add_argument('--input-dir', required=True, help='Input directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--save-masks', action='store_true', help='Save segmentation masks')
    parser.add_argument('--save-analysis', action='store_true', help='Save detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = SegmentationEnhancer(args.model)
    
    # Process directory
    print("🚀 Starting Segmentation Enhancement...")
    summary = enhancer.process_directory(
        args.input_dir, 
        args.output_dir, 
        args.save_masks, 
        args.save_analysis
    )
    
    # Print summary
    print("\n📊 SEGMENTATION ENHANCEMENT SUMMARY")
    print("=" * 50)
    print(f"📸 Total images: {summary['total_images']}")
    print(f"✅ Processed: {summary['processed_images']}")
    print(f"⏱️ Total time: {summary['total_time']:.2f}s")
    print(f"⚡ Average time per image: {summary['average_time_per_image']:.2f}s")
    print(f"🔢 Total objects: {summary['total_objects']}")
    
    print(f"\n📊 Class Distribution:")
    for class_name, count in summary['class_distribution'].items():
        print(f"   {class_name}: {count}")
    
    print(f"\n📏 Measurement Summary:")
    for class_name, data in summary['measurement_summary'].items():
        if data['count'] > 0:
            print(f"   {class_name}:")
            print(f"     Count: {data['count']}")
            print(f"     Avg Area: {data['average_area']} cm²")
            print(f"     Avg Perimeter: {data['average_perimeter']} cm")
            print(f"     Area Range: {data['min_area']} - {data['max_area']} cm²")
    
    print(f"\n💾 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
