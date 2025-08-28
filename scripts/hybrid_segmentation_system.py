#!/usr/bin/env python3
"""
Hybrid Segmentation System
Combines perfect YOLO detection with existing SAM2 segmentation masks
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
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt

class HybridSegmentationSystem:
    def __init__(self, yolo_model_path, sam2_annotations_dir):
        """Initialize the hybrid segmentation system."""
        self.yolo_model = YOLO(yolo_model_path)
        self.sam2_annotations_dir = Path(sam2_annotations_dir)
        
        # Class names from your YOLO model
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Color palette for visualization
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        # Load SAM2 annotations
        self.sam2_annotations = self._load_sam2_annotations()
        
        # Measurement reference objects (in pixels)
        self.reference_objects = {
            'ruler': 100,  # 100 pixels = 1 cm
            'big_ruler': 200,  # 200 pixels = 1 cm
            'color_chart': 50   # 50 pixels = 1 cm
        }
        
    def _load_sam2_annotations(self):
        """Load all SAM2 annotations from the directory."""
        annotations = {}
        
        # Look for JSON files in train, valid, and test directories
        for split in ['train', 'valid', 'test']:
            split_dir = self.sam2_annotations_dir / split
            if split_dir.exists():
                for json_file in split_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            
                        # Extract image info
                        image_name = data['image']['file_name']
                        image_height = data['image']['height']
                        image_width = data['image']['width']
                        
                        # Store annotations
                        annotations[image_name] = {
                            'image_info': {
                                'height': image_height,
                                'width': image_width,
                                'file_name': image_name
                            },
                            'annotations': data['annotations']
                        }
                        
                    except Exception as e:
                        print(f"Warning: Could not load {json_file}: {e}")
        
        print(f"📚 Loaded {len(annotations)} SAM2 annotation files")
        return annotations
    
    def _find_matching_sam2_annotation(self, image_path):
        """Find matching SAM2 annotation for an image."""
        # Try to find exact match
        image_name = image_path.name
        
        if image_name in self.sam2_annotations:
            return self.sam2_annotations[image_name]
        
        # Try to find by base name (without Roboflow suffix)
        base_name = image_path.stem.split('.rf.')[0] if '.rf.' in image_path.stem else image_path.stem
        
        for key, value in self.sam2_annotations.items():
            if base_name in key:
                return value
        
        return None
    
    def _decode_rle_mask(self, rle_data, image_shape):
        """Decode RLE (Run-Length Encoding) mask to binary mask."""
        try:
            # Extract RLE components
            counts = rle_data['counts']
            size = rle_data['size']
            
            # Decode using pycocotools
            mask = mask_util.decode({
                'counts': counts,
                'size': size
            })
            
            return mask.astype(np.uint8) * 255
            
        except Exception as e:
            print(f"Warning: Could not decode RLE mask: {e}")
            return None
    
    def _find_best_matching_mask(self, yolo_bbox, sam2_annotations, image_shape):
        """Find the best matching SAM2 mask for a YOLO detection."""
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
        yolo_center_x = (yolo_x1 + yolo_x2) / 2
        yolo_center_y = (yolo_y1 + yolo_y2) / 2
        yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
        
        best_match = None
        best_iou = 0
        
        for annotation in sam2_annotations:
            sam2_bbox = annotation['bbox']
            sam2_x1, sam2_y1, sam2_w, sam2_h = sam2_bbox
            sam2_x2 = sam2_x1 + sam2_w
            sam2_y2 = sam2_y1 + sam2_h
            sam2_center_x = (sam2_x1 + sam2_x2) / 2
            sam2_center_y = (sam2_y1 + sam2_y2) / 2
            sam2_area = sam2_w * sam2_h
            
            # Calculate center distance
            center_distance = np.sqrt((yolo_center_x - sam2_center_x)**2 + 
                                    (yolo_center_y - sam2_center_y)**2)
            
            # Calculate area similarity
            area_ratio = min(yolo_area, sam2_area) / max(yolo_area, sam2_area)
            
            # Calculate IoU
            x1 = max(yolo_x1, sam2_x1)
            y1 = max(yolo_y1, sam2_y1)
            x2 = min(yolo_x2, sam2_x2)
            y2 = min(yolo_y2, sam2_y2)
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                union = yolo_area + sam2_area - intersection
                iou = intersection / union if union > 0 else 0
            else:
                iou = 0
            
            # Combined score (prioritize IoU, then center distance, then area ratio)
            score = iou * 0.6 + (1 - center_distance / 100) * 0.3 + area_ratio * 0.1
            
            if score > best_iou:
                best_iou = score
                best_match = annotation
        
        return best_match, best_iou
    
    def _extract_measurements_from_mask(self, mask, class_name, reference_pixels_per_cm):
        """Extract precise measurements from segmentation mask."""
        if mask is None:
            return {}
        
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
        
        # Centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x = int(x + w/2)
            centroid_y = int(y + h/2)
        
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
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'mask_quality': 'SAM2'  # Indicate this came from SAM2
        }
    
    def _find_reference_scale(self, detections, image_shape):
        """Find reference scale from ruler or color chart."""
        height, width = image_shape[:2]
        
        # Look for reference objects
        for detection in detections:
            class_name = self.class_names[int(detection.cls)]
            if class_name in self.reference_objects:
                bbox = detection.xyxy[0]
                ref_width = float(bbox[2] - bbox[0])
                ref_height = float(bbox[3] - bbox[1])
                
                # Use the larger dimension for scale
                ref_size = max(ref_width, ref_height)
                expected_cm = self.reference_objects[class_name]
                
                return ref_size / expected_cm
        
        # Default scale if no reference found
        return 100  # 100 pixels = 1 cm
    
    def process_image(self, image_path, output_dir, save_masks=True, save_analysis=True):
        """Process single image with hybrid YOLO + SAM2 approach."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        
        # Find matching SAM2 annotation
        sam2_data = self._find_matching_sam2_annotation(image_path)
        
        # Run YOLO detection
        results = self.yolo_model(image, verbose=False)
        detections = results[0]
        
        # Find reference scale
        pixels_per_cm = self._find_reference_scale(detections.boxes, image.shape)
        
        # Initialize results
        image_results = {
            'image_path': str(image_path),
            'image_size': {'width': width, 'height': height},
            'scale_factor': pixels_per_cm,
            'has_sam2_annotations': sam2_data is not None,
            'detections': [],
            'measurements': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Create output directories
        masks_dir = output_dir / 'masks' if save_masks else None
        if save_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each YOLO detection
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Try to find matching SAM2 mask
            sam2_mask = None
            mask_source = "Generated"
            mask_quality_score = 0
            
            if sam2_data:
                best_match, quality_score = self._find_best_matching_mask(
                    bbox, sam2_data['annotations'], image.shape
                )
                
                if best_match and quality_score > 0.3:  # Good match threshold
                    # Decode SAM2 mask
                    sam2_mask = self._decode_rle_mask(
                        best_match['segmentation'], 
                        image.shape
                    )
                    mask_source = "SAM2"
                    mask_quality_score = quality_score
            
            # If no good SAM2 match, generate mask from bbox
            if sam2_mask is None:
                sam2_mask = self._generate_bbox_mask(image, bbox, class_id)
                mask_source = "Generated"
                mask_quality_score = 0
            
            # Save mask if requested
            if save_masks:
                mask_filename = f"{image_path.stem}_mask_{i:03d}_{class_name}.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), sam2_mask)
            
            # Extract measurements
            measurements = self._extract_measurements_from_mask(
                sam2_mask, class_name, pixels_per_cm
            )
            
            # Store detection info
            detection_info = {
                'id': i,
                'class': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'bbox': bbox.tolist(),
                'mask_path': str(mask_path) if save_masks else None,
                'mask_source': mask_source,
                'mask_quality_score': mask_quality_score,
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
    
    def _generate_bbox_mask(self, image, bbox, class_id):
        """Generate basic mask from bounding box when SAM2 not available."""
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
                    sam2_count = sum(1 for d in result['detections'] if d['mask_source'] == 'SAM2')
                    print(f"   ✅ Processed: {len(result['detections'])} objects ({sam2_count} SAM2 masks)")
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
            'sam2_masks_used': sum(sum(1 for d in r['detections'] if d['mask_source'] == 'SAM2') for r in all_results),
            'generated_masks': sum(sum(1 for d in r['detections'] if d['mask_source'] == 'Generated') for r in all_results),
            'class_distribution': defaultdict(int),
            'measurement_summary': {},
            'mask_quality_stats': {
                'average_sam2_quality': 0,
                'total_sam2_masks': 0
            }
        }
        
        # Analyze class distribution and measurements
        total_sam2_quality = 0
        sam2_mask_count = 0
        
        for result in all_results:
            for detection in result['detections']:
                class_name = detection['class']
                summary['class_distribution'][class_name] += 1
                
                # Track mask quality
                if detection['mask_source'] == 'SAM2':
                    total_sam2_quality += detection['mask_quality_score']
                    sam2_mask_count += 1
                
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
        
        # Calculate mask quality statistics
        if sam2_mask_count > 0:
            summary['mask_quality_stats']['average_sam2_quality'] = round(total_sam2_quality / sam2_mask_count, 3)
            summary['mask_quality_stats']['total_sam2_masks'] = sam2_mask_count
        
        # Calculate measurement averages
        for class_name, data in summary['measurement_summary'].items():
            if data['count'] > 0:
                data['average_area'] = round(data['total_area'] / data['count'], 2)
                data['average_perimeter'] = round(data['total_perimeter'] / data['count'], 2)
                data['min_area'] = round(min(data['areas']), 2) if data['areas'] else 0
                data['max_area'] = round(max(data['areas']), 2) if data['areas'] else 0
        
        # Save results
        if save_analysis:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Convert all results
            serializable_results = convert_numpy_types(all_results)
            serializable_summary = convert_numpy_types(summary)
            
            # Save detailed results
            with open(output_path / 'analysis' / 'detailed_results.json', 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            # Save summary
            with open(output_path / 'analysis' / 'summary.json', 'w') as f:
                json.dump(serializable_summary, f, indent=2)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Hybrid YOLO + SAM2 Segmentation System')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--sam2-annotations', required=True, help='Path to SAM2 annotations directory')
    parser.add_argument('--input-dir', required=True, help='Input directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--save-masks', action='store_true', help='Save segmentation masks')
    parser.add_argument('--save-analysis', action='store_true', help='Save detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize hybrid system
    hybrid_system = HybridSegmentationSystem(args.yolo_model, args.sam2_annotations)
    
    # Process directory
    print("🚀 Starting Hybrid YOLO + SAM2 Segmentation...")
    summary = hybrid_system.process_directory(
        args.input_dir, 
        args.output_dir, 
        args.save_masks, 
        args.save_analysis
    )
    
    # Print summary
    print("\n📊 HYBRID SEGMENTATION SUMMARY")
    print("=" * 50)
    print(f"📸 Total images: {summary['total_images']}")
    print(f"✅ Processed: {summary['processed_images']}")
    print(f"⏱️ Total time: {summary['total_time']:.2f}s")
    print(f"⚡ Average time per image: {summary['average_time_per_image']:.2f}s")
    print(f"🔢 Total objects: {summary['total_objects']}")
    print(f"🎯 SAM2 masks used: {summary['sam2_masks_used']}")
    print(f"🔧 Generated masks: {summary['generated_masks']}")
    
    if summary['mask_quality_stats']['total_sam2_masks'] > 0:
        print(f"🏆 Average SAM2 quality: {summary['mask_quality_stats']['average_sam2_quality']}")
    
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
