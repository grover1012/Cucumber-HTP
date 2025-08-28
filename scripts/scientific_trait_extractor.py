#!/usr/bin/env python3
"""
Scientific Cucumber Trait Extractor
Based on research paper specifications for comprehensive cucumber analysis
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
from scipy import ndimage
from skimage import measure, morphology, color
import pandas as pd

class ScientificTraitExtractor:
    def __init__(self, yolo_model_path, sam2_annotations_dir):
        """Initialize the scientific trait extractor."""
        self.yolo_model = YOLO(yolo_model_path)
        self.sam2_annotations_dir = Path(sam2_annotations_dir)
        
        # Class names from your YOLO model
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Load SAM2 annotations
        self.sam2_annotations = self._load_sam2_annotations()
        
        # Measurement reference objects (in pixels)
        self.reference_objects = {
            'ruler': 100,  # 100 pixels = 1 cm
            'big_ruler': 200,  # 200 pixels = 1 cm
            'color_chart': 50   # 50 pixels = 1 cm
        }
        
        # Netting scale reference (1-4)
        self.netting_scale = {
            1: "Smooth",
            2: "Light netting", 
            3: "Moderate netting",
            4: "Deep netting"
        }
        
    def _load_sam2_annotations(self):
        """Load all SAM2 annotations from the directory."""
        annotations = {}
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.sam2_annotations_dir / split
            if split_dir.exists():
                for json_file in split_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            
                        image_name = data['image']['file_name']
                        image_height = data['image']['height']
                        image_width = data['image']['width']
                        
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
        image_name = image_path.name
        
        if image_name in self.sam2_annotations:
            return self.sam2_annotations[image_name]
        
        base_name = image_path.stem.split('.rf.')[0] if '.rf.' in image_path.stem else image_path.stem
        
        for key, value in self.sam2_annotations.items():
            if base_name in key:
                return value
        
        return None
    
    def _decode_rle_mask(self, rle_data, image_shape):
        """Decode RLE mask to binary mask."""
        try:
            counts = rle_data['counts']
            size = rle_data['size']
            
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
            
            # Combined score
            score = iou * 0.6 + (1 - center_distance / 100) * 0.3 + area_ratio * 0.1
            
            if score > best_iou:
                best_iou = score
                best_match = annotation
        
        return best_match, best_iou
    
    def _extract_fruit_length(self, mask, pixels_per_cm):
        """Extract fruit length following the curve (Supplementary Fig. S5A)."""
        if mask is None:
            return {}
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate curved length (following the shape)
        curved_length_pixels = cv2.arcLength(largest_contour, False)
        
        # Calculate straight length (end to end)
        x, y, w, h = cv2.boundingRect(largest_contour)
        straight_length_pixels = np.sqrt(w**2 + h**2)
        
        # Convert to cm
        curved_length_cm = curved_length_pixels / pixels_per_cm
        straight_length_cm = straight_length_pixels / pixels_per_cm
        
        return {
            'curved_length_pixels': int(curved_length_pixels),
            'curved_length_cm': round(curved_length_cm, 2),
            'straight_length_pixels': int(straight_length_pixels),
            'straight_length_cm': round(straight_length_cm, 2)
        }
    
    def _extract_diameter_and_internal_traits(self, mask, pixels_per_cm):
        """Extract diameter and internal traits (Supplementary Fig. S5B)."""
        if mask is None:
            return {}
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate diameter (use the smaller dimension for cross-section)
        diameter_pixels = min(w, h)
        diameter_cm = diameter_pixels / pixels_per_cm
        
        # Estimate internal structures (simplified approach)
        # In real implementation, you'd need cross-section images
        estimated_seed_cavity_pixels = int(diameter_pixels * 0.3)  # 30% of diameter
        estimated_flesh_thickness_pixels = int(diameter_pixels * 0.2)  # 20% of diameter
        
        # Calculate areas
        total_area_pixels = cv2.contourArea(largest_contour)
        estimated_seed_cavity_area_pixels = np.pi * (estimated_seed_cavity_pixels / 2)**2
        estimated_flesh_area_pixels = total_area_pixels - estimated_seed_cavity_area_pixels
        
        # Estimate hollowness (percentage of void area)
        hollowness_ratio = estimated_seed_cavity_area_pixels / total_area_pixels if total_area_pixels > 0 else 0
        
        return {
            'diameter_pixels': int(diameter_pixels),
            'diameter_cm': round(diameter_cm, 2),
            'estimated_seed_cavity_pixels': estimated_seed_cavity_pixels,
            'estimated_flesh_thickness_pixels': estimated_flesh_thickness_pixels,
            'total_area_pixels': int(total_area_pixels),
            'seed_cavity_area_pixels': int(estimated_seed_cavity_area_pixels),
            'flesh_area_pixels': int(estimated_flesh_area_pixels),
            'hollowness_ratio': round(hollowness_ratio, 3),
            'hollowness_percentage': round(hollowness_ratio * 100, 1)
        }
    
    def _extract_curvature(self, mask, pixels_per_cm):
        """Extract curvature measurement (Supplementary Fig. S5C)."""
        if mask is None:
            return {}
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate center of fruit
        center_x = x + w/2
        center_y = y + h/2
        
        # Calculate diameter at center (d)
        diameter_at_center_pixels = min(w, h)
        diameter_at_center_cm = diameter_at_center_pixels / pixels_per_cm
        
        # Calculate distance from center to straight line connecting ends (d')
        # This is a simplified approximation
        distance_to_straight_line_pixels = abs(center_x - (x + w/2)) + abs(center_y - (y + h/2))
        distance_to_straight_line_cm = distance_to_straight_line_pixels / pixels_per_cm
        
        # Calculate curvature ratio d/d'
        curvature_ratio = diameter_at_center_pixels / distance_to_straight_line_pixels if distance_to_straight_line_pixels > 0 else 0
        
        return {
            'diameter_at_center_pixels': int(diameter_at_center_pixels),
            'diameter_at_center_cm': round(diameter_at_center_cm, 2),
            'distance_to_straight_line_pixels': int(distance_to_straight_line_pixels),
            'distance_to_straight_line_cm': round(distance_to_straight_line_cm, 2),
            'curvature_ratio': round(curvature_ratio, 3)
        }
    
    def _extract_tapering(self, mask, pixels_per_cm):
        """Extract tapering measurement (Supplementary Fig. S5D)."""
        if mask is None:
            return {}
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate widths at different sections
        # First quarter (q1) - top section
        q1_y = y + h * 0.25
        q1_width_pixels = self._get_width_at_height(mask, q1_y)
        
        # Third quarter (q3) - bottom section  
        q3_y = y + h * 0.75
        q3_width_pixels = self._get_width_at_height(mask, q3_y)
        
        # Calculate tapering ratio q1/q3
        tapering_ratio = q1_width_pixels / q3_width_pixels if q3_width_pixels > 0 else 0
        
        return {
            'q1_width_pixels': int(q1_width_pixels),
            'q1_width_cm': round(q1_width_pixels / pixels_per_cm, 2),
            'q3_width_pixels': int(q3_width_pixels),
            'q3_width_cm': round(q3_width_pixels / pixels_per_cm, 2),
            'tapering_ratio': round(tapering_ratio, 3)
        }
    
    def _get_width_at_height(self, mask, y_coord):
        """Get width of fruit at specific height."""
        height, width = mask.shape
        
        if y_coord < 0 or y_coord >= height:
            return 0
        
        # Get horizontal line at y_coord
        line = mask[int(y_coord), :]
        
        # Find left and right boundaries
        left_bound = 0
        right_bound = width - 1
        
        # Find left boundary
        for i in range(width):
            if line[i] > 0:
                left_bound = i
                break
        
        # Find right boundary
        for i in range(width-1, -1, -1):
            if line[i] > 0:
                right_bound = i
                break
        
        return right_bound - left_bound
    
    def _extract_color_analysis(self, image, mask, pixels_per_cm):
        """Extract color analysis at different sections."""
        if mask is None or image is None:
            return {}
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get fruit region
        fruit_region = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        
        # Define sections for color analysis
        height, width = mask.shape
        
        # Top section (first quarter)
        top_y = height * 0.25
        top_colors = self._get_section_colors(fruit_region, mask, top_y, 10)
        
        # Middle section
        middle_y = height * 0.5
        middle_colors = self._get_section_colors(fruit_region, mask, middle_y, 10)
        
        # Bottom section (third quarter)
        bottom_y = height * 0.75
        bottom_colors = self._get_section_colors(fruit_region, mask, bottom_y, 10)
        
        return {
            'top_section_rgb': top_colors,
            'middle_section_rgb': middle_colors,
            'bottom_section_rgb': bottom_colors,
            'overall_average_rgb': self._calculate_average_color(fruit_region, mask)
        }
    
    def _get_section_colors(self, image, mask, y_coord, thickness=5):
        """Get average RGB colors for a horizontal section."""
        height, width = image.shape[:2]
        
        if y_coord < 0 or y_coord >= height:
            return {'R': 0, 'G': 0, 'B': 0}
        
        # Define section boundaries
        y_start = max(0, int(y_coord - thickness/2))
        y_end = min(height, int(y_coord + thickness/2))
        
        section_colors = []
        
        for y in range(y_start, y_end):
            for x in range(width):
                if mask[y, x] > 0:
                    section_colors.append(image[y, x])
        
        if not section_colors:
            return {'R': 0, 'G': 0, 'B': 0}
        
        # Calculate average
        avg_color = np.mean(section_colors, axis=0)
        
        return {
            'R': int(avg_color[0]),
            'G': int(avg_color[1]), 
            'B': int(avg_color[2])
        }
    
    def _calculate_average_color(self, image, mask):
        """Calculate overall average RGB color of fruit."""
        # Get all non-zero pixels
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return {'R': 0, 'G': 0, 'B': 0}
        
        # Extract colors
        colors = image[coords[0], coords[1]]
        
        # Calculate average
        avg_color = np.mean(colors, axis=0)
        
        return {
            'R': int(avg_color[0]),
            'G': int(avg_color[1]),
            'B': int(avg_color[2])
        }
    
    def _extract_netting_score(self, image, mask):
        """Extract netting score based on surface texture analysis."""
        if mask is None or image is None:
            return {'netting_score': 1, 'netting_description': 'Smooth'}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calculate texture metrics
        # Standard deviation of pixel values (higher = more texture)
        std_dev = np.std(masked_gray[masked_gray > 0])
        
        # Edge density (higher = more netting)
        edges = cv2.Canny(masked_gray, 50, 150)
        edge_density = np.sum(edges > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        
        # Determine netting score based on texture metrics
        if std_dev < 20 and edge_density < 0.01:
            score = 1  # Smooth
        elif std_dev < 35 and edge_density < 0.02:
            score = 2  # Light netting
        elif std_dev < 50 and edge_density < 0.03:
            score = 3  # Moderate netting
        else:
            score = 4  # Deep netting
        
        return {
            'netting_score': score,
            'netting_description': self.netting_scale[score],
            'texture_std_dev': round(std_dev, 2),
            'edge_density': round(edge_density, 4)
        }
    
    def _extract_spine_density(self, image, mask, pixels_per_cm):
        """Extract spine density measurement."""
        if mask is None or image is None:
            return {'spine_count': 0, 'spine_density': 0}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Simple spine detection (this is a simplified approach)
        # In practice, you'd need more sophisticated spine detection
        # For now, we'll estimate based on edge density
        
        # Apply edge detection
        edges = cv2.Canny(masked_gray, 30, 100)
        
        # Count potential spines (high-contrast edges)
        spine_candidates = np.sum(edges > 0)
        
        # Calculate fruit area
        fruit_area_pixels = np.sum(mask > 0)
        fruit_area_cm2 = fruit_area_pixels / (pixels_per_cm ** 2)
        
        # Estimate spine density
        estimated_spine_count = int(spine_candidates * 0.1)  # Rough estimate
        spine_density = estimated_spine_count / fruit_area_cm2 if fruit_area_cm2 > 0 else 0
        
        return {
            'spine_count': estimated_spine_count,
            'spine_density_per_cm2': round(spine_density, 2),
            'fruit_area_cm2': round(fruit_area_cm2, 2)
        }
    
    def _find_reference_scale(self, detections, image_shape):
        """Find reference scale from ruler or color chart."""
        height, width = image_shape[:2]
        
        for detection in detections:
            class_name = self.class_names[int(detection.cls)]
            if class_name in self.reference_objects:
                bbox = detection.xyxy[0]
                ref_width = float(bbox[2] - bbox[0])
                ref_height = float(bbox[3] - bbox[1])
                
                ref_size = max(ref_width, ref_height)
                expected_cm = self.reference_objects[class_name]
                
                return ref_size / expected_cm
        
        return 100  # Default: 100 pixels = 1 cm
    
    def process_image(self, image_path, output_dir):
        """Process single image and extract all scientific traits."""
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
            'fruits': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Process each cucumber detection
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Only process cucumbers and slices
            if class_name not in ['cucumber', 'slice']:
                continue
            
            # Try to find matching SAM2 mask
            sam2_mask = None
            mask_source = "Generated"
            mask_quality_score = 0
            
            if sam2_data:
                best_match, quality_score = self._find_best_matching_mask(
                    bbox, sam2_data['annotations'], image.shape
                )
                
                if best_match and quality_score > 0.3:
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
            
            # Extract all scientific traits
            fruit_traits = {
                'id': i,
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox.tolist(),
                'mask_source': mask_source,
                'mask_quality_score': mask_quality_score,
                
                # Physical measurements
                'length': self._extract_fruit_length(sam2_mask, pixels_per_cm),
                'diameter_internal': self._extract_diameter_and_internal_traits(sam2_mask, pixels_per_cm),
                
                # Shape analysis
                'curvature': self._extract_curvature(sam2_mask, pixels_per_cm),
                'tapering': self._extract_tapering(sam2_mask, pixels_per_cm),
                
                # Visual traits
                'color_analysis': self._extract_color_analysis(image, sam2_mask, pixels_per_cm),
                'netting': self._extract_netting_score(image, sam2_mask),
                'spine_density': self._extract_spine_density(image, sam2_mask, pixels_per_cm)
            }
            
            # Calculate derived traits
            if fruit_traits['length'] and fruit_traits['diameter_internal']:
                length_cm = fruit_traits['length']['curved_length_cm']
                diameter_cm = fruit_traits['diameter_internal']['diameter_cm']
                
                if diameter_cm > 0:
                    fruit_traits['derived_traits'] = {
                        'fruit_shape_index': round(length_cm / diameter_cm, 3),
                        'length_diameter_ratio': round(length_cm / diameter_cm, 3)
                    }
                else:
                    fruit_traits['derived_traits'] = {'fruit_shape_index': 0, 'length_diameter_ratio': 0}
            
            image_results['fruits'].append(fruit_traits)
        
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
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in directory and extract scientific traits."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'analysis').mkdir(exist_ok=True)
        (output_path / 'csv_reports').mkdir(exist_ok=True)
        
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
                result = self.process_image(image_file, output_path)
                if result:
                    all_results.append(result)
                    cucumber_count = len([f for f in result['fruits'] if f['class'] in ['cucumber', 'slice']])
                    print(f"   ✅ Processed: {cucumber_count} cucumbers/slices")
                else:
                    print(f"   ❌ Failed to process")
            except Exception as e:
                print(f"   ❌ Error processing {image_file.name}: {e}")
        
        # Generate comprehensive reports
        self._generate_reports(all_results, output_path)
        
        return all_results
    
    def _generate_reports(self, all_results, output_path):
        """Generate comprehensive CSV reports and analysis."""
        # Convert to pandas DataFrame for easy analysis
        all_fruits = []
        
        for result in all_results:
            for fruit in result['fruits']:
                fruit_data = {
                    'image_path': result['image_path'],
                    'class': fruit['class'],
                    'confidence': fruit['confidence'],
                    'mask_source': fruit['mask_source'],
                    'mask_quality_score': fruit['mask_quality_score']
                }
                
                # Add length data
                if fruit['length']:
                    fruit_data.update({
                        'curved_length_cm': fruit['length'].get('curved_length_cm', 0),
                        'straight_length_cm': fruit['length'].get('straight_length_cm', 0)
                    })
                
                # Add diameter data
                if fruit['diameter_internal']:
                    fruit_data.update({
                        'diameter_cm': fruit['diameter_internal'].get('diameter_cm', 0),
                        'hollowness_percentage': fruit['diameter_internal'].get('hollowness_percentage', 0)
                    })
                
                # Add shape data
                if fruit['curvature']:
                    fruit_data.update({
                        'curvature_ratio': fruit['curvature'].get('curvature_ratio', 0)
                    })
                
                if fruit['tapering']:
                    fruit_data.update({
                        'tapering_ratio': fruit['tapering'].get('tapering_ratio', 0)
                    })
                
                # Add derived traits
                if fruit.get('derived_traits'):
                    fruit_data.update({
                        'fruit_shape_index': fruit['derived_traits'].get('fruit_shape_index', 0)
                    })
                
                # Add color data
                if fruit['color_analysis']:
                    overall_color = fruit['color_analysis'].get('overall_average_rgb', {})
                    fruit_data.update({
                        'avg_red': overall_color.get('R', 0),
                        'avg_green': overall_color.get('G', 0),
                        'avg_blue': overall_color.get('B', 0)
                    })
                
                # Add netting data
                if fruit['netting']:
                    fruit_data.update({
                        'netting_score': fruit['netting'].get('netting_score', 1),
                        'netting_description': fruit['netting'].get('netting_description', 'Smooth')
                    })
                
                # Add spine data
                if fruit['spine_density']:
                    fruit_data.update({
                        'spine_count': fruit['spine_density'].get('spine_count', 0),
                        'spine_density_per_cm2': fruit['spine_density'].get('spine_density_per_cm2', 0)
                    })
                
                all_fruits.append(fruit_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_fruits)
        
        # Save comprehensive CSV report
        csv_path = output_path / 'csv_reports' / 'comprehensive_cucumber_traits.csv'
        df.to_csv(csv_path, index=False)
        
        # Save detailed JSON results
        json_path = output_path / 'analysis' / 'detailed_trait_results.json'
        
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
        
        serializable_results = convert_numpy_types(all_results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(df)
        summary_path = output_path / 'analysis' / 'trait_summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n📊 Generated Reports:")
        print(f"   📋 Comprehensive CSV: {csv_path}")
        print(f"   📊 Detailed JSON: {json_path}")
        print(f"   📈 Summary Statistics: {summary_path}")
    
    def _generate_summary_statistics(self, df):
        """Generate summary statistics for all traits."""
        summary = {
            'total_fruits': len(df),
            'class_distribution': df['class'].value_counts().to_dict(),
            'trait_statistics': {}
        }
        
        # Numeric columns for statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['confidence', 'mask_quality_score']:
                continue  # Skip non-trait columns
            
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary['trait_statistics'][col] = {
                    'count': int(len(col_data)),
                    'mean': round(float(col_data.mean()), 3),
                    'std': round(float(col_data.std()), 3),
                    'min': round(float(col_data.min()), 3),
                    'max': round(float(col_data.max()), 3),
                    'median': round(float(col_data.median()), 3)
                }
        
        # Categorical statistics
        if 'netting_description' in df.columns:
            summary['netting_distribution'] = df['netting_description'].value_counts().to_dict()
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Scientific Cucumber Trait Extractor')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--sam2-annotations', required=True, help='Path to SAM2 annotations directory')
    parser.add_argument('--input-dir', required=True, help='Input directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize trait extractor
    extractor = ScientificTraitExtractor(args.yolo_model, args.sam2_annotations)
    
    # Process directory
    print("🚀 Starting Scientific Cucumber Trait Extraction...")
    print("📊 Extracting: Length, Diameter, Shape Index, Curvature, Tapering, Color, Netting, Spine Density")
    
    all_results = extractor.process_directory(args.input_dir, args.output_dir)
    
    # Print summary
    total_fruits = sum(len(r['fruits']) for r in all_results)
    print(f"\n🎉 SCIENTIFIC TRAIT EXTRACTION COMPLETE!")
    print(f"📸 Total images processed: {len(all_results)}")
    print(f"🥒 Total cucumbers/slices analyzed: {total_fruits}")
    print(f"💾 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
