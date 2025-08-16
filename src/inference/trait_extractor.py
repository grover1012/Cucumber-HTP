"""
Main trait extraction pipeline for cucumber analysis.
Integrates YOLO detection, calibration, and trait measurement.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from pathlib import Path

# Import utility modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import load_image, save_image, create_visualization
from utils.trait_extraction import extract_all_traits, validate_measurements
from utils.calibration import (
    detect_ruler_markings, calibrate_with_known_object, detect_color_chart,
    normalize_colors, calculate_illumination_correction, apply_illumination_correction,
    create_calibration_report
)
from utils.ocr_utils import (
    extract_accession_id, validate_accession_id, correct_common_ocr_errors,
    create_ocr_report
)


class CucumberTraitExtractor:
    """
    Main class for extracting phenotypic traits from cucumber images.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the trait extractor.
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        # Updated to match your 12-class dataset
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
        
        # Load YOLO model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image using YOLO.
        
        Args:
            image: Input image
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # YOLO12 is a detection model, so we don't have masks
                    # We'll use bounding boxes for trait extraction
                    mask = None
                    
                    # Safely get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'mask': mask
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def calibrate_measurements(self, image: np.ndarray, 
                              ruler_detection: Dict,
                              color_chart_detection: Dict) -> Dict:
        """
        Calibrate measurements using ruler and color chart.
        
        Args:
            image: Input image
            ruler_detection: Ruler detection result
            color_chart_detection: Color chart detection result
            
        Returns:
            Calibration results
        """
        calibration_results = {}
        
        # Ruler calibration
        if ruler_detection and 'mask' in ruler_detection:
            ruler_mask = ruler_detection['mask']
            
            # Try automatic ruler marking detection
            ruler_cal = detect_ruler_markings(ruler_mask, image)
            
            # If automatic detection fails, use known ruler length
            if ruler_cal['confidence'] < 0.5:
                # Assume standard 15cm ruler
                ruler_cal = calibrate_with_known_object(ruler_mask, image, 150.0)
            
            calibration_results['ruler'] = ruler_cal
        else:
            calibration_results['ruler'] = {'pixel_to_mm_ratio': 1.0, 'confidence': 0.0}
        
        # Color chart calibration
        if color_chart_detection and 'mask' in color_chart_detection:
            color_chart_mask = color_chart_detection['mask']
            color_cal = detect_color_chart(color_chart_mask, image)
            calibration_results['color_chart'] = color_cal
        else:
            calibration_results['color_chart'] = {'reference_colors': np.array([]), 'confidence': 0.0}
        
        # Illumination correction
        if color_chart_detection and 'mask' in color_chart_detection:
            illumination_cal = calculate_illumination_correction(image, color_chart_detection['mask'])
            calibration_results['illumination'] = illumination_cal
        else:
            calibration_results['illumination'] = {'correction_factor': 1.0}
        
        return calibration_results
    
    def extract_cucumber_traits(self, image: np.ndarray,
                               cucumber_detection: Dict,
                               calibration: Dict) -> Dict:
        """
        Extract phenotypic traits from cucumber detection using bounding box.
        
        Args:
            image: Input image
            cucumber_detection: Cucumber detection result
            calibration: Calibration results
            
        Returns:
            Dictionary of extracted traits
        """
        if not cucumber_detection or 'bbox' not in cucumber_detection:
            return {}
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = cucumber_detection['bbox']
        pixel_to_mm_ratio = calibration.get('ruler', {}).get('pixel_to_mm_ratio', 1.0)
        
        # Create a simple mask from bounding box for trait extraction
        # This is a simplified approach - in production you might want more sophisticated methods
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 255
        
        # Extract traits using the bounding box mask
        traits = extract_all_traits(mask, image, pixel_to_mm_ratio)
        
        # Add bounding box based measurements
        width_px = x2 - x1
        height_px = y2 - y1
        length_px = max(width_px, height_px)
        width_px = min(width_px, height_px)
        
        # Convert to real units
        length_mm = length_px * pixel_to_mm_ratio
        width_mm = width_px * pixel_to_mm_ratio
        aspect_ratio = length_mm / width_mm if width_mm > 0 else 0
        
        # Update traits with bounding box measurements
        traits.update({
            'length_mm': length_mm,
            'width_mm': width_mm,
            'aspect_ratio': aspect_ratio,
            'length_px': length_px,
            'width_px': width_px,
            'bbox_area_px': width_px * height_px,
            'bbox_area_mm2': (width_px * height_px) * (pixel_to_mm_ratio ** 2)
        })
        
        # Validate measurements
        validation = validate_measurements(traits)
        traits['validation'] = validation
        
        return traits
    
    def extract_accession_id(self, image: np.ndarray,
                            label_detection: Dict) -> Dict:
        """
        Extract accession ID from label detection.
        
        Args:
            image: Input image
            label_detection: Label detection result
            
        Returns:
            Dictionary with accession ID information
        """
        if not label_detection or 'mask' not in label_detection:
            return {'accession_id': '', 'confidence': 0.0, 'is_valid': False}
        
        label_mask = label_detection['mask']
        
        # Extract accession ID
        extraction_result = extract_accession_id(image, label_mask)
        
        # Validate the extracted ID
        if extraction_result['accession_id']:
            validation = validate_accession_id(extraction_result['accession_id'])
            extraction_result['validation'] = validation
            
            # Correct common OCR errors
            corrected_id = correct_common_ocr_errors(extraction_result['accession_id'])
            if corrected_id != extraction_result['accession_id']:
                extraction_result['corrected_id'] = corrected_id
                extraction_result['was_corrected'] = True
        
        return extraction_result
    
    def process_image(self, image_path: str, 
                     output_dir: str = None) -> Dict:
        """
        Process a single image and extract all traits.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            
        Returns:
            Dictionary with all extracted information
        """
        # Load image
        image = load_image(image_path)
        
        # Detect objects
        detections = self.detect_objects(image)
        
        # Separate detections by class (handle both old and new class names)
        cucumber_detections = [d for d in detections if d['class_name'] in ['cucumber', '4']]
        ruler_detections = [d for d in detections if d['class_name'] in ['ruler', '10', 'big_ruler', '0']]
        label_detections = [d for d in detections if d['class_name'] in ['label', '7']]
        color_chart_detections = [d for d in detections if d['class_name'] in ['color_chart', '3']]
        
        # Get best detection for each class (highest confidence)
        best_cucumber = max(cucumber_detections, key=lambda x: x['confidence']) if cucumber_detections else None
        best_ruler = max(ruler_detections, key=lambda x: x['confidence']) if ruler_detections else None
        best_label = max(label_detections, key=lambda x: x['confidence']) if label_detections else None
        best_color_chart = max(color_chart_detections, key=lambda x: x['confidence']) if color_chart_detections else None
        
        # Calibrate measurements
        calibration = self.calibrate_measurements(image, best_ruler, best_color_chart)
        
        # Extract cucumber traits
        cucumber_traits = {}
        if best_cucumber:
            cucumber_traits = self.extract_cucumber_traits(image, best_cucumber, calibration)
        
        # Extract accession ID
        accession_info = {}
        if best_label:
            accession_info = self.extract_accession_id(image, best_label)
        
        # Apply color normalization if color chart is available
        normalized_image = image.copy()
        if best_color_chart and calibration['color_chart']['confidence'] > 0.5:
            reference_colors = calibration['color_chart']['reference_colors']
            if len(reference_colors) > 0:
                normalized_image = normalize_colors(image, reference_colors)
        
        # Apply illumination correction
        if calibration['illumination']['correction_factor'] != 1.0:
            normalized_image = apply_illumination_correction(normalized_image, 
                                                          calibration['illumination']['correction_factor'])
        
        # Create results dictionary
        results = {
            'image_path': image_path,
            'detections': {
                'cucumber': best_cucumber,
                'ruler': best_ruler,
                'label': best_label,
                'color_chart': best_color_chart
            },
            'calibration': calibration,
            'cucumber_traits': cucumber_traits,
            'accession_id': accession_info,
            'processing_info': {
                'total_detections': len(detections),
                'confidence_threshold': self.confidence_threshold,
                'model_path': self.model_path
            }
        }
        
        # Save results if output directory is specified
        if output_dir:
            self._save_results(results, image_path, output_dir, normalized_image)
        
        return results
    
    def _save_results(self, results: Dict, image_path: str, 
                     output_dir: str, normalized_image: np.ndarray):
        """
        Save processing results and outputs.
        
        Args:
            results: Processing results
            image_path: Input image path
            output_dir: Output directory
            normalized_image: Color-normalized image
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = Path(image_path).stem
        
        # Save normalized image
        normalized_path = os.path.join(output_dir, f"{base_name}_normalized.jpg")
        save_image(normalized_image, normalized_path)
        
        # Create visualization
        detections_list = [results['detections'][k] for k in ['cucumber', 'ruler', 'label', 'color_chart'] if results['detections'][k]]
        if detections_list:
            vis_image = create_visualization(normalized_image, detections_list, self.class_names, self.class_colors)
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            save_image(vis_image, vis_path)
        
        # Save results as JSON
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save calibration report
        if results['calibration']:
            cal_report = create_calibration_report(
                results['calibration'].get('ruler', {}),
                results['calibration'].get('color_chart', {}),
                results['calibration'].get('illumination', {})
            )
            
            cal_report_path = os.path.join(output_dir, f"{base_name}_calibration.txt")
            with open(cal_report_path, 'w') as f:
                f.write(cal_report)
        
        # Save OCR report if accession ID was extracted
        if results['accession_id'] and results['accession_id'].get('accession_id'):
            ocr_report = create_ocr_report(
                [results['accession_id']],
                results['accession_id'].get('validation', {})
            )
            
            ocr_report_path = os.path.join(output_dir, f"{base_name}_ocr.txt")
            with open(ocr_report_path, 'w') as f:
                f.write(ocr_report)
        
        print(f"Results saved to {output_dir}")
    
    def _prepare_for_json(self, data: Union[Dict, List, np.ndarray]) -> Union[Dict, List]:
        """
        Prepare data for JSON serialization by converting numpy types.
        
        Args:
            data: Data to prepare
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data
    
    def batch_process(self, image_paths: List[str], 
                     output_dir: str) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
                
                # Create subdirectory for each image
                image_name = Path(image_path).stem
                image_output_dir = os.path.join(output_dir, image_name)
                
                # Process image
                result = self.process_image(image_path, image_output_dir)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def create_summary_report(self, batch_results: List[Dict]) -> str:
        """
        Create a summary report for batch processing.
        
        Args:
            batch_results: List of batch processing results
            
        Returns:
            Formatted summary report
        """
        if not batch_results:
            return "No results to summarize."
        
        # Count successful and failed processing
        successful = [r for r in batch_results if 'error' not in r]
        failed = [r for r in batch_results if 'error' in r]
        
        # Count detections by class
        detection_counts = {'cucumber': 0, 'ruler': 0, 'label': 0, 'color_chart': 0}
        
        for result in successful:
            for class_name in detection_counts:
                if result['detections'][class_name]:
                    detection_counts[class_name] += 1
        
        # Calculate average confidence
        confidences = []
        for result in successful:
            for detection in result['detections'].values():
                if detection:
                    confidences.append(detection['confidence'])
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Create report
        report = "=== BATCH PROCESSING SUMMARY ===\n\n"
        report += f"Total images: {len(batch_results)}\n"
        report += f"Successfully processed: {len(successful)}\n"
        report += f"Failed: {len(failed)}\n"
        report += f"Success rate: {len(successful)/len(batch_results)*100:.1f}%\n\n"
        
        report += "DETECTION SUMMARY:\n"
        for class_name, count in detection_counts.items():
            report += f"  {class_name}: {count}/{len(successful)} ({count/len(successful)*100:.1f}%)\n"
        
        report += f"\nAverage detection confidence: {avg_confidence:.3f}\n"
        
        if failed:
            report += "\nFAILED IMAGES:\n"
            for result in failed:
                report += f"  {result['image_path']}: {result['error']}\n"
        
        return report
