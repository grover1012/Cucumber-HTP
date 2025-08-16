"""
Calibration utilities for cucumber trait extraction.
Includes ruler-based size calibration and color chart normalization.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


def detect_ruler_markings(ruler_mask: np.ndarray, 
                          image: np.ndarray,
                          expected_mm_per_cm: float = 10.0) -> Dict[str, float]:
    """
    Detect ruler markings and calculate pixel-to-mm ratio.
    
    Args:
        ruler_mask: Binary mask of the ruler region
        image: Original image
        expected_mm_per_cm: Expected millimeters per centimeter on ruler
        
    Returns:
        Dictionary with calibration parameters
    """
    # Apply mask to get ruler region
    ruler_region = cv2.bitwise_and(image, image, mask=ruler_mask.astype(np.uint8))
    
    # Convert to grayscale
    gray = cv2.cvtColor(ruler_region, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=30, maxLineGap=10)
    
    if lines is None:
        return {'pixel_to_mm_ratio': 1.0, 'confidence': 0.0}
    
    # Filter horizontal lines (ruler markings)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Consider lines within 10 degrees of horizontal as ruler markings
        if angle < 10 or angle > 170:
            horizontal_lines.append(line[0])
    
    if len(horizontal_lines) < 2:
        return {'pixel_to_mm_ratio': 1.0, 'confidence': 0.0}
    
    # Calculate distances between consecutive lines
    distances = []
    for i in range(len(horizontal_lines) - 1):
        line1 = horizontal_lines[i]
        line2 = horizontal_lines[i + 1]
        
        # Calculate average y-coordinate for each line
        y1 = (line1[1] + line1[3]) / 2
        y2 = (line2[1] + line2[3]) / 2
        
        distance = abs(y2 - y1)
        distances.append(distance)
    
    if not distances:
        return {'pixel_to_mm_ratio': 1.0, 'confidence': 0.0}
    
    # Calculate average distance between markings
    avg_distance_px = np.mean(distances)
    
    # Assuming markings are 1cm apart
    pixel_to_mm_ratio = expected_mm_per_cm / avg_distance_px
    
    # Calculate confidence based on consistency
    distance_std = np.std(distances)
    confidence = 1.0 / (1.0 + distance_std / avg_distance_px)
    
    return {
        'pixel_to_mm_ratio': pixel_to_mm_ratio,
        'confidence': confidence,
        'avg_marking_distance_px': avg_distance_px,
        'marking_distance_std': distance_std
    }


def calibrate_with_known_object(ruler_mask: np.ndarray,
                               image: np.ndarray,
                               known_length_mm: float = 100.0) -> Dict[str, float]:
    """
    Calibrate using a known object length (e.g., ruler with known dimensions).
    
    Args:
        ruler_mask: Binary mask of the ruler region
        image: Original image
        known_length_mm: Known length of the ruler in millimeters
        
    Returns:
        Dictionary with calibration parameters
    """
    # Find contours of the ruler
    contours, _ = cv2.findContours(ruler_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'pixel_to_mm_ratio': 1.0, 'confidence': 0.0}
    
    # Get largest contour (should be the ruler)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    width_px, height_px = rect[1]
    
    # Use the longer dimension as the ruler length
    ruler_length_px = max(width_px, height_px)
    
    # Calculate pixel-to-mm ratio
    pixel_to_mm_ratio = known_length_mm / ruler_length_px
    
    return {
        'pixel_to_mm_ratio': pixel_to_mm_ratio,
        'confidence': 1.0,
        'ruler_length_px': ruler_length_px,
        'known_length_mm': known_length_mm
    }


def detect_color_chart(color_chart_mask: np.ndarray,
                      image: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    Detect color chart and extract reference colors.
    
    Args:
        color_chart_mask: Binary mask of the color chart region
        image: Original image
        
    Returns:
        Dictionary with color chart information
    """
    # Apply mask to get color chart region
    color_region = cv2.bitwise_and(image, image, mask=color_chart_mask.astype(np.uint8))
    
    # Convert to RGB for analysis
    rgb_region = cv2.cvtColor(color_region, cv2.COLOR_BGR2RGB)
    
    # Find contours to identify individual color patches
    contours, _ = cv2.findContours(color_chart_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'reference_colors': np.array([]), 'confidence': 0.0}
    
    # Filter contours by area to get color patches
    min_area = 100  # Minimum area for a color patch
    color_patches = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract color from center of patch
            center_x = x + w // 2
            center_y = y + h // 2
            
            if (0 <= center_y < rgb_region.shape[0] and 
                0 <= center_x < rgb_region.shape[1]):
                color = rgb_region[center_y, center_x]
                color_patches.append({
                    'color': color,
                    'position': (center_x, center_y),
                    'area': area
                })
    
    if not color_patches:
        return {'reference_colors': np.array([]), 'confidence': 0.0}
    
    # Extract reference colors
    reference_colors = np.array([patch['color'] for patch in color_patches])
    
    # Calculate confidence based on number of detected patches
    confidence = min(len(color_patches) / 10.0, 1.0)  # Assume 10 patches is ideal
    
    return {
        'reference_colors': reference_colors,
        'confidence': confidence,
        'num_patches': len(color_patches),
        'patch_info': color_patches
    }


def normalize_colors(image: np.ndarray,
                    reference_colors: np.ndarray,
                    target_colors: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize image colors using reference color chart.
    
    Args:
        image: Input image
        reference_colors: Detected reference colors from color chart
        target_colors: Target reference colors (if known)
        
    Returns:
        Color-normalized image
    """
    if len(reference_colors) == 0:
        return image
    
    # If target colors not provided, use standard color chart values
    if target_colors is None:
        # Standard color chart colors (RGB values)
        target_colors = np.array([
            [255, 255, 255],  # White
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [128, 128, 128],  # Gray
            [0, 0, 0],        # Black
            [255, 128, 0]     # Orange
        ])
    
    # Ensure we have matching numbers of colors
    num_colors = min(len(reference_colors), len(target_colors))
    if num_colors == 0:
        return image
    
    # Use only the available colors
    ref_colors = reference_colors[:num_colors]
    tgt_colors = target_colors[:num_colors]
    
    # Calculate color transformation matrix
    # Using least squares to find transformation matrix
    try:
        # Add bias term
        ref_colors_with_bias = np.column_stack([ref_colors, np.ones(num_colors)])
        
        # Solve for transformation matrix
        transformation_matrix = np.linalg.lstsq(ref_colors_with_bias, tgt_colors, rcond=None)[0]
        
        # Apply transformation to image
        image_float = image.astype(np.float32)
        image_reshaped = image_float.reshape(-1, 3)
        
        # Add bias term
        image_with_bias = np.column_stack([image_reshaped, np.ones(image_reshaped.shape[0])])
        
        # Apply transformation
        normalized_reshaped = np.dot(image_with_bias, transformation_matrix)
        
        # Clip values and reshape back
        normalized = np.clip(normalized_reshaped, 0, 255).reshape(image.shape).astype(np.uint8)
        
        return normalized
        
    except np.linalg.LinAlgError:
        # If transformation fails, return original image
        return image


def calculate_illumination_correction(image: np.ndarray,
                                    color_chart_mask: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    Calculate illumination correction factors using color chart.
    
    Args:
        image: Input image
        color_chart_mask: Binary mask of the color chart region
        
    Returns:
        Dictionary with illumination correction parameters
    """
    # Apply mask to get color chart region
    color_region = cv2.bitwise_and(image, image, mask=color_chart_mask.astype(np.uint8))
    
    # Convert to LAB color space for better illumination analysis
    lab_region = cv2.cvtColor(color_region, cv2.COLOR_BGR2LAB)
    
    # Calculate mean L (lightness) value
    l_channel = lab_region[:, :, 0]
    mean_l = np.mean(l_channel[l_channel > 0])  # Only non-zero pixels
    
    # Calculate standard deviation of L
    l_std = np.std(l_channel[l_channel > 0])
    
    # Calculate illumination correction factor
    # Assuming ideal L value is around 128 (middle gray)
    ideal_l = 128.0
    correction_factor = ideal_l / mean_l if mean_l > 0 else 1.0
    
    # Limit correction factor to reasonable range
    correction_factor = np.clip(correction_factor, 0.5, 2.0)
    
    return {
        'correction_factor': correction_factor,
        'mean_lightness': mean_l,
        'lightness_std': l_std,
        'ideal_lightness': ideal_l
    }


def apply_illumination_correction(image: np.ndarray,
                                 correction_factor: float) -> np.ndarray:
    """
    Apply illumination correction to image.
    
    Args:
        image: Input image
        correction_factor: Illumination correction factor
        
    Returns:
        Corrected image
    """
    # Convert to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply correction to L channel
    lab_image[:, :, 0] = np.clip(lab_image[:, :, 0] * correction_factor, 0, 255)
    
    # Convert back to BGR
    corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    
    return corrected_image


def create_calibration_report(ruler_calibration: Dict[str, float],
                             color_calibration: Dict[str, Union[np.ndarray, float]],
                             illumination_calibration: Dict[str, Union[np.ndarray, float]]) -> str:
    """
    Create a human-readable calibration report.
    
    Args:
        ruler_calibration: Ruler calibration results
        color_calibration: Color chart calibration results
        illumination_calibration: Illumination calibration results
        
    Returns:
        Formatted calibration report
    """
    report = "=== CALIBRATION REPORT ===\n\n"
    
    # Ruler calibration
    report += "RULER CALIBRATION:\n"
    report += f"  Pixel-to-mm ratio: {ruler_calibration.get('pixel_to_mm_ratio', 0):.4f}\n"
    report += f"  Confidence: {ruler_calibration.get('confidence', 0):.2f}\n"
    
    if 'ruler_length_px' in ruler_calibration:
        report += f"  Ruler length (pixels): {ruler_calibration['ruler_length_px']:.1f}\n"
    
    if 'marking_distance_std' in ruler_calibration:
        report += f"  Marking consistency: {ruler_calibration['marking_distance_std']:.2f}\n"
    
    report += "\n"
    
    # Color calibration
    report += "COLOR CALIBRATION:\n"
    report += f"  Number of color patches: {color_calibration.get('num_patches', 0)}\n"
    report += f"  Confidence: {color_calibration.get('confidence', 0):.2f}\n"
    
    if 'reference_colors' in color_calibration:
        colors = color_calibration['reference_colors']
        if len(colors) > 0:
            report += f"  Reference colors detected: {len(colors)}\n"
    
    report += "\n"
    
    # Illumination calibration
    report += "ILLUMINATION CALIBRATION:\n"
    report += f"  Correction factor: {illumination_calibration.get('correction_factor', 1.0):.3f}\n"
    report += f"  Mean lightness: {illumination_calibration.get('mean_lightness', 0):.1f}\n"
    report += f"  Lightness std: {illumination_calibration.get('lightness_std', 0):.1f}\n"
    
    # Overall assessment
    overall_confidence = (
        ruler_calibration.get('confidence', 0) * 0.5 +
        color_calibration.get('confidence', 0) * 0.3 +
        (1.0 - abs(illumination_calibration.get('correction_factor', 1.0) - 1.0)) * 0.2
    )
    
    report += f"\nOVERALL CALIBRATION CONFIDENCE: {overall_confidence:.2f}\n"
    
    if overall_confidence > 0.8:
        report += "Status: EXCELLENT - Ready for accurate measurements\n"
    elif overall_confidence > 0.6:
        report += "Status: GOOD - Minor adjustments may be needed\n"
    elif overall_confidence > 0.4:
        report += "Status: FAIR - Consider recalibrating\n"
    else:
        report += "Status: POOR - Recalibration required\n"
    
    return report
