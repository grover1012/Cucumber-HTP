"""
Trait extraction utilities for cucumber phenotypic measurements.
Includes length, width, aspect ratio, curvature, and other geometric measurements.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from typing import Tuple, List, Dict, Optional, Union
import math


def calculate_length_width(mask: np.ndarray, 
                          pixel_to_mm_ratio: float = 1.0) -> Dict[str, float]:
    """
    Calculate length and width of cucumber from segmentation mask.
    
    Args:
        mask: Binary segmentation mask
        pixel_to_mm_ratio: Conversion ratio from pixels to millimeters
        
    Returns:
        Dictionary with length, width, and aspect ratio
    """
    # Validate mask
    if mask is None:
        return {'length': 0.0, 'width': 0.0, 'aspect_ratio': 0.0, 'error': 'No mask provided'}
    
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'length': 0.0, 'width': 0.0, 'aspect_ratio': 0.0}
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    
    # Get width and height
    width_px, height_px = rect[1]
    
    # Length is the longer dimension
    length_px = max(width_px, height_px)
    width_px = min(width_px, height_px)
    
    # Convert to real units
    length_mm = length_px * pixel_to_mm_ratio
    width_mm = width_px * pixel_to_mm_ratio
    
    # Calculate aspect ratio
    aspect_ratio = length_mm / width_mm if width_mm > 0 else 0
    
    return {
        'length': length_mm,
        'width': width_mm,
        'aspect_ratio': aspect_ratio,
        'length_px': length_px,
        'width_px': width_px
    }


def calculate_curvature(mask: np.ndarray, 
                       pixel_to_mm_ratio: float = 1.0,
                       num_points: int = 100) -> Dict[str, float]:
    """
    Calculate curvature of cucumber from segmentation mask.
    
    Args:
        mask: Binary segmentation mask
        pixel_to_mm_ratio: Conversion ratio from pixels to millimeters
        num_points: Number of points to sample along the contour
        
    Returns:
        Dictionary with curvature measurements
    """
    # Validate mask
    if mask is None:
        return {'curvature': 0.0, 'straightness': 0.0, 'error': 'No mask provided'}
    
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'curvature': 0.0, 'straightness': 0.0}
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Resample contour to get evenly spaced points
    contour_length = cv2.arcLength(largest_contour, closed=False)
    step = contour_length / num_points
    
    sampled_points = []
    current_length = 0
    
    # Get evenly spaced points along the contour
    for i in range(num_points):
        current_length += step
        # Find the point at current_length along the contour
        if len(largest_contour) > 0:
            # Simple approach: get points at regular intervals
            idx = int((i / num_points) * len(largest_contour))
            if idx < len(largest_contour):
                point = largest_contour[idx][0]
                sampled_points.append(point)
            else:
                sampled_points.append([0, 0])
        else:
            sampled_points.append([0, 0])
    
    if len(sampled_points) < 3:
        return {'curvature': 0.0, 'straightness': 0.0}
    
    # Calculate curvature at each point
    curvatures = []
    for i in range(1, len(sampled_points) - 1):
        p1 = np.array(sampled_points[i-1])
        p2 = np.array(sampled_points[i])
        p3 = np.array(sampled_points[i+1])
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Curvature is the rate of change of angle
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            curvature = angle / (np.linalg.norm(v1) + np.linalg.norm(v2)) * 2
            curvatures.append(curvature)
    
    if not curvatures:
        return {'curvature': 0.0, 'straightness': 0.0}
    
    # Calculate average curvature
    avg_curvature = np.mean(curvatures) * pixel_to_mm_ratio
    
    # Calculate straightness (1 = perfectly straight, 0 = very curved)
    straightness = 1.0 / (1.0 + avg_curvature)
    
    return {
        'curvature': avg_curvature,
        'straightness': straightness,
        'curvature_std': np.std(curvatures) * pixel_to_mm_ratio
    }


def calculate_area_volume(mask: np.ndarray, 
                         pixel_to_mm_ratio: float = 1.0) -> Dict[str, float]:
    """
    Calculate area and estimated volume of cucumber.
    
    Args:
        mask: Binary segmentation mask
        pixel_to_mm_ratio: Conversion ratio from pixels to millimeters
        
    Returns:
        Dictionary with area and volume measurements
    """
    # Calculate area in pixels
    area_px = np.sum(mask)
    
    # Convert to real units
    area_mm2 = area_px * (pixel_to_mm_ratio ** 2)
    
    # Estimate volume assuming cylindrical shape
    # Volume = area * average width
    width_measurements = calculate_length_width(mask, pixel_to_mm_ratio)
    avg_width = width_measurements['width']
    
    # Volume estimation (area * width gives approximate volume)
    volume_mm3 = area_mm2 * avg_width
    
    return {
        'area_mm2': area_mm2,
        'area_px': area_px,
        'volume_mm3': volume_mm3,
        'volume_cm3': volume_mm3 / 1000  # Convert to cm³
    }


def calculate_centroid_and_orientation(mask: np.ndarray) -> Dict[str, Union[Tuple[float, float], float]]:
    """
    Calculate centroid and orientation of cucumber.
    
    Args:
        mask: Binary segmentation mask
        
    Returns:
        Dictionary with centroid coordinates and orientation angle
    """
    # Validate mask
    if mask is None:
        return {'centroid': (0.0, 0.0), 'orientation': 0.0, 'error': 'No mask provided'}
    
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {'centroid': (0.0, 0.0), 'orientation': 0.0}
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Calculate orientation using PCA
    if len(largest_contour) > 2:
        # Reshape contour for PCA
        contour_array = largest_contour.reshape(-1, 2).astype(np.float32)
        
        # Center the data
        mean = np.mean(contour_array, axis=0)
        centered = contour_array - mean
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get the principal component (largest eigenvalue)
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Calculate angle
        angle = np.arctan2(principal_component[1], principal_component[0])
        angle_degrees = np.degrees(angle)
    else:
        angle_degrees = 0.0
    
    return {
        'centroid': (cx, cy),
        'orientation': angle_degrees,
        'centroid_x': cx,
        'centroid_y': cy
    }


def calculate_color_metrics(mask: np.ndarray, 
                           image: np.ndarray,
                           color_space: str = 'hsv') -> Dict[str, float]:
    """
    Calculate color metrics of cucumber from image and mask.
    
    Args:
        mask: Binary segmentation mask
        image: Original image (BGR format)
        color_space: Color space for analysis ('hsv', 'lab', 'rgb')
        
    Returns:
        Dictionary with color metrics
    """
    # Validate mask
    if mask is None:
        return {'error': 'No mask provided'}
    
    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    
    # Get non-zero pixels
    non_zero_pixels = masked_image[mask > 0]
    
    if len(non_zero_pixels) == 0:
        return {}
    
    if color_space == 'hsv':
        # Convert to HSV
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_image[mask > 0]
        
        # Calculate statistics
        h_mean = np.mean(hsv_pixels[:, 0])
        s_mean = np.mean(hsv_pixels[:, 1])
        v_mean = np.mean(hsv_pixels[:, 2])
        
        h_std = np.std(hsv_pixels[:, 0])
        s_std = np.std(hsv_pixels[:, 1])
        v_std = np.std(hsv_pixels[:, 2])
        
        return {
            'hue_mean': h_mean,
            'saturation_mean': s_mean,
            'value_mean': v_mean,
            'hue_std': h_std,
            'saturation_std': s_std,
            'value_std': v_std
        }
        
    elif color_space == 'lab':
        # Convert to LAB
        lab_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        lab_pixels = lab_image[mask > 0]
        
        # Calculate statistics
        l_mean = np.mean(lab_pixels[:, 0])
        a_mean = np.mean(lab_pixels[:, 1])
        b_mean = np.mean(lab_pixels[:, 2])
        
        l_std = np.std(lab_pixels[:, 0])
        a_std = np.std(lab_pixels[:, 1])
        b_std = np.std(lab_pixels[:, 2])
        
        return {
            'l_mean': l_mean,
            'a_mean': a_mean,
            'b_mean': b_mean,
            'l_std': l_std,
            'a_std': a_std,
            'b_std': b_std
        }
        
    else:  # RGB
        # Convert to RGB
        rgb_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        rgb_pixels = rgb_image[mask > 0]
        
        # Calculate statistics
        r_mean = np.mean(rgb_pixels[:, 0])
        g_mean = np.mean(rgb_pixels[:, 1])
        b_mean = np.mean(rgb_pixels[:, 2])
        
        r_std = np.std(rgb_pixels[:, 0])
        g_std = np.std(rgb_pixels[:, 1])
        b_std = np.std(rgb_pixels[:, 2])
        
        return {
            'red_mean': r_mean,
            'green_mean': g_mean,
            'blue_mean': b_mean,
            'red_std': r_std,
            'green_std': g_std,
            'blue_std': b_std
        }


def extract_all_traits(mask: np.ndarray, 
                       image: np.ndarray,
                       pixel_to_mm_ratio: float = 1.0) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Extract all phenotypic traits from cucumber mask and image.
    
    Args:
        mask: Binary segmentation mask
        image: Original image
        pixel_to_mm_ratio: Conversion ratio from pixels to millimeters
        
    Returns:
        Dictionary with all extracted traits
    """
    traits = {}
    
    # Basic measurements
    traits.update(calculate_length_width(mask, pixel_to_mm_ratio))
    traits.update(calculate_curvature(mask, pixel_to_mm_ratio))
    traits.update(calculate_area_volume(mask, pixel_to_mm_ratio))
    traits.update(calculate_centroid_and_orientation(mask))
    
    # Color metrics
    traits.update(calculate_color_metrics(mask, image, 'hsv'))
    
    # Additional derived traits
    if 'length' in traits and 'width' in traits:
        traits['length_width_ratio'] = traits['length'] / traits['width'] if traits['width'] > 0 else 0
        
    if 'area_mm2' in traits and 'length' in traits:
        traits['compactness'] = (4 * np.pi * traits['area_mm2']) / (traits['length'] ** 2) if traits['length'] > 0 else 0
    
    return traits


def validate_measurements(traits: Dict[str, Union[float, Tuple[float, float]]]) -> Dict[str, bool]:
    """
    Validate extracted measurements for reasonableness.
    
    Args:
        traits: Dictionary of extracted traits
        
    Returns:
        Dictionary indicating which measurements are valid
    """
    validation = {}
    
    # Length validation (typical cucumber: 10-40 cm)
    if 'length' in traits:
        length_cm = traits['length'] / 10  # Convert mm to cm
        validation['length_valid'] = 5 <= length_cm <= 50
    
    # Width validation (typical cucumber: 2-8 cm)
    if 'width' in traits:
        width_cm = traits['width'] / 10  # Convert mm to cm
        validation['width_valid'] = 1 <= width_cm <= 10
    
    # Aspect ratio validation
    if 'aspect_ratio' in traits:
        validation['aspect_ratio_valid'] = 2 <= traits['aspect_ratio'] <= 20
    
    # Area validation
    if 'area_mm2' in traits:
        area_cm2 = traits['area_mm2'] / 100  # Convert mm² to cm²
        validation['area_valid'] = 1 <= area_cm2 <= 100
    
    # Volume validation
    if 'volume_cm3' in traits:
        validation['volume_valid'] = 0.1 <= traits['volume_cm3'] <= 100
    
    # Color validation (HSV ranges)
    if 'hue_mean' in traits:
        validation['hue_valid'] = 0 <= traits['hue_mean'] <= 180
    
    if 'saturation_mean' in traits:
        validation['saturation_valid'] = 0 <= traits['saturation_mean'] <= 255
    
    if 'value_mean' in traits:
        validation['value_valid'] = 0 <= traits['value_mean'] <= 255
    
    return validation
