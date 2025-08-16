"""
Image utility functions for cucumber trait extraction.
Includes preprocessing, visualization, and basic image operations.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List, Optional, Union
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array (BGR format for OpenCV)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Save using PIL for better quality control
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(output_path, quality=quality, optimize=True)


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if maintain_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, method: str = "imagenet") -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Input image
        method: Normalization method ("imagenet", "minmax", "zscore")
        
    Returns:
        Normalized image
    """
    if method == "imagenet":
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Convert to float and normalize
        image_float = image.astype(np.float32) / 255.0
        normalized = (image_float - mean) / std
        
        return normalized
        
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        return (image - image.min()) / (image.max() - image.min())
        
    elif method == "zscore":
        # Z-score normalization
        mean = image.mean()
        std = image.std()
        return (image - mean) / std
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def enhance_image(image: np.ndarray, 
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 saturation: float = 1.0) -> np.ndarray:
    """
    Enhance image quality with basic adjustments.
    
    Args:
        image: Input image
        brightness: Brightness multiplier
        contrast: Contrast multiplier
        saturation: Saturation multiplier
        
    Returns:
        Enhanced image
    """
    # Convert to float
    image_float = image.astype(np.float32) / 255.0
    
    # Apply brightness and contrast
    enhanced = image_float * contrast + (brightness - 1.0)
    enhanced = np.clip(enhanced, 0, 1)
    
    # Apply saturation
    if saturation != 1.0:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Convert back to uint8
    return (enhanced * 255).astype(np.uint8)


def remove_background(image: np.ndarray, 
                     method: str = "grabcut") -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background from image.
    
    Args:
        image: Input image
        method: Background removal method ("grabcut", "threshold")
        
    Returns:
        Tuple of (foreground, mask)
    """
    if method == "grabcut":
        # Initialize mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Create temporary arrays
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Define rectangle (you might want to make this adaptive)
        rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask
        foreground = image * mask2[:, :, np.newaxis]
        
        return foreground, mask2
        
    elif method == "threshold":
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Invert mask
        mask = cv2.bitwise_not(thresh)
        
        # Apply mask
        foreground = image * (mask[:, :, np.newaxis] / 255.0)
        
        return foreground, mask
        
    else:
        raise ValueError(f"Unknown background removal method: {method}")


def create_visualization(image: np.ndarray, 
                        detections: List[dict],
                        class_names: List[str],
                        class_colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Create visualization of detections on image.
    
    Args:
        image: Input image
        detections: List of detection dictionaries with 'bbox', 'class_id', 'confidence'
        class_names: List of class names
        class_colors: List of BGR colors for each class
        
    Returns:
        Image with visualizations
    """
    # Create copy of image
    vis_image = image.copy()
    
    # Default colors if not provided
    if class_colors is None:
        class_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
    
    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        confidence = detection.get('confidence', 0.0)
        
        # Get class info
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        color = class_colors[class_id % len(class_colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(vis_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return vis_image


def save_visualization(image: np.ndarray, 
                      detections: List[dict],
                      class_names: List[str],
                      output_path: str,
                      class_colors: Optional[List[Tuple[int, int, int]]] = None) -> None:
    """
    Save visualization of detections to file.
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        output_path: Output file path
        class_colors: List of BGR colors for each class
    """
    vis_image = create_visualization(image, detections, class_names, class_colors)
    save_image(vis_image, output_path)


def get_image_info(image_path: str) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    image = load_image(image_path)
    
    info = {
        'path': image_path,
        'shape': image.shape,
        'height': image.shape[0],
        'width': image.shape[1],
        'channels': image.shape[2] if len(image.shape) == 3 else 1,
        'dtype': str(image.dtype),
        'size_mb': os.path.getsize(image_path) / (1024 * 1024),
        'aspect_ratio': image.shape[1] / image.shape[0]
    }
    
    return info


def batch_process_images(image_paths: List[str], 
                        output_dir: str,
                        process_func: callable,
                        **kwargs) -> List[str]:
    """
    Process multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        output_dir: Output directory for processed images
        process_func: Function to apply to each image
        **kwargs: Additional arguments for process_func
        
    Returns:
        List of output file paths
    """
    output_paths = []
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            image = load_image(image_path)
            
            # Process image
            processed = process_func(image, **kwargs)
            
            # Generate output path
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_processed{ext}")
            
            # Save processed image
            save_image(processed, output_path)
            output_paths.append(output_path)
            
            print(f"Processed {i+1}/{len(image_paths)}: {filename}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return output_paths
