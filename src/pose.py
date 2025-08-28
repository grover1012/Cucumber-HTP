import cv2
import numpy as np
from typing import Tuple, Optional, Union
from skimage.measure import regionprops
from scipy import ndimage
import matplotlib.pyplot as plt

def rotate_to_major_axis(img: np.ndarray, mask: np.ndarray, 
                         target_angle: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Rotate image and mask so that the major axis of the cucumber is horizontal.
    Inspired by TomatoScanner's pose normalization approach.
    
    Args:
        img (np.ndarray): Input BGR image
        mask (np.ndarray): Binary mask (0/255 or 0/1)
        target_angle (float): Target angle in degrees (0 = horizontal)
    
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (rotated_image, rotated_mask, rotation_angle)
    """
    # Ensure mask is binary
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img, mask, 0.0
    
    # Find the largest contour (main cucumber)
    main_contour = max(contours, key=cv2.contourArea)
    
    if len(main_contour) < 5:
        return img, mask, 0.0
    
    # Method 1: Using contour moments and principal components
    try:
        # Convert contour to points
        points = main_contour.reshape(-1, 2).astype(np.float32)
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Center points around centroid
        centered_points = points - centroid
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Major axis is eigenvector with largest eigenvalue
        major_axis_idx = np.argmax(eigenvalues)
        major_axis = eigenvectors[:, major_axis_idx]
        
        # Compute angle relative to horizontal
        angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
        
        # Rotate to make major axis horizontal
        rotation_angle = target_angle - angle
        
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: Use minimum area rectangle
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get width and height
        width, height = rect[1]
        angle = rect[2]
        
        # Normalize angle to [-90, 90]
        if width < height:
            angle = angle + 90
        
        rotation_angle = target_angle - angle
    
    # Perform rotation
    if abs(rotation_angle) > 0.1:  # Only rotate if significant
        # Get image center
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Apply rotation to image
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
        
        # Apply rotation to mask
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height), 
                                     flags=cv2.INTER_NEAREST, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=0)
        
        return rotated_img, rotated_mask, rotation_angle
    
    return img, mask, 0.0

def normalize_multiple_cucumbers(img: np.ndarray, masks: list, 
                               target_angle: float = 0.0) -> Tuple[np.ndarray, list, list]:
    """
    Normalize pose for multiple cucumbers in the same image.
    
    Args:
        img (np.ndarray): Input BGR image
        masks (list): List of binary masks
        target_angle (float): Target angle in degrees
    
    Returns:
        Tuple[np.ndarray, list, list]: (rotated_image, rotated_masks, rotation_angles)
    """
    if not masks:
        return img, masks, []
    
    # Find the largest cucumber to determine overall rotation
    largest_mask = max(masks, key=lambda m: np.sum(m > 0))
    
    # Rotate image based on largest cucumber
    rotated_img, _, rotation_angle = rotate_to_major_axis(img, largest_mask, target_angle)
    
    # Apply same rotation to all masks
    rotated_masks = []
    rotation_angles = [rotation_angle] * len(masks)
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    for mask in masks:
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height), 
                                     flags=cv2.INTER_NEAREST, 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=0)
        rotated_masks.append(rotated_mask)
    
    return rotated_img, rotated_masks, rotation_angles

def compute_pose_metrics(original_mask: np.ndarray, rotated_mask: np.ndarray) -> dict:
    """
    Compute metrics to evaluate pose normalization quality.
    
    Args:
        original_mask (np.ndarray): Original mask
        rotated_mask (np.ndarray): Rotated mask
    
    Returns:
        dict: Dictionary containing pose metrics
    """
    # Ensure masks are binary
    if original_mask.max() > 1:
        original_mask = (original_mask > 127).astype(np.uint8)
    if rotated_mask.max() > 1:
        rotated_mask = (rotated_mask > 127).astype(np.uint8)
    
    # Compute major axis angles
    def get_major_axis_angle(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        main_contour = max(contours, key=cv2.contourArea)
        if len(main_contour) < 5:
            return 0.0
        
        try:
            points = main_contour.reshape(-1, 2).astype(np.float32)
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            major_axis_idx = np.argmax(eigenvalues)
            major_axis = eigenvectors[:, major_axis_idx]
            angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))
            return angle
        except:
            return 0.0
    
    original_angle = get_major_axis_angle(original_mask)
    rotated_angle = get_major_axis_angle(rotated_mask)
    
    # Compute aspect ratio (length/width)
    def get_aspect_ratio(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 1.0
        
        main_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(main_contour)
        width, height = rect[1]
        return max(width, height) / (min(width, height) + 1e-8)
    
    original_aspect = get_aspect_ratio(original_mask)
    rotated_aspect = get_aspect_ratio(rotated_mask)
    
    return {
        'original_angle': original_angle,
        'rotated_angle': rotated_angle,
        'angle_change': abs(rotated_angle - original_angle),
        'original_aspect': original_aspect,
        'rotated_aspect': rotated_aspect,
        'aspect_change': abs(rotated_aspect - original_aspect)
    }

def visualize_pose_normalization(img: np.ndarray, masks: list, 
                               rotated_img: np.ndarray, rotated_masks: list,
                               save_path: Optional[str] = None) -> None:
    """
    Visualize pose normalization results.
    
    Args:
        img (np.ndarray): Original image
        masks (list): Original masks
        rotated_img (np.ndarray): Rotated image
        rotated_masks (list): Rotated masks
        save_path (Optional[str]): Path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image with masks
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for mask in masks:
        if mask.max() > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                axes[0, 0].plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
    axes[0, 0].set_title('Original Image with Masks')
    axes[0, 0].axis('off')
    
    # Original masks only
    axes[0, 1].imshow(np.sum(masks, axis=0), cmap='gray')
    axes[0, 1].set_title('Original Masks')
    axes[0, 1].axis('off')
    
    # Rotated image with masks
    axes[1, 0].imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
    for mask in rotated_masks:
        if mask.max() > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                axes[1, 0].plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2)
    axes[1, 0].set_title('Rotated Image with Masks')
    axes[1, 0].axis('off')
    
    # Rotated masks only
    axes[1, 1].imshow(np.sum(rotated_masks, axis=0), cmap='gray')
    axes[1, 1].set_title('Rotated Masks')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pose normalization visualization saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test pose normalization with dummy data
    print("Testing pose normalization...")
    
    # Create a dummy image and mask
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create an elliptical mask (simulating a cucumber)
    y, x = np.ogrid[:256, :256]
    mask = ((x - 128) / 80)**2 + ((y - 128) / 40)**2 <= 1
    mask = (mask * 255).astype(np.uint8)
    
    # Test pose normalization
    rotated_img, rotated_mask, rotation_angle = rotate_to_major_axis(img, mask)
    
    print(f"Rotation angle: {rotation_angle:.2f} degrees")
    
    # Test metrics
    metrics = compute_pose_metrics(mask, rotated_mask)
    print("Pose metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("Pose normalization test completed successfully!")
