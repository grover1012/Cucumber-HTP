import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
from shapely.geometry import Polygon, LineString
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class CucumberTraitExtractor:
    """
    Extract morphological traits from cucumber segmentation masks.
    Inspired by TomatoScanner's approach for precise phenotyping.
    
    Args:
        min_contour_area (int): Minimum contour area to consider valid
        curvature_smoothing (float): Gaussian smoothing for curvature computation
    """
    
    def __init__(self, min_contour_area: int = 100, curvature_smoothing: float = 2.0):
        self.min_contour_area = min_contour_area
        self.curvature_smoothing = curvature_smoothing
    
    def extract_basic_traits(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract basic morphological traits from mask.
        
        Args:
            mask (np.ndarray): Binary mask (0/255 or 0/1)
        
        Returns:
            Dict[str, float]: Dictionary of basic traits
        """
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Find largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(main_contour) < self.min_contour_area:
            return {}
        
        # Basic measurements
        area_px = cv2.contourArea(main_contour)
        perimeter_px = cv2.arcLength(main_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Minimum area rectangle (more accurate for rotated objects)
        rect = cv2.minAreaRect(main_contour)
        (cx, cy), (width, height), angle = rect
        
        # Ensure length is always the longer dimension
        length_px = max(width, height)
        width_px = min(width, height)
        aspect_ratio = length_px / (width_px + 1e-8)
        
        # Circularity (4π * area / perimeter²)
        circularity = 4 * np.pi * area_px / (perimeter_px * perimeter_px + 1e-8)
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / (hull_area + 1e-8)
        
        return {
            'area_px': float(area_px),
            'perimeter_px': float(perimeter_px),
            'length_px': float(length_px),
            'width_px': float(width_px),
            'aspect_ratio': float(aspect_ratio),
            'circularity': float(circularity),
            'solidity': float(solidity),
            'bounding_width': float(w),
            'bounding_height': float(h),
            'center_x': float(cx),
            'center_y': float(cy),
            'rotation_angle': float(angle)
        }
    
    def extract_advanced_traits(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract advanced morphological traits including curvature analysis.
        
        Args:
            mask (np.ndarray): Binary mask (0/255 or 0/1)
        
        Returns:
            Dict[str, float]: Dictionary of advanced traits
        """
        basic_traits = self.extract_basic_traits(mask)
        if not basic_traits:
            return {}
        
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return basic_traits
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Curvature analysis
        curvature_traits = self._compute_curvature_traits(main_contour)
        
        # Shape complexity
        complexity_traits = self._compute_complexity_traits(main_contour, basic_traits)
        
        # Combine all traits
        all_traits = {**basic_traits, **curvature_traits, **complexity_traits}
        
        return all_traits
    
    def _compute_curvature_traits(self, contour: np.ndarray) -> Dict[str, float]:
        """
        Compute curvature-related traits.
        
        Args:
            contour (np.ndarray): Contour points
        
        Returns:
            Dict[str, float]: Curvature traits
        """
        try:
            # Convert contour to points
            points = contour.reshape(-1, 2).astype(np.float32)
            
            if len(points) < 10:
                return {}
            
            # Smooth contour for better curvature computation
            points_smooth = ndimage.gaussian_filter1d(points, sigma=self.curvature_smoothing, axis=0)
            
            # Compute curvature using finite differences
            dx = np.gradient(points_smooth[:, 0])
            dy = np.gradient(points_smooth[:, 1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Curvature = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
            curvature = np.abs(dx * d2y - dy * d2x) / (dx * dx + dy * dy + 1e-8) ** 1.5
            
            # Remove infinite/NaN values
            curvature = curvature[np.isfinite(curvature)]
            
            if len(curvature) == 0:
                return {}
            
            # Curvature statistics
            mean_curvature = np.mean(curvature)
            max_curvature = np.max(curvature)
            curvature_variance = np.var(curvature)
            
            # Find regions of high curvature (potential bends)
            high_curvature_threshold = np.percentile(curvature, 90)
            high_curvature_count = np.sum(curvature > high_curvature_threshold)
            
            return {
                'mean_curvature': float(mean_curvature),
                'max_curvature': float(max_curvature),
                'curvature_variance': float(curvature_variance),
                'high_curvature_count': int(high_curvature_count),
                'curvature_range': float(max_curvature - np.min(curvature))
            }
            
        except Exception as e:
            print(f"Curvature computation failed: {e}")
            return {}
    
    def _compute_complexity_traits(self, contour: np.ndarray, basic_traits: Dict[str, float]) -> Dict[str, float]:
        """
        Compute shape complexity traits.
        
        Args:
            contour (np.ndarray): Contour points
            basic_traits (Dict[str, float]): Basic traits already computed
        
        Returns:
            Dict[str, float]: Complexity traits
        """
        try:
            # Convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            
            if defects is not None:
                defect_depths = []
                for defect in defects:
                    s, e, f, d = defect[0]
                    defect_depths.append(d / 256.0)  # Convert to pixels
                
                if defect_depths:
                    max_defect_depth = max(defect_depths)
                    mean_defect_depth = np.mean(defect_depths)
                    defect_count = len(defect_depths)
                else:
                    max_defect_depth = mean_defect_depth = defect_count = 0
            else:
                max_defect_depth = mean_defect_depth = defect_count = 0
            
            # Fractal dimension approximation (box counting)
            fractal_dim = self._estimate_fractal_dimension(contour)
            
            # Shape elongation (Feret diameter ratio)
            # Use minimum area rectangle dimensions
            length = basic_traits.get('length_px', 0)
            width = basic_traits.get('width_px', 0)
            elongation = length / (width + 1e-8)
            
            # Compactness (perimeter² / area)
            area = basic_traits.get('area_px', 0)
            perimeter = basic_traits.get('perimeter_px', 0)
            compactness = (perimeter * perimeter) / (4 * np.pi * area + 1e-8)
            
            return {
                'max_defect_depth': float(max_defect_depth),
                'mean_defect_depth': float(mean_defect_depth),
                'defect_count': int(defect_count),
                'fractal_dimension': float(fractal_dim),
                'elongation': float(elongation),
                'compactness': float(compactness)
            }
            
        except Exception as e:
            print(f"Complexity computation failed: {e}")
            return {}
    
    def _estimate_fractal_dimension(self, contour: np.ndarray, max_box_size: int = 64) -> float:
        """
        Estimate fractal dimension using box counting method.
        
        Args:
            contour (np.ndarray): Contour points
            max_box_size (int): Maximum box size for counting
        
        Returns:
            float: Estimated fractal dimension
        """
        try:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create binary mask for the contour
            mask = np.zeros((h, w), dtype=np.uint8)
            contour_shifted = contour - np.array([x, y])
            cv2.fillPoly(mask, [contour_shifted], 255)
            
            # Box counting at different scales
            box_sizes = []
            box_counts = []
            
            for box_size in range(2, min(max_box_size, min(w, h) // 2)):
                if box_size > 1:
                    # Count boxes that contain contour pixels
                    h_boxes = h // box_size
                    w_boxes = w // box_size
                    
                    count = 0
                    for i in range(h_boxes):
                        for j in range(w_boxes):
                            y1, y2 = i * box_size, (i + 1) * box_size
                            x1, x2 = j * box_size, (j + 1) * box_size
                            
                            if np.any(mask[y1:y2, x1:x2] > 0):
                                count += 1
                    
                    if count > 0:
                        box_sizes.append(box_size)
                        box_counts.append(count)
            
            if len(box_sizes) < 3:
                return 1.0  # Default for simple shapes
            
            # Compute fractal dimension as slope of log-log plot
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
            
            # Linear regression
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = -coeffs[0]  # Negative slope
            
            return max(1.0, min(2.0, fractal_dim))  # Clamp to reasonable range
            
        except Exception as e:
            print(f"Fractal dimension computation failed: {e}")
            return 1.0
    
    def convert_to_cm(self, traits: Dict[str, float], px_per_cm: float) -> Dict[str, float]:
        """
        Convert pixel-based traits to centimeter-based traits.
        
        Args:
            traits (Dict[str, float]): Traits in pixels
            px_per_cm (float): Pixels per centimeter conversion factor
        
        Returns:
            Dict[str, float]: Traits in centimeters
        """
        if px_per_cm is None or px_per_cm <= 0:
            return traits
        
        cm_traits = {}
        for key, value in traits.items():
            if key.endswith('_px'):
                # Convert linear measurements
                cm_key = key.replace('_px', '_cm')
                cm_traits[cm_key] = value / px_per_cm
            elif key.endswith('_px2'):
                # Convert area measurements
                cm_key = key.replace('_px2', '_cm2')
                cm_traits[cm_key] = value / (px_per_cm * px_per_cm)
            else:
                # Keep dimensionless traits unchanged
                cm_traits[key] = value
        
        return cm_traits
    
    def visualize_traits(self, mask: np.ndarray, traits: Dict[str, float], 
                        save_path: Optional[str] = None) -> None:
        """
        Visualize extracted traits on the mask.
        
        Args:
            mask (np.ndarray): Binary mask
            traits (Dict[str, float]): Extracted traits
            save_path (Optional[str]): Path to save visualization
        """
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found for visualization")
            return
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original mask with contour
        axes[0].imshow(mask, cmap='gray')
        axes[0].plot(main_contour[:, 0, 0], main_contour[:, 0, 1], 'r-', linewidth=2)
        
        # Draw bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=2)
        axes[0].add_patch(rect)
        
        # Draw minimum area rectangle
        min_rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)
        axes[0].plot(box[:, 0], box[:, 1], 'g-', linewidth=2)
        
        axes[0].set_title('Mask with Contours and Bounding Boxes')
        axes[0].axis('off')
        
        # Trait summary
        axes[1].axis('off')
        trait_text = "Extracted Traits:\n\n"
        
        # Group traits by category
        basic_keys = ['area_px', 'length_px', 'width_px', 'aspect_ratio']
        advanced_keys = ['circularity', 'solidity', 'curvature_variance', 'fractal_dimension']
        
        trait_text += "Basic Traits:\n"
        for key in basic_keys:
            if key in traits:
                trait_text += f"  {key}: {traits[key]:.3f}\n"
        
        trait_text += "\nAdvanced Traits:\n"
        for key in advanced_keys:
            if key in traits:
                trait_text += f"  {key}: {traits[key]:.3f}\n"
        
        axes[1].text(0.1, 0.9, trait_text, transform=axes[1].transAxes, 
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trait visualization saved to: {save_path}")
        
        plt.show()

def extract_traits_simple(mask: np.ndarray) -> Dict[str, float]:
    """
    Simple function to extract basic traits from mask.
    
    Args:
        mask (np.ndarray): Binary mask
    
    Returns:
        Dict[str, float]: Basic traits
    """
    extractor = CucumberTraitExtractor()
    return extractor.extract_basic_traits(mask)

def extract_traits_advanced(mask: np.ndarray) -> Dict[str, float]:
    """
    Advanced function to extract comprehensive traits from mask.
    
    Args:
        mask (np.ndarray): Binary mask
    
    Returns:
        Dict[str, float]: Comprehensive traits
    """
    extractor = CucumberTraitExtractor()
    return extractor.extract_advanced_traits(mask)

def px_to_cm(val_px: float, px_per_cm: float) -> float:
    """
    Convert pixels to centimeters.
    
    Args:
        val_px (float): Value in pixels
        px_per_cm (float): Pixels per centimeter
    
    Returns:
        float: Value in centimeters
    """
    if px_per_cm is None or px_per_cm <= 0:
        return val_px
    return val_px / px_per_cm

def px2_to_cm2(val_px2: float, px_per_cm: float) -> float:
    """
    Convert square pixels to square centimeters.
    
    Args:
        val_px2 (float): Value in square pixels
        px_per_cm (float): Pixels per centimeter
    
    Returns:
        float: Value in square centimeters
    """
    if px_per_cm is None or px_per_cm <= 0:
        return val_px2
    return val_px2 / (px_per_cm * px_per_cm)

if __name__ == "__main__":
    # Test trait extraction with dummy data
    print("Testing trait extraction...")
    
    # Create a dummy elliptical mask (simulating a cucumber)
    mask = np.zeros((256, 256), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    mask = ((x - 128) / 80)**2 + ((y - 128) / 40)**2 <= 1
    mask = (mask * 255).astype(np.uint8)
    
    # Test basic trait extraction
    basic_traits = extract_traits_simple(mask)
    print("Basic traits:")
    for key, value in basic_traits.items():
        print(f"  {key}: {value:.3f}")
    
    # Test advanced trait extraction
    advanced_traits = extract_traits_advanced(mask)
    print("\nAdvanced traits:")
    for key, value in advanced_traits.items():
        if key not in basic_traits:
            print(f"  {key}: {value:.3f}")
    
    # Test conversion to cm
    px_per_cm = 50.0  # Example conversion factor
    cm_traits = CucumberTraitExtractor().convert_to_cm(basic_traits, px_per_cm)
    print(f"\nTraits in cm (assuming {px_per_cm} px/cm):")
    for key, value in cm_traits.items():
        if key.endswith('_cm') or key.endswith('_cm2'):
            print(f"  {key}: {value:.3f}")
    
    print("Trait extraction test completed successfully!")
