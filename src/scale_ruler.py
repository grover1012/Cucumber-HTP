import cv2
import numpy as np
import pytesseract
import re
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

class RulerDetector:
    """
    Ruler detection and scale computation for pixel-to-cm conversion.
    Inspired by TomatoScanner's approach but adapted for cucumber phenotyping.
    
    Args:
        min_line_length (int): Minimum line length for ruler detection
        min_line_gap (int): Minimum gap between ruler lines
        tick_threshold (float): Threshold for tick detection (percentile)
        ruler_width (int): Expected width of ruler region for cropping
    """
    
    def __init__(self, min_line_length: int = 200, min_line_gap: int = 10, 
                 tick_threshold: float = 95, ruler_width: int = 80):
        self.min_line_length = min_line_length
        self.min_line_gap = min_line_gap
        self.tick_threshold = tick_threshold
        self.ruler_width = ruler_width
    
    def find_ruler_region(self, img: np.ndarray) -> Tuple[int, int, int]:
        """
        Find the ruler region in the image using line detection.
        
        Args:
            img (np.ndarray): Input BGR image
        
        Returns:
            Tuple[int, int, int]: (x_center, x_start, x_end) of ruler region
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 80, 160)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, 
                               minLineLength=self.min_line_length, 
                               maxLineGap=self.min_line_gap)
        
        if lines is None:
            raise RuntimeError("No lines found for ruler detection")
        
        # Filter vertical lines (ruler ticks)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly vertical
            if abs(x1 - x2) < 5 and abs(y1 - y2) > 50:
                vertical_lines.append((x1 + x2) // 2)
        
        if not vertical_lines:
            raise RuntimeError("No vertical ruler lines found")
        
        # Find the densest cluster of vertical lines
        vertical_lines = np.array(vertical_lines)
        x_center = int(np.median(vertical_lines))
        
        # Define ruler region boundaries
        x_start = max(0, x_center - self.ruler_width // 2)
        x_end = min(img.shape[1], x_center + self.ruler_width // 2)
        
        return x_center, x_start, x_end
    
    def detect_ticks(self, img_strip: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect ruler ticks in a vertical strip.
        
        Args:
            img_strip (np.ndarray): Vertical strip around ruler
        
        Returns:
            List[Tuple[int, int]]: List of (start_row, end_row) for each tick
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 80, 160)
        
        # Project edges horizontally (sum along columns)
        projection = edges.sum(axis=1)
        
        # Smooth projection to reduce noise
        projection_smooth = gaussian_filter1d(projection.astype(float), sigma=1.0)
        
        # Find peaks above threshold
        threshold = np.percentile(projection_smooth, self.tick_threshold)
        peak_indices = np.where(projection_smooth > threshold)[0]
        
        if len(peak_indices) < 5:
            raise RuntimeError("Not enough ticks detected")
        
        # Group consecutive rows into tick regions
        ticks = []
        if len(peak_indices) > 0:
            start = peak_indices[0]
            for i in range(1, len(peak_indices)):
                if peak_indices[i] != peak_indices[i-1] + 1:
                    ticks.append((start, peak_indices[i-1]))
                    start = peak_indices[i]
            ticks.append((start, peak_indices[-1]))
        
        return ticks
    
    def compute_scale(self, img: np.ndarray, expected_tick_distance_cm: float = 1.0) -> float:
        """
        Compute pixels-per-cm scale from ruler.
        
        Args:
            img (np.ndarray): Input BGR image
            expected_tick_distance_cm (float): Expected distance between major ticks in cm
        
        Returns:
            float: Pixels per cm
        """
        try:
            # Find ruler region
            x_center, x_start, x_end = self.find_ruler_region(img)
            
            # Crop vertical strip around ruler
            img_strip = img[:, x_start:x_end]
            
            # Detect ticks
            ticks = self.detect_ticks(img_strip)
            
            if len(ticks) < 5:
                raise RuntimeError("Not enough ticks for reliable scale computation")
            
            # Compute tick centers
            tick_centers = np.array([(start + end) // 2 for start, end in ticks])
            
            # Compute distances between consecutive ticks
            tick_distances = np.diff(tick_centers)
            
            # Use median distance for robustness
            median_distance_px = np.median(tick_distances)
            
            # Convert to pixels per cm
            px_per_cm = median_distance_px / expected_tick_distance_cm
            
            return px_per_cm
            
        except Exception as e:
            print(f"Scale detection failed: {e}")
            return None
    
    def read_ruler_numbers(self, img_strip: np.ndarray) -> List[Tuple[int, str]]:
        """
        Read numbers from ruler using OCR.
        
        Args:
            img_strip (np.ndarray): Vertical strip around ruler
        
        Returns:
            List[Tuple[int, str]]: List of (row_position, number_text)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better OCR
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Try to read text with different PSM modes
        numbers = []
        for psm in [6, 7, 8]:  # Different page segmentation modes
            try:
                config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(enhanced, config=config)
                
                # Extract numbers
                number_matches = re.findall(r'\d+', text)
                if number_matches:
                    # For now, return the first number found
                    # In a more sophisticated version, you'd map positions to numbers
                    numbers.append((0, number_matches[0]))
                    break
            except:
                continue
        
        return numbers
    
    def compute_scale_with_ocr(self, img: np.ndarray, 
                              expected_tick_distance_cm: float = 1.0) -> float:
        """
        Compute scale using both tick detection and OCR for validation.
        
        Args:
            img (np.ndarray): Input BGR image
            expected_tick_distance_cm (float): Expected distance between major ticks in cm
        
        Returns:
            float: Pixels per cm
        """
        try:
            # Find ruler region
            x_center, x_start, x_end = self.find_ruler_region(img)
            
            # Crop vertical strip around ruler
            img_strip = img[:, x_start:x_end]
            
            # Detect ticks
            ticks = self.detect_ticks(img_strip)
            
            if len(ticks) < 5:
                raise RuntimeError("Not enough ticks for reliable scale computation")
            
            # Compute tick centers
            tick_centers = np.array([(start + end) // 2 for start, end in ticks])
            
            # Compute distances between consecutive ticks
            tick_distances = np.diff(tick_centers)
            
            # Use median distance for robustness
            median_distance_px = np.median(tick_distances)
            
            # Convert to pixels per cm
            px_per_cm = median_distance_px / expected_tick_distance_cm
            
            # Optional: Validate with OCR if available
            try:
                numbers = self.read_ruler_numbers(img_strip)
                if numbers:
                    print(f"OCR detected numbers: {[num[1] for num in numbers]}")
            except:
                pass
            
            return px_per_cm
            
        except Exception as e:
            print(f"Scale detection with OCR failed: {e}")
            return None
    
    def visualize_ruler_detection(self, img: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Visualize ruler detection results.
        
        Args:
            img (np.ndarray): Input BGR image
            save_path (Optional[str]): Path to save visualization
        """
        try:
            # Find ruler region
            x_center, x_start, x_end = self.find_ruler_region(img)
            
            # Crop vertical strip around ruler
            img_strip = img[:, x_start:x_end]
            
            # Detect ticks
            ticks = self.detect_ticks(img_strip)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 8))
            
            # Original image with ruler region highlighted
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].axvline(x=x_start, color='red', linestyle='--', linewidth=2)
            axes[0].axvline(x=x_end, color='red', linestyle='--', linewidth=2)
            axes[0].axvline(x=x_center, color='green', linewidth=2)
            axes[0].set_title('Ruler Region Detection')
            axes[0].axis('off')
            
            # Ruler strip
            axes[1].imshow(cv2.cvtColor(img_strip, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Ruler Strip')
            axes[1].axis('off')
            
            # Tick detection visualization
            gray = cv2.cvtColor(img_strip, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            projection = edges.sum(axis=1)
            
            axes[2].plot(projection, range(len(projection)))
            axes[2].set_ylim(len(projection), 0)  # Invert y-axis to match image
            axes[2].set_title('Edge Projection & Tick Detection')
            axes[2].set_xlabel('Edge Intensity')
            axes[2].set_ylabel('Row Position')
            
            # Mark detected ticks
            for start, end in ticks:
                center = (start + end) // 2
                axes[2].axhline(y=center, color='red', alpha=0.7, linewidth=2)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Ruler detection visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {e}")

def find_ppcm_simple(img: np.ndarray, expected_tick_distance_cm: float = 1.0) -> float:
    """
    Simple function to find pixels per cm from ruler.
    
    Args:
        img (np.ndarray): Input BGR image
        expected_tick_distance_cm (float): Expected distance between major ticks in cm
    
    Returns:
        float: Pixels per cm, or None if detection failed
    """
    detector = RulerDetector()
    return detector.compute_scale(img, expected_tick_distance_cm)

def find_ppcm_robust(img: np.ndarray, expected_tick_distance_cm: float = 1.0) -> float:
    """
    Robust function to find pixels per cm with OCR validation.
    
    Args:
        img (np.ndarray): Input BGR image
        expected_tick_distance_cm (float): Expected distance between major ticks in cm
    
    Returns:
        float: Pixels per cm, or None if detection failed
    """
    detector = RulerDetector()
    return detector.compute_scale_with_ocr(img, expected_tick_distance_cm)

if __name__ == "__main__":
    # Test ruler detection with dummy data
    print("Testing ruler detection...")
    
    # Create a dummy image with a ruler
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add some vertical lines to simulate ruler ticks
    for i in range(0, 512, 50):
        img[:, i:i+2] = [255, 255, 255]  # White lines
    
    # Test scale detection
    try:
        px_per_cm = find_ppcm_simple(img)
        print(f"Detected scale: {px_per_cm:.2f} pixels/cm")
    except Exception as e:
        print(f"Scale detection failed: {e}")
    
    print("Ruler detection test completed successfully!")
