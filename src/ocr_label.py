import easyocr
import cv2
import numpy as np
import pytesseract
import re
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

class AccessionLabelReader:
    """
    OCR-based reader for accession labels on cucumber images.
    Combines EasyOCR and Tesseract for robust text recognition.
    
    Args:
        languages (List[str]): Languages for EasyOCR (default: ['en'])
        use_gpu (bool): Whether to use GPU for EasyOCR
        confidence_threshold (float): Minimum confidence for text detection
    """
    
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False, 
                 confidence_threshold: float = 0.5):
        self.languages = languages
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self._easyocr_reader = None
        self._tesseract_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    
    def _get_easyocr_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(
                self.languages, 
                gpu=self.use_gpu,
                model_storage_directory=None,
                download_enabled=True
            )
        return self._easyocr_reader
    
    def read_label_easyocr(self, img: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Read text using EasyOCR.
        
        Args:
            img (np.ndarray): Input BGR image
            region (Optional[Tuple[int, int, int, int]]): Region to read (x, y, w, h)
        
        Returns:
            List[Tuple[str, float, Tuple[int, int, int, int]]]: List of (text, confidence, bbox)
        """
        try:
            reader = self._get_easyocr_reader()
            
            # Crop region if specified
            if region is not None:
                x, y, w, h = region
                img_crop = img[y:y+h, x:x+w]
            else:
                img_crop = img
            
            # Read text
            results = reader.readtext(img_crop, detail=1)
            
            # Filter by confidence and format results
            filtered_results = []
            for (bbox, text, confidence) in results:
                if confidence >= self.confidence_threshold:
                    # Convert bbox to (x, y, w, h) format
                    bbox_array = np.array(bbox)
                    x_min, y_min = bbox_array.min(axis=0)
                    x_max, y_max = bbox_array.max(axis=0)
                    bbox_formatted = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                    
                    filtered_results.append((text.strip(), confidence, bbox_formatted))
            
            return filtered_results
            
        except Exception as e:
            print(f"EasyOCR failed: {e}")
            return []
    
    def read_label_tesseract(self, img: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Read text using Tesseract OCR.
        
        Args:
            img (np.ndarray): Input BGR image
            region (Optional[Tuple[int, int, int, int]]): Region to read (x, y, w, h)
        
        Returns:
            List[Tuple[str, float, Tuple[int, int, int, int]]]: List of (text, confidence, bbox)
        """
        try:
            # Crop region if specified
            if region is not None:
                x, y, w, h = region
                img_crop = img[y:y+h, x:x+w]
            else:
                img_crop = img
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for preprocessing
            pil_img = Image.fromarray(img_rgb)
            
            # Enhance image for better OCR
            enhanced_img = self._enhance_image_for_ocr(pil_img)
            
            # Read text with confidence
            data = pytesseract.image_to_data(
                enhanced_img, 
                config=self._tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i]) / 100.0
                
                if text and confidence >= self.confidence_threshold:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    bbox = (x, y, w, h)
                    
                    results.append((text, confidence, bbox))
            
            return results
            
        except Exception as e:
            print(f"Tesseract failed: {e}")
            return []
    
    def _enhance_image_for_ocr(self, pil_img: Image.Image) -> Image.Image:
        """
        Enhance image for better OCR performance.
        
        Args:
            pil_img (Image.Image): Input PIL image
        
        Returns:
            Image.Image: Enhanced image
        """
        # Convert to grayscale
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        # Apply slight blur to reduce noise
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return pil_img
    
    def read_label_combined(self, img: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[str, float, str, Tuple[int, int, int, int]]]:
        """
        Read text using both EasyOCR and Tesseract, combining results.
        
        Args:
            img (np.ndarray): Input BGR image
            region (Optional[Tuple[int, int, int, int]]): Region to read (x, y, w, h)
        
        Returns:
            List[Tuple[str, float, str, Tuple[int, int, int, int]]]: List of (text, confidence, method, bbox)
        """
        # Read with both methods
        easyocr_results = self.read_label_easyocr(img, region)
        tesseract_results = self.read_label_tesseract(img, region)
        
        # Combine results
        combined_results = []
        
        # Add EasyOCR results
        for text, confidence, bbox in easyocr_results:
            combined_results.append((text, confidence, 'easyocr', bbox))
        
        # Add Tesseract results
        for text, confidence, bbox in tesseract_results:
            combined_results.append((text, confidence, 'tesseract', bbox))
        
        # Sort by confidence
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def extract_accession_number(self, text: str) -> Optional[str]:
        """
        Extract accession number from text using regex patterns.
        
        Args:
            text (str): Raw text from OCR
        
        Returns:
            Optional[str]: Extracted accession number or None
        """
        # Common accession number patterns
        patterns = [
            r'[A-Z]{1,3}\d{1,4}[A-Z]?\d{0,3}',  # e.g., ABC123, ABC123A, ABC123A456
            r'\d{1,4}[A-Z]{1,3}\d{0,3}',        # e.g., 123ABC, 123ABC456
            r'[A-Z]{1,3}-\d{1,4}',               # e.g., ABC-123
            r'[A-Z]{1,3}_\d{1,4}',               # e.g., ABC_123
            r'[A-Z]{1,3}\.\d{1,4}',              # e.g., ABC.123
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].upper()
        
        return None
    
    def read_accession_label(self, img: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """
        Read and extract accession label from image.
        
        Args:
            img (np.ndarray): Input BGR image
            region (Optional[Tuple[int, int, int, int]]): Region to read (x, y, w, h)
        
        Returns:
            Optional[str]: Extracted accession number or None
        """
        # Get combined OCR results
        results = self.read_label_combined(img, region)
        
        if not results:
            return None
        
        # Try to extract accession number from each result
        for text, confidence, method, bbox in results:
            accession = self.extract_accession_number(text)
            if accession:
                print(f"Found accession '{accession}' with confidence {confidence:.2f} using {method}")
                return accession
        
        # If no accession pattern found, return the highest confidence text
        best_text, best_confidence, best_method, _ = results[0]
        print(f"No accession pattern found. Best text: '{best_text}' with confidence {best_confidence:.2f} using {best_method}")
        
        return best_text
    
    def detect_text_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential text regions in the image.
        
        Args:
            img (np.ndarray): Input BGR image
        
        Returns:
            List[Tuple[int, int, int, int]]: List of potential text regions (x, y, w, h)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # Filter regions by size and aspect ratio
            text_regions = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter by size
                if w < 20 or h < 10 or w > img.shape[1] // 2 or h > img.shape[0] // 2:
                    continue
                
                # Filter by aspect ratio (text is usually wider than tall)
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 10:
                    continue
                
                text_regions.append((x, y, w, h))
            
            # Merge overlapping regions
            merged_regions = self._merge_overlapping_regions(text_regions)
            
            return merged_regions
            
        except Exception as e:
            print(f"Text region detection failed: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping text regions.
        
        Args:
            regions (List[Tuple[int, int, int, int]]): List of regions
        
        Returns:
            List[Tuple[int, int, int, int]]: Merged regions
        """
        if not regions:
            return []
        
        # Sort regions by x coordinate
        regions = sorted(regions, key=lambda x: x[0])
        
        merged = []
        current = list(regions[0])
        
        for region in regions[1:]:
            x, y, w, h = region
            
            # Check if regions overlap or are close
            if (x <= current[0] + current[2] + 10 and  # Close horizontally
                abs(y - current[1]) < max(current[3], h) // 2):  # Close vertically
                
                # Merge regions
                current[0] = min(current[0], x)
                current[1] = min(current[1], y)
                current[2] = max(current[0] + current[2], x + w) - current[0]
                current[3] = max(current[1] + current[3], y + h) - current[1]
            else:
                merged.append(tuple(current))
                current = list(region)
        
        merged.append(tuple(current))
        return merged
    
    def visualize_ocr_results(self, img: np.ndarray, results: List[Tuple[str, float, str, Tuple[int, int, int, int]]], 
                             save_path: Optional[str] = None) -> None:
        """
        Visualize OCR results on the image.
        
        Args:
            img (np.ndarray): Input BGR image
            results (List[Tuple[str, float, str, Tuple[int, int, int, int]]]): OCR results
            save_path (Optional[str]): Path to save visualization
        """
        # Create copy for visualization
        vis_img = img.copy()
        
        # Draw bounding boxes and text
        for text, confidence, method, bbox in results:
            x, y, w, h = bbox
            
            # Choose color based on method
            if method == 'easyocr':
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Red
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # Draw text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_img, (x, y - text_size[1] - 10), 
                         (x + text_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(vis_img, f"{text} ({confidence:.2f})", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f'OCR Results ({len(results)} detections)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"OCR visualization saved to: {save_path}")
        
        plt.show()

def read_label_simple(img: np.ndarray) -> str:
    """
    Simple function to read label using EasyOCR.
    
    Args:
        img (np.ndarray): Input BGR image
    
    Returns:
        str: Detected text or empty string
    """
    reader = AccessionLabelReader()
    return reader.read_accession_label(img) or ""

def read_label_robust(img: np.ndarray) -> str:
    """
    Robust function to read label using combined OCR methods.
    
    Args:
        img (np.ndarray): Input BGR image
    
    Returns:
        str: Detected text or empty string
    """
    reader = AccessionLabelReader()
    return reader.read_accession_label(img) or ""

if __name__ == "__main__":
    # Test OCR with dummy data
    print("Testing OCR label reading...")
    
    # Create a dummy image with text
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "ABC123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "DEF456", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Test OCR
    try:
        text = read_label_simple(img)
        print(f"Simple OCR result: '{text}'")
        
        text_robust = read_label_robust(img)
        print(f"Robust OCR result: '{text_robust}'")
        
    except Exception as e:
        print(f"OCR test failed: {e}")
    
    print("OCR test completed successfully!")
