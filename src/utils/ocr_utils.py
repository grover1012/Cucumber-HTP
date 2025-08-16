"""
OCR utilities for extracting accession IDs from cucumber labels.
Uses Tesseract OCR with image preprocessing for better accuracy.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Optional, Tuple
import re
import os


def preprocess_for_ocr(image: np.ndarray, 
                       label_mask: np.ndarray,
                       enhancement_level: float = 1.5) -> np.ndarray:
    """
    Preprocess image region for better OCR accuracy.
    
    Args:
        image: Original image
        label_mask: Binary mask of the label region
        enhancement_level: Enhancement factor for contrast
        
    Returns:
        Preprocessed image optimized for OCR
    """
    # Apply mask to get label region
    label_region = cv2.bitwise_and(image, image, mask=label_mask.astype(np.uint8))
    
    # Convert to grayscale
    gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def enhance_text_region(image: np.ndarray) -> np.ndarray:
    """
    Apply additional text enhancement techniques.
    
    Args:
        image: Preprocessed image
        
    Returns:
        Enhanced image
    """
    # Convert to PIL Image for enhancement
    pil_image = Image.fromarray(image)
    
    # Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = contrast_enhancer.enhance(1.5)
    
    # Enhance sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = sharpness_enhancer.enhance(2.0)
    
    # Convert back to numpy array
    return np.array(enhanced)


def extract_text_with_tesseract(image: np.ndarray,
                               config: str = '--psm 8 --oem 3',
                               preprocess: bool = True) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image: Input image
        config: Tesseract configuration
        preprocess: Whether to apply preprocessing
        
    Returns:
        Extracted text
    """
    try:
        if preprocess:
            # Apply preprocessing
            processed = preprocess_for_ocr(image, np.ones_like(image[:, :, 0]))
            enhanced = enhance_text_region(processed)
        else:
            enhanced = image
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(enhanced, config=config)
        
        return text.strip()
        
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""


def extract_accession_id(image: np.ndarray,
                        label_mask: np.ndarray,
                        expected_pattern: str = r'[A-Z]{2,4}\d{3,6}',
                        confidence_threshold: float = 0.6) -> Dict[str, str]:
    """
    Extract accession ID from label image.
    
    Args:
        image: Original image
        label_mask: Binary mask of the label region
        expected_pattern: Regex pattern for accession ID format
        confidence_threshold: Minimum confidence for valid extraction
        
    Returns:
        Dictionary with extracted accession ID and confidence
    """
    # Preprocess the label region
    processed = preprocess_for_ocr(image, label_mask)
    enhanced = enhance_text_region(processed)
    
    # Try different OCR configurations
    configs = [
        '--psm 8 --oem 3',      # Single word
        '--psm 7 --oem 3',      # Single text line
        '--psm 6 --oem 3',      # Uniform block of text
        '--psm 13 --oem 3',     # Raw line
    ]
    
    best_result = ""
    best_confidence = 0.0
    
    for config in configs:
        try:
            # Extract text
            text = pytesseract.image_to_string(enhanced, config=config)
            
            # Clean text
            cleaned_text = re.sub(r'[^\w\d]', '', text.upper())
            
            # Look for accession ID pattern
            matches = re.findall(expected_pattern, cleaned_text)
            
            if matches:
                # Calculate confidence based on pattern match
                confidence = len(matches[0]) / max(len(cleaned_text), 1)
                
                if confidence > best_confidence:
                    best_result = matches[0]
                    best_confidence = confidence
                    
        except Exception as e:
            continue
    
    # If no pattern match found, try to extract any alphanumeric text
    if not best_result:
        try:
            # Use simpler configuration
            text = pytesseract.image_to_string(enhanced, config='--psm 8 --oem 3')
            
            # Extract alphanumeric characters
            alphanumeric = re.sub(r'[^\w\d]', '', text.upper())
            
            if len(alphanumeric) >= 3:  # Minimum length for accession ID
                best_result = alphanumeric
                best_confidence = 0.3  # Lower confidence for non-pattern matches
                
        except Exception as e:
            pass
    
    return {
        'accession_id': best_result,
        'confidence': best_confidence,
        'is_valid': best_confidence >= confidence_threshold
    }


def validate_accession_id(accession_id: str, 
                         expected_formats: List[str] = None) -> Dict[str, bool]:
    """
    Validate extracted accession ID against expected formats.
    
    Args:
        accession_id: Extracted accession ID
        expected_formats: List of expected format patterns
        
    Returns:
        Dictionary with validation results
    """
    if expected_formats is None:
        # Common accession ID formats
        expected_formats = [
            r'^[A-Z]{2,4}\d{3,6}$',           # e.g., CU001, CUC12345
            r'^[A-Z]{1,2}\d{2,4}[A-Z]{1,2}$', # e.g., C12A, CU123B
            r'^\d{3,6}[A-Z]{1,3}$',           # e.g., 123CU, 4567CUC
        ]
    
    validation_results = {}
    
    # Check length
    validation_results['length_valid'] = 3 <= len(accession_id) <= 10
    
    # Check alphanumeric
    validation_results['alphanumeric'] = accession_id.isalnum()
    
    # Check format patterns
    format_matches = []
    for pattern in expected_formats:
        if re.match(pattern, accession_id):
            format_matches.append(True)
        else:
            format_matches.append(False)
    
    validation_results['format_valid'] = any(format_matches)
    
    # Check for common OCR errors
    common_errors = ['0', 'O', '1', 'I', '5', 'S', '8', 'B']
    has_potential_errors = any(char in accession_id for char in common_errors)
    validation_results['has_potential_errors'] = has_potential_errors
    
    # Overall validation
    validation_results['overall_valid'] = (
        validation_results['length_valid'] and
        validation_results['alphanumeric'] and
        validation_results['format_valid']
    )
    
    return validation_results


def correct_common_ocr_errors(accession_id: str) -> str:
    """
    Attempt to correct common OCR errors in accession IDs.
    
    Args:
        accession_id: Raw accession ID from OCR
        
    Returns:
        Corrected accession ID
    """
    # Common OCR substitutions
    corrections = {
        '0': 'O',  # Zero to letter O
        'O': '0',  # Letter O to zero (context-dependent)
        '1': 'I',  # One to letter I
        'I': '1',  # Letter I to one (context-dependent)
        '5': 'S',  # Five to letter S
        'S': '5',  # Letter S to five (context-dependent)
        '8': 'B',  # Eight to letter B
        'B': '8',  # Letter B to eight (context-dependent)
    }
    
    corrected = accession_id
    
    # Apply corrections based on context
    for i, char in enumerate(corrected):
        if char in corrections:
            # Try to determine if it should be a letter or number
            # based on surrounding characters
            if i > 0 and i < len(corrected) - 1:
                prev_char = corrected[i-1]
                next_char = corrected[i+1]
                
                # If surrounded by letters, likely a letter
                if prev_char.isalpha() and next_char.isalpha():
                    if char.isdigit():
                        corrected = corrected[:i] + corrections[char] + corrected[i+1:]
                
                # If surrounded by digits, likely a digit
                elif prev_char.isdigit() and next_char.isdigit():
                    if char.isalpha():
                        # Find the reverse correction
                        for k, v in corrections.items():
                            if v == char:
                                corrected = corrected[:i] + k + corrected[i+1:]
                                break
    
    return corrected


def extract_multiple_accession_ids(image: np.ndarray,
                                 label_mask: np.ndarray,
                                 max_ids: int = 5) -> List[Dict[str, str]]:
    """
    Extract multiple potential accession IDs from label image.
    
    Args:
        image: Original image
        label_mask: Binary mask of the label region
        max_ids: Maximum number of IDs to extract
        
    Returns:
        List of dictionaries with extracted IDs and confidence
    """
    # Preprocess the label region
    processed = preprocess_for_ocr(image, label_mask)
    enhanced = enhance_text_region(processed)
    
    # Try different OCR configurations to get multiple results
    configs = [
        '--psm 6 --oem 3',      # Uniform block of text
        '--psm 7 --oem 3',      # Single text line
        '--psm 8 --oem 3',      # Single word
        '--psm 13 --oem 3',     # Raw line
    ]
    
    all_results = []
    
    for config in configs:
        try:
            # Extract text
            text = pytesseract.image_to_string(enhanced, config=config)
            
            # Split into words and clean
            words = text.split()
            for word in words:
                cleaned_word = re.sub(r'[^\w\d]', '', word.upper())
                
                if len(cleaned_word) >= 3:  # Minimum length
                    # Look for accession ID patterns
                    patterns = [
                        r'[A-Z]{2,4}\d{3,6}',
                        r'[A-Z]{1,2}\d{2,4}[A-Z]{1,2}',
                        r'\d{3,6}[A-Z]{1,3}',
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, cleaned_word)
                        for match in matches:
                            # Calculate confidence
                            confidence = len(match) / len(cleaned_word)
                            
                            # Correct common errors
                            corrected = correct_common_ocr_errors(match)
                            
                            result = {
                                'accession_id': corrected,
                                'confidence': confidence,
                                'original': match,
                                'config': config
                            }
                            
                            # Check if this result is already in the list
                            if not any(r['accession_id'] == corrected for r in all_results):
                                all_results.append(result)
                                
        except Exception as e:
            continue
    
    # Sort by confidence and return top results
    all_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return all_results[:max_ids]


def create_ocr_report(extraction_results: List[Dict[str, str]],
                     validation_results: Dict[str, bool]) -> str:
    """
    Create a human-readable OCR extraction report.
    
    Args:
        extraction_results: List of extraction results
        validation_results: Validation results for the best match
        
    Returns:
        Formatted OCR report
    """
    report = "=== OCR EXTRACTION REPORT ===\n\n"
    
    if not extraction_results:
        report += "No text extracted from label.\n"
        return report
    
    # Best result
    best_result = extraction_results[0]
    report += f"BEST MATCH:\n"
    report += f"  Accession ID: {best_result['accession_id']}\n"
    report += f"  Confidence: {best_result['confidence']:.2f}\n"
    
    if 'original' in best_result:
        report += f"  Original OCR: {best_result['original']}\n"
    
    report += f"  OCR Config: {best_result.get('config', 'Unknown')}\n"
    
    # Validation results
    report += f"\nVALIDATION:\n"
    for key, value in validation_results.items():
        status = "✓" if value else "✗"
        report += f"  {key}: {status}\n"
    
    # All results
    report += f"\nALL EXTRACTIONS:\n"
    for i, result in enumerate(extraction_results):
        report += f"  {i+1}. {result['accession_id']} (confidence: {result['confidence']:.2f})\n"
    
    # Recommendations
    report += f"\nRECOMMENDATIONS:\n"
    if best_result['confidence'] < 0.6:
        report += "  - Low confidence extraction, consider manual verification\n"
    
    if validation_results.get('has_potential_errors', False):
        report += "  - Potential OCR errors detected, review manually\n"
    
    if not validation_results.get('overall_valid', False):
        report += "  - Accession ID format validation failed\n"
    
    return report
