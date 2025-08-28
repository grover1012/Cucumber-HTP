import cv2
import numpy as np
from typing import Tuple, Union
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class EdgeBoost(ImageOnlyTransform):
    """
    EdgeBoost augmentation that enhances edge detection by improving contrast and acutance.
    Inspired by TomatoScanner's approach for better segmentation accuracy on edges.
    
    Args:
        lambda_c (float): Contrast enhancement factor (default: 1.5)
        lambda_a (float): Acutance (sharpness) enhancement factor (default: 1.6)
        p (float): Probability of applying the transform (default: 0.5)
    """
    
    def __init__(self, lambda_c: float = 1.5, lambda_a: float = 1.6, p: float = 0.5, always_apply: bool = False):
        super().__init__(always_apply, p)
        self.lambda_c = lambda_c
        self.lambda_a = lambda_a
    
    def apply(self, img: np.ndarray, **kwargs) -> np.ndarray:
        return self.edgeboost(img, self.lambda_c, self.lambda_a)
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("lambda_c", "lambda_a")

def edgeboost(img: np.ndarray, lambda_c: float = 1.5, lambda_a: float = 1.6) -> np.ndarray:
    """
    Apply EdgeBoost enhancement to improve edge detection.
    
    Args:
        img (np.ndarray): Input BGR image
        lambda_c (float): Contrast enhancement factor
        lambda_a (float): Acutance enhancement factor
    
    Returns:
        np.ndarray: Enhanced BGR image
    """
    # Step 1: Contrast Enhancement using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create CLAHE object for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=lambda_c, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge enhanced L channel back
    lab = cv2.merge((cl, a, b))
    img_c = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Step 2: Acutance Enhancement using Unsharp Masking
    # Create Gaussian blur for unsharp masking
    blur = cv2.GaussianBlur(img_c, (0, 0), sigmaX=1.0)
    
    # Apply unsharp mask: enhanced = original + lambda * (original - blur)
    img_sharp = cv2.addWeighted(img_c, 1 + lambda_a * 0.25, blur, -lambda_a * 0.25, 0)
    
    # Ensure pixel values are in valid range [0, 255]
    img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)
    
    return img_sharp

def edgeboost_transform(im: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply EdgeBoost transformation to image and labels.
    This function is designed to be used as a hook in the training pipeline.
    
    Args:
        im (np.ndarray): Input image
        labels (np.ndarray): Input labels
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed image and labels
    """
    im = edgeboost(im)
    return im, labels

def create_edgeboost_pipeline(lambda_c: float = 1.5, lambda_a: float = 1.6, p: float = 0.5) -> A.Compose:
    """
    Create an Albumentations pipeline that includes EdgeBoost.
    
    Args:
        lambda_c (float): Contrast enhancement factor
        lambda_a (float): Acutance enhancement factor
        p (float): Probability of applying EdgeBoost
    
    Returns:
        A.Compose: Augmentation pipeline with EdgeBoost
    """
    return A.Compose([
        EdgeBoost(lambda_c=lambda_c, lambda_a=lambda_a, p=p),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Blur(blur_limit=3),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.3),
    ])

if __name__ == "__main__":
    # Test EdgeBoost on a sample image
    import matplotlib.pyplot as plt
    
    # Create a test image (or load one)
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Apply EdgeBoost
    enhanced_img = edgeboost(test_img, lambda_c=1.5, lambda_a=1.6)
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('EdgeBoost Enhanced')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("EdgeBoost test completed successfully!")
