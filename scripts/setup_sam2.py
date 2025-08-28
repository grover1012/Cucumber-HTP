#!/usr/bin/env python3
"""
SAM2 Setup and Integration Script
Sets up SAM2 for proper segmentation and integrates with cucumber detection
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse

class SAM2Setup:
    def __init__(self):
        """Initialize SAM2 setup."""
        self.sam2_available = False
        self.check_sam2_status()
    
    def check_sam2_status(self):
        """Check if SAM2 is available."""
        try:
            import segment_anything
            self.sam2_available = True
            print("✅ SAM2 is already installed")
        except ImportError:
            try:
                import segment_anything_hq
                self.sam2_available = True
                print("✅ SAM-HQ is already installed")
            except ImportError:
                print("❌ SAM2 not found - will install")
                self.sam2_available = False
    
    def install_sam2(self):
        """Install SAM2 and dependencies."""
        print("🔧 Installing SAM2...")
        print("=" * 50)
        
        # Install required packages
        packages = [
            'torch',
            'torchvision',
            'opencv-python',
            'matplotlib',
            'numpy',
            'pillow'
        ]
        
        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
        
        # Install SAM2
        print("\nInstalling SAM2...")
        try:
            # Try official SAM2 first
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/facebookresearch/segment-anything.git'])
            print("✅ SAM2 installed successfully")
            self.sam2_available = True
        except subprocess.CalledProcessError:
            print("⚠️ Official SAM2 failed, trying SAM-HQ...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/SysCV/sam-hq.git'])
                print("✅ SAM-HQ installed successfully")
                self.sam2_available = True
            except subprocess.CalledProcessError:
                print("❌ Both SAM2 and SAM-HQ failed to install")
                self.sam2_available = False
    
    def download_sam2_models(self, output_dir):
        """Download SAM2 model checkpoints."""
        if not self.sam2_available:
            print("❌ SAM2 not available - install first")
            return False
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading SAM2 models to {output_dir}")
        print("=" * 50)
        
        # Model URLs
        models = {
            'sam2_h.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/sam2_h.pt',
            'sam2_l.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/sam2_l.pt',
            'sam2_b.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/sam2_b.pt'
        }
        
        for model_name, url in models.items():
            model_path = output_dir / model_name
            if model_path.exists():
                print(f"✅ {model_name} already exists")
                continue
            
            print(f"Downloading {model_name}...")
            try:
                subprocess.check_call(['curl', '-L', '-o', str(model_path), url])
                print(f"✅ {model_name} downloaded successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to download {model_name}")
                print(f"   Manual download: {url}")
        
        return True
    
    def create_sam2_integration_script(self, output_path):
        """Create a script that integrates SAM2 with cucumber detection."""
        script_content = '''#!/usr/bin/env python3
"""
SAM2 Integration for Cucumber Segmentation
Combines YOLO detection with SAM2 segmentation
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO

# Try to import SAM2
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("✅ SAM2 available")
except ImportError:
    try:
        from segment_anything_hq import sam_model_registry, SamPredictor
        SAM_AVAILABLE = True
        print("✅ SAM-HQ available")
    except ImportError:
        SAM_AVAILABLE = False
        print("❌ SAM2 not available")

class SAM2CucumberSegmenter:
    def __init__(self, yolo_model_path, sam_checkpoint_path):
        """Initialize the segmenter."""
        self.yolo_model = YOLO(yolo_model_path)
        
        if SAM_AVAILABLE and sam_checkpoint_path:
            try:
                print(f"🔄 Loading SAM2 from {sam_checkpoint_path}")
                sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
                self.sam_predictor = SamPredictor(sam)
                print("✅ SAM2 loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load SAM2: {e}")
                self.sam_predictor = None
        else:
            self.sam_predictor = None
    
    def segment_cucumbers(self, image_path, output_dir, conf_threshold=0.3):
        """Segment cucumbers using YOLO + SAM2."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"🔍 Processing image: {Path(image_path).name}")
        
        # Run YOLO detection
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)
        detections = results[0]
        
        if not detections.boxes:
            print("❌ No detections found")
            return None
        
        # Process each detection
        segmentation_results = []
        
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            
            # Only process cucumbers
            if class_id == 4:  # cucumber class
                print(f"  Processing cucumber {i+1}: conf={confidence:.3f}")
                
                # Get SAM2 segmentation mask
                mask = self._get_sam2_mask(image_rgb, bbox)
                
                if mask is not None:
                    # Save mask
                    mask_filename = f"{Path(image_path).stem}_cucumber_{i+1}_mask.png"
                    mask_path = Path(output_dir) / mask_filename
                    cv2.imwrite(str(mask_path), mask)
                    
                    # Store results
                    segmentation_results.append({
                        'id': i,
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'mask_path': str(mask_path),
                        'mask_area': np.sum(mask > 0)
                    })
                    
                    print(f"    ✅ Mask saved: {mask_filename}")
                else:
                    print(f"    ⚠️ Failed to generate mask")
        
        return segmentation_results
    
    def _get_sam2_mask(self, image, bbox):
        """Get segmentation mask from SAM2."""
        if self.sam_predictor is None:
            return None
        
        try:
            # Set image in SAM2
            self.sam_predictor.set_image(image)
            
            # Convert bbox to SAM2 format
            x1, y1, x2, y2 = bbox
            sam_bbox = np.array([x1, y1, x2, y2])
            
            # Get masks from SAM2
            masks, scores, logits = self.sam_predictor.predict(
                box=sam_bbox,
                multimask_output=True
            )
            
            # Choose best mask by score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            
            # Convert to uint8
            mask = (best_mask * 255).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            print(f"⚠️ SAM2 failed for bbox {bbox}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='SAM2 Cucumber Segmentation')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--sam-checkpoint', required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--image-path', required=True, help='Path to image to segment')
    parser.add_argument('--output-dir', required=True, help='Output directory for masks')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize segmenter
    segmenter = SAM2CucumberSegmenter(args.yolo_model, args.sam_checkpoint)
    
    # Run segmentation
    results = segmenter.segment_cucumbers(args.image_path, args.output_dir, args.conf)
    
    if results:
        print(f"✅ Segmentation completed: {len(results)} cucumbers processed")

if __name__ == "__main__":
    main()
'''
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        print(f"✅ SAM2 integration script created: {output_path}")
        return output_path
    
    def create_requirements_txt(self, output_path):
        """Create requirements.txt for SAM2 dependencies."""
        requirements = '''# SAM2 and YOLO dependencies
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
matplotlib>=3.7.0
numpy>=1.24.0
pillow>=10.0.0
ultralytics>=8.0.0
PyYAML>=6.0

# SAM2 (choose one)
# git+https://github.com/facebookresearch/segment-anything.git
# git+https://github.com/SysCV/sam-hq.git
'''
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(requirements)
        
        print(f"✅ Requirements file created: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Setup SAM2 for cucumber segmentation')
    parser.add_argument('--install', action='store_true', help='Install SAM2 and dependencies')
    parser.add_argument('--download-models', action='store_true', help='Download SAM2 model checkpoints')
    parser.add_argument('--models-dir', default='models/sam2', help='Directory for SAM2 models')
    parser.add_argument('--create-script', action='store_true', help='Create SAM2 integration script')
    parser.add_argument('--create-requirements', action='store_true', help='Create requirements.txt')
    
    args = parser.parse_args()
    
    setup = SAM2Setup()
    
    if args.install:
        setup.install_sam2()
    
    if args.download_models:
        setup.download_sam2_models(args.models_dir)
    
    if args.create_script:
        setup.create_sam2_integration_script('scripts/sam2_cucumber_segmentation.py')
    
    if args.create_requirements:
        setup.create_requirements_txt('requirements_sam2.txt')
    
    if not any([args.install, args.download_models, args.create_script, args.create_requirements]):
        print("ℹ️ Use --install, --download-models, --create-script, or --create-requirements to proceed")
        print("📋 Example: python3 scripts/setup_sam2.py --install --download-models --create-script")

if __name__ == "__main__":
    main()
