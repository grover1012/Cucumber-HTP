#!/usr/bin/env python3
"""
Auto-label images using existing trained YOLO12 model
Converts detections to YOLO format for Roboflow import
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO

class AutoLabeler:
    def __init__(self, model_path):
        """Initialize auto-labeler with trained model."""
        self.model = YOLO(model_path)
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
    def convert_to_yolo_format(self, results, image_path):
        """Convert YOLO results to YOLO format labels."""
        yolo_labels = []
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            return yolo_labels
            
        height, width = img.shape[:2]
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Get class ID
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Only include high-confidence detections
                    if confidence > 0.5:
                        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        return yolo_labels
    
    def auto_label_directory(self, input_dir, output_dir, confidence_threshold=0.5):
        """Auto-label all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"📸 Found {len(image_files)} images to auto-label")
        
        labeled_count = 0
        total_detections = 0
        
        for i, image_file in enumerate(image_files):
            print(f"🔄 Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Run inference
                results = self.model(str(image_file), conf=confidence_threshold)
                
                # Convert to YOLO format
                yolo_labels = self.convert_to_yolo_format(results, image_file)
                
                if yolo_labels:
                    # Save YOLO format labels
                    label_file = output_path / f"{image_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        for label in yolo_labels:
                            f.write(label + '\n')
                    
                    labeled_count += 1
                    total_detections += len(yolo_labels)
                    
                    print(f"  ✅ Labeled with {len(yolo_labels)} objects")
                else:
                    print(f"  ⚠️ No objects detected")
                    
            except Exception as e:
                print(f"  ❌ Error processing {image_file.name}: {e}")
        
        print(f"\n🎉 Auto-labeling complete!")
        print(f"📊 Results:")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Images labeled: {labeled_count}")
        print(f"  Total detections: {total_detections}")
        print(f"  Labels saved to: {output_path}")
        
        return labeled_count, total_detections
    
    def create_roboflow_import_guide(self, output_dir):
        """Create guide for importing auto-labeled data to Roboflow."""
        guide = f"""
# 🚀 Roboflow Import Guide for Auto-Labeled Images

## 📁 Auto-Labeled Data Structure:
```
{output_dir}/
├── train_images/          # Your training images
├── train_labels/          # Auto-generated YOLO labels
├── valid_images/          # Validation images  
├── valid_labels/          # Auto-generated YOLO labels
├── test_images/           # Test images
└── test_labels/           # Auto-generated YOLO labels
```

## 🔄 Import Steps:

### 1. Prepare Data for Roboflow
- Images and labels should have matching names
- Example: `image_001.jpg` and `image_001.txt`

### 2. Upload to Roboflow
- Go to your Roboflow project
- Click "Upload Images"
- Select both images and labels folders
- Maintain train/valid/test splits

### 3. Review and Correct
- **Review auto-labels** for accuracy
- **Correct mislabeled objects** (much faster than manual labeling)
- **Add missing objects** that weren't detected
- **Remove false positives**

### 4. Quality Control
- Check bounding box accuracy
- Verify class labels
- Ensure consistent standards
- Multiple annotator review

## ⚡ Time Savings:
- **Manual labeling**: 2-3 months for 1,652 images
- **Auto-labeling + review**: 1-2 weeks for 1,652 images
- **Time saved**: 80-90%

## 🎯 Next Steps:
1. Review auto-generated labels
2. Correct errors and add missing objects
3. Upload corrected data to Roboflow
4. Begin advanced training pipeline
"""
        
        guide_file = Path(output_dir) / "ROBOFLOW_IMPORT_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide)
        
        print(f"📖 Roboflow import guide created: {guide_file}")

def main():
    """Main function for auto-labeling images."""
    parser = argparse.ArgumentParser(description="Auto-label images using trained YOLO12 model")
    parser.add_argument("--model", required=True, help="Path to trained YOLO12 model")
    parser.add_argument("--input-dir", required=True, help="Directory containing images to label")
    parser.add_argument("--output-dir", required=True, help="Output directory for labels")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Initialize auto-labeler
    labeler = AutoLabeler(args.model)
    
    # Auto-label images
    labeled_count, total_detections = labeler.auto_label_directory(
        args.input_dir, 
        args.output_dir, 
        args.confidence
    )
    
    # Create Roboflow import guide
    labeler.create_roboflow_import_guide(args.output_dir)
    
    print(f"\n🎯 Auto-labeling pipeline complete!")
    print(f"📁 Labels saved to: {args.output_dir}")
    print(f"📖 Check ROBOFLOW_IMPORT_GUIDE.md for next steps")

if __name__ == "__main__":
    main()
