#!/usr/bin/env python3
"""
Prepare 2000 cucumber images for annotation pipeline
Organizes images, creates dataset splits, and prepares for Roboflow annotation
"""

import os
import shutil
import random
from pathlib import Path
import json

class CucumberDatasetPreparer:
    def __init__(self, raw_images_dir, output_dir):
        self.raw_images_dir = Path(raw_images_dir)
        self.output_dir = Path(output_dir)
        
    def organize_images(self):
        """Organize images and create dataset structure."""
        print("🥒 Organizing 2000 cucumber images for annotation...")
        
        # Create directory structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "valid").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.raw_images_dir.glob(f"*{ext}"))
            image_files.extend(self.raw_images_dir.glob(f"*{ext.upper()}"))
        
        print(f"📸 Found {len(image_files)} image files")
        
        if len(image_files) < 100:
            print("⚠️ Warning: Less than 100 images found. Please check your image directory.")
            return False
            
        # Shuffle images for random split
        random.shuffle(image_files)
        
        # Calculate splits
        total_images = len(image_files)
        train_count = int(total_images * 0.80)
        val_count = int(total_images * 0.15)
        test_count = total_images - train_count - val_count
        
        print(f"📊 Dataset split:")
        print(f"  Training: {train_count} images (80%)")
        print(f"  Validation: {val_count} images (15%)")
        print(f"  Testing: {test_count} images (5%)")
        
        # Split images
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        # Copy images to respective directories
        self._copy_images(train_images, "train")
        self._copy_images(val_images, "valid")
        self._copy_images(test_images, "test")
        
        # Create dataset info
        self._create_dataset_info(total_images, train_count, val_count, test_count)
        
        print("✅ Dataset organization complete!")
        return True
    
    def _copy_images(self, image_list, split_name):
        """Copy images to the specified split directory."""
        split_dir = self.output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"📁 Copying {len(image_list)} images to {split_name}/...")
        
        for i, img_path in enumerate(image_list):
            # Create a clean filename
            new_name = f"{split_name}_{i:04d}{img_path.suffix}"
            dest_path = split_dir / new_name
            
            try:
                shutil.copy2(img_path, dest_path)
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i + 1}/{len(image_list)} images copied")
            except Exception as e:
                print(f"❌ Error copying {img_path}: {e}")
    
    def _create_dataset_info(self, total, train, val, test):
        """Create dataset information file."""
        info = {
            "dataset_info": {
                "name": "Cucumber HTP Dataset",
                "description": "2000+ cucumber images for high-throughput phenotyping",
                "total_images": total,
                "train_images": train,
                "val_images": val,
                "test_images": test,
                "splits": {
                    "train": 0.80,
                    "validation": 0.15,
                    "test": 0.05
                }
            },
            "annotation_guidelines": {
                "classes": [
                    "cucumber_whole",
                    "cucumber_sliced", 
                    "cucumber_partial",
                    "cucumber_damaged",
                    "ruler",
                    "label",
                    "color_chart",
                    "background_objects"
                ],
                "annotation_type": "bounding_box",
                "format": "YOLO"
            },
            "next_steps": [
                "1. Upload images to Roboflow for annotation",
                "2. Annotate all images with consistent standards",
                "3. Export YOLO format annotations",
                "4. Train YOLO12 model with transfer learning",
                "5. Fine-tune on cucumber dataset"
            ]
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📋 Dataset info saved to: {info_file}")
    
    def create_roboflow_upload_guide(self):
        """Create guide for uploading to Roboflow."""
        guide = f"""
# 🚀 Roboflow Upload Guide for {len(list(self.raw_images_dir.glob('*'))) if self.raw_images_dir.exists() else 'Unknown'} Cucumber Images

## 📁 Dataset Structure Created:
```
{self.output_dir}/
├── train/          # Training images
├── valid/          # Validation images  
├── test/           # Test images
└── dataset_info.json
```

## 🔄 Upload Steps:

### 1. Create Roboflow Project
- Go to [Roboflow](https://roboflow.com)
- Create new project: "Cucumber-HTP-Dataset"
- Choose "Object Detection" project type

### 2. Upload Images
- Upload images from each split folder
- Maintain train/valid/test structure
- Use consistent naming convention

### 3. Annotation Guidelines
- **cucumber_whole**: Complete, intact cucumber
- **cucumber_sliced**: Cut pieces
- **cucumber_partial**: Partially visible
- **cucumber_damaged**: Bruised/damaged
- **ruler**: Measurement reference
- **label**: Accession ID
- **color_chart**: Color calibration
- **background_objects**: Other objects

### 4. Quality Control
- Multiple annotators
- Regular reviews
- Consistent standards
- Inter-annotator agreement

### 5. Export
- Format: YOLO
- Include train/valid/test splits
- Download annotations

## 🎯 Next Steps After Annotation:
1. Train YOLO12 model with transfer learning
2. Fine-tune on cucumber dataset
3. Validate on test set
4. Deploy in laboratory setting
"""
        
        guide_file = self.output_dir / "ROBOFLOW_UPLOAD_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide)
        
        print(f"📖 Roboflow upload guide created: {guide_file}")

def main():
    """Main function to prepare cucumber dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare cucumber images for annotation")
    parser.add_argument("--raw-images", required=True, 
                       help="Directory containing raw cucumber images")
    parser.add_argument("--output", default="data/prepared_for_annotation",
                       help="Output directory for organized dataset")
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = CucumberDatasetPreparer(args.raw_images, args.output)
    
    # Organize images
    if preparer.organize_images():
        # Create Roboflow guide
        preparer.create_roboflow_upload_guide()
        
        print("\n🎉 Dataset preparation complete!")
        print(f"📁 Organized dataset saved to: {args.output}")
        print("📖 Check ROBOFLOW_UPLOAD_GUIDE.md for next steps")
    else:
        print("❌ Dataset preparation failed. Please check your image directory.")

if __name__ == "__main__":
    main()
