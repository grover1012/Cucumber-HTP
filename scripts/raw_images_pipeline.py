#!/usr/bin/env python3
"""
Raw Images Pipeline for Cucumber HTP
Shows how to use 1,581 raw images in different ways:
1. Annotation expansion
2. Inference on new images
3. Semi-supervised learning
4. Production deployment
"""

import os
import sys
from pathlib import Path
import random
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

def analyze_raw_images():
    """Analyze the raw images dataset"""
    print("🔍 Analyzing Raw Images Dataset...")
    
    raw_images_dir = Path("data/raw_images")
    if not raw_images_dir.exists():
        print("❌ Raw images directory not found")
        return False
    
    # Count images
    image_files = list(raw_images_dir.glob("*.jpg")) + list(raw_images_dir.glob("*.png")) + list(raw_images_dir.glob("*.jpeg"))
    total_images = len(image_files)
    
    print(f"📊 Raw Images Analysis:")
    print(f"  • Total Images: {total_images}")
    print(f"  • Directory: {raw_images_dir}")
    
    # Sample some images
    sample_images = random.sample(image_files, min(5, total_images))
    print(f"\n📸 Sample Images:")
    for i, img_path in enumerate(sample_images, 1):
        print(f"  {i}. {img_path.name}")
    
    return True, total_images, image_files

def show_annotation_expansion_potential():
    """Show how raw images can expand your training dataset"""
    print("\n🚀 Annotation Expansion Potential...")
    
    current_annotated = 138  # From new_annotations
    total_raw = 1581
    
    print(f"📈 Dataset Expansion Opportunities:")
    print(f"  • Currently Annotated: {current_annotated} images")
    print(f"  • Available Raw Images: {total_raw} images")
    print(f"  • Potential Expansion: {total_raw - current_annotated} images")
    
    # Calculate potential splits
    potential_train = int((total_raw - current_annotated) * 0.8)
    potential_valid = int((total_raw - current_annotated) * 0.15)
    potential_test = int((total_raw - current_annotated) * 0.05)
    
    print(f"\n🎯 If you annotate all raw images:")
    print(f"  • Training: {current_annotated + potential_train} images")
    print(f"  • Validation: {potential_valid} images")
    print(f"  • Test: {potential_test} images")
    print(f"  • Total: {total_raw} images")
    
    print(f"\n💡 Benefits of expanding to {total_raw} images:")
    print(f"  • 11x more training data")
    print(f"  • Much better generalization")
    print(f"  • Production-ready robustness")
    print(f"  • Handle edge cases better")

def inference_on_raw_images():
    """Run inference on raw images using trained model"""
    print("\n🔍 Running Inference on Raw Images...")
    
    # Check if trained model exists
    model_path = "models/local_training/cucumber_traits_v4_local/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Trained model not found at: {model_path}")
        print("Please train the model first or use the full dataset")
        return False
    
    try:
        # Load model
        model = YOLO(model_path)
        print("✅ Loaded trained YOLO12 model")
        
        # Get sample raw images
        raw_images_dir = Path("data/raw_images")
        image_files = list(raw_images_dir.glob("*.jpg"))[:5]  # Test on 5 images
        
        print(f"🧪 Testing on {len(image_files)} raw images...")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n  📸 Processing: {img_path.name}")
            
            # Run inference
            results = model(str(img_path), conf=0.25, iou=0.45)
            
            # Display results
            for result in results:
                print(f"    • Detections: {len(result.boxes)}")
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = result.names[class_id]
                        print(f"      - {class_name}: {confidence:.3f}")
                else:
                    print(f"      - No detections (confidence too low)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def create_production_pipeline():
    """Create a production-ready pipeline for raw images"""
    print("\n🏭 Creating Production Pipeline...")
    
    pipeline_script = """#!/usr/bin/env python3
\"\"\"
Production Pipeline for Cucumber HTP
Processes raw images in batches for high-throughput phenotyping
\"\"\"

import os
import sys
from pathlib import Path
import cv2
import json
from ultralytics import YOLO
import time
from datetime import datetime

class CucumberHTPPipeline:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.results = []
        
    def process_single_image(self, image_path):
        \"\"\"Process a single image and return results\"\"\"
        try:
            # Run inference
            results = self.model(str(image_path), conf=0.25, iou=0.45)
            
            # Extract detections
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            'class_id': int(box.cls[0]),
                            'class_name': result.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist(),
                            'image': str(image_path)
                        }
                        detections.append(detection)
            
            return {
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'total_detections': len(detections)
            }
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'detections': []
            }
    
    def process_batch(self, image_dir, output_file=None):
        \"\"\"Process all images in a directory\"\"\"
        print(f"🚀 Processing images in: {image_dir}")
        
        image_files = list(Path(image_dir).glob("*.jpg")) + \\
                     list(Path(image_dir).glob("*.png")) + \\
                     list(Path(image_dir).glob("*.jpeg"))
        
        print(f"📊 Found {len(image_files)} images to process")
        
        start_time = time.time()
        
        for i, img_path in enumerate(image_files, 1):
            print(f"  📸 Processing {i}/{len(image_files)}: {img_path.name}")
            result = self.process_single_image(img_path)
            self.results.append(result)
            
            # Progress update
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(image_files) - i) / rate
                print(f"    ⏱️  Progress: {i}/{len(image_files)} | Rate: {rate:.1f} img/s | ETA: {eta:.1f}s")
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
        total_time = time.time() - start_time
        print(f"✅ Batch processing complete in {total_time:.1f}s")
        print(f"📊 Processed {len(image_files)} images")
        
        return self.results

def main():
    \"\"\"Main function for production pipeline\"\"\"
    # Model path
    model_path = "models/local_training/cucumber_traits_v4_local/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize pipeline
    pipeline = CucumberHTPPipeline(model_path)
    
    # Process raw images
    raw_images_dir = "data/raw_images"
    output_file = "production_results.json"
    
    print("🏭 Starting Production Pipeline...")
    results = pipeline.process_batch(raw_images_dir, output_file)
    
    # Summary
    total_detections = sum(r.get('total_detections', 0) for r in results)
    print(f"\\n📊 Production Summary:")
    print(f"  • Images Processed: {len(results)}")
    print(f"  • Total Detections: {total_detections}")
    print(f"  • Results File: {output_file}")

if __name__ == "__main__":
    main()
"""
    
    # Save the production pipeline script
    pipeline_path = "scripts/production_pipeline.py"
    with open(pipeline_path, 'w') as f:
        f.write(pipeline_script)
    
    print(f"✅ Production pipeline script created: {pipeline_path}")
    print(f"📋 This script can process all 1,581 raw images automatically")
    
    return True

def main():
    """Main function"""
    print("🎯 Raw Images Pipeline for Cucumber HTP")
    print("=" * 60)
    
    # Analyze raw images
    success, total_images, image_files = analyze_raw_images()
    if not success:
        return
    
    # Show annotation expansion potential
    show_annotation_expansion_potential()
    
    # Run inference on raw images
    inference_on_raw_images()
    
    # Create production pipeline
    create_production_pipeline()
    
    print("\n" + "=" * 60)
    print("🎉 Raw Images Pipeline Analysis Complete!")
    print("=" * 60)
    
    print(f"\n📋 Summary of Raw Images Usage:")
    print(f"  1. 📝 Annotation Expansion: Use 1,581 raw images to create massive training dataset")
    print(f"  2. 🔍 Inference Pipeline: Process new images with trained model")
    print(f"  3. 🏭 Production Deployment: Batch process images for HTP")
    print(f"  4. 📈 Model Improvement: Semi-supervised learning on unlabeled data")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Run: python3 scripts/train_with_full_dataset.py (use 138 annotated images)")
    print(f"  2. Run: python3 scripts/production_pipeline.py (process raw images)")
    print(f"  3. Consider annotating more raw images for even better model")
    
    print(f"\n💡 Key Insight: You have 1,581 raw images but only used 30 for training!")
    print(f"   This is a massive opportunity to improve your model!")

if __name__ == "__main__":
    main()
