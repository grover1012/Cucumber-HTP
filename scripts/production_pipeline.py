#!/usr/bin/env python3
"""
Production Pipeline for Cucumber HTP
Processes raw images in batches for high-throughput phenotyping
"""

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
        """Process a single image and return results"""
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
        """Process all images in a directory"""
        print(f"🚀 Processing images in: {image_dir}")
        
        image_files = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png")) + \
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
    """Main function for production pipeline"""
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
    print(f"\n📊 Production Summary:")
    print(f"  • Images Processed: {len(results)}")
    print(f"  • Total Detections: {total_detections}")
    print(f"  • Results File: {output_file}")

if __name__ == "__main__":
    main()
