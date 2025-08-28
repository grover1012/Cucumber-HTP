#!/usr/bin/env python3
"""
Debug script to see what the model is actually detecting
"""

import cv2
import numpy as np
from ultralytics import YOLO

def debug_detections():
    print("🔍 Debugging Model Detections")
    print("=" * 50)
    
    # Load model
    model_path = "outputs/models/train/weights/best.pt"
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Class names from data.yaml
    class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
    print(f"📚 Model classes: {class_names}")
    
    # Test image
    test_image = "data/yolov12/test/images/AM030_YF_2021_jpg.rf.541ac98d17535ed79e21ac842779f108.jpg"
    print(f"📸 Testing with image: {test_image}")
    
    # Load image
    img = cv2.imread(test_image)
    if img is None:
        print("❌ Failed to load image")
        return
    
    print(f"📐 Image shape: {img.shape}")
    
    # Run inference with different confidence thresholds
    for conf_threshold in [0.1, 0.3, 0.5, 0.7]:
        print(f"\n🎯 Testing confidence threshold: {conf_threshold}")
        print("-" * 40)
        
        results = model.predict(
            source=img,
            imgsz=1024,
            conf=conf_threshold,
            iou=0.45,
            verbose=False
        )[0]
        
        if results.boxes is not None:
            boxes = results.boxes.data
            print(f"📦 Detected {len(boxes)} objects")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                class_id = int(cls)
                class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
                
                print(f"  Object {i+1}:")
                print(f"    Class: {class_name} (ID: {class_id})")
                print(f"    Confidence: {conf:.3f}")
                print(f"    BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                print(f"    Size: {x2-x1:.1f} x {y2-y1:.1f} pixels")
        else:
            print("⚠️  No objects detected")
    
    print("\n🎯 Debug completed!")

if __name__ == "__main__":
    debug_detections()
