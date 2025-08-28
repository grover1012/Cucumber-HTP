#!/usr/bin/env python3
"""
Simple model performance test
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

def test_model():
    print("🚀 Testing YOLO Model Performance")
    print("=" * 50)
    
    # Load model
    print("📁 Loading model...")
    model_path = "outputs/models/train/weights/best.pt"
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Test image
    test_image = "data/yolov12/train/images/AM001_F_2019_jpg.rf.22364f57ff0590d763fa3971cf9e589b.jpg"
    print(f"📸 Testing with image: {test_image}")
    
    # Load image
    img = cv2.imread(test_image)
    if img is None:
        print("❌ Failed to load image")
        return
    
    print(f"📐 Image shape: {img.shape}")
    
    # Run inference
    print("🔍 Running inference...")
    start_time = time.time()
    
    results = model.predict(
        source=img,
        imgsz=1024,
        conf=0.3,
        iou=0.45,
        verbose=False
    )[0]
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    print(f"⏱️  Inference time: {inference_time:.2f} seconds")
    
    # Check results
    if results.masks is not None:
        masks = results.masks.data
        print(f"✅ Detected {len(masks)} objects with masks")
        
        # Check first detection
        if len(masks) > 0:
            mask = masks[0]
            print(f"📊 First mask shape: {mask.shape}")
            print(f"📊 Mask data type: {mask.dtype}")
            print(f"📊 Mask value range: {mask.min():.3f} to {mask.max():.3f}")
            
            # Count non-zero pixels
            non_zero = (mask > 0).sum().item()
            print(f"📊 Non-zero pixels in mask: {non_zero}")
            
    else:
        print("⚠️  No masks detected")
    
    if results.boxes is not None:
        boxes = results.boxes.data
        print(f"📦 Detected {len(boxes)} bounding boxes")
        
        if len(boxes) > 0:
            # Show first box
            box = boxes[0]
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            print(f"📦 First box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print(f"📦 Confidence: {conf:.3f}")
            print(f"📦 Class: {cls}")
    else:
        print("⚠️  No bounding boxes detected")
    
    print("\n🎯 Model test completed!")

if __name__ == "__main__":
    test_model()
