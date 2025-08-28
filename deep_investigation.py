#!/usr/bin/env python3
"""
Deep investigation of the model's behavior and segmentation quality
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def deep_investigation():
    print("🔍 DEEP INVESTIGATION: What is the model actually doing?")
    print("=" * 70)
    
    # Load model
    model_path = "outputs/models/train/weights/best.pt"
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # Test image
    test_image = "data/yolov12/test/images/AM032_F_2019_jpg.rf.bf6c613712c5f500c0574c77a00d728a.jpg"
    print(f"📸 Testing with image: {test_image}")
    
    # Load image
    img = cv2.imread(test_image)
    if img is None:
        print("❌ Failed to load image")
        return
    
    img_height, img_width = img.shape[:2]
    print(f"📐 Image dimensions: {img_width}x{img_height}")
    
    # Run inference with very low confidence to see everything
    print(f"\n🎯 Running inference with confidence 0.1...")
    results = model.predict(
        source=img,
        imgsz=1024,
        conf=0.1,
        iou=0.45,
        verbose=False
    )[0]
    
    if results.boxes is not None:
        boxes = results.boxes.data
        masks = results.masks.data if results.masks is not None else []
        
        print(f"📦 Detected {len(boxes)} objects with {len(masks)} masks")
        
        # Class names
        class_names = ['big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice']
        
        # Analyze each detection
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            class_id = int(cls)
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
            
            print(f"\n🔍 Object {i+1}: {class_name} (ID: {class_id})")
            print(f"  📊 Confidence: {conf:.3f}")
            print(f"  📦 BBox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print(f"  📏 Size: {x2-x1:.1f} x {y2-y1:.1f} pixels")
            
            # Analyze mask quality
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            print(f"  🎭 Mask shape: {mask_np.shape}")
            print(f"  🎭 Mask value range: {mask_np.min()} to {mask_np.min()}")
            
            # Count mask pixels
            non_zero = (mask_np > 0).sum().item()
            total_pixels = mask_np.shape[0] * mask_np.shape[1]
            coverage = (non_zero / total_pixels) * 100
            print(f"  🎭 Mask coverage: {non_zero}/{total_pixels} pixels ({coverage:.2f}%)")
            
            # Check if mask makes sense for the bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            mask_area = non_zero
            print(f"  📊 BBox area: {bbox_area:.1f} pixels")
            print(f"  📊 Mask area: {mask_area} pixels")
            print(f"  📊 Mask/bbox ratio: {mask_area/bbox_area:.3f}")
            
            # Analyze mask shape
            if non_zero > 0:
                # Find mask boundaries
                rows = np.any(mask_np > 0, axis=1)
                cols = np.any(mask_np > 0, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                mask_width = cmax - cmin
                mask_height = rmax - rmin
                print(f"  📐 Mask dimensions: {mask_width} x {mask_height} pixels")
                print(f"  📐 Mask aspect ratio: {mask_width/mask_height:.3f}")
                
                # Check if mask is reasonable size
                if mask_width < 10 or mask_height < 10:
                    print(f"  ⚠️  WARNING: Mask is very small - might be noise!")
                if mask_width > 1000 or mask_height > 1000:
                    print(f"  ⚠️  WARNING: Mask is very large - might be wrong!")
    
    else:
        print("⚠️  No objects detected")
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print("=" * 70)
    print("This will show us:")
    print("• What objects the model thinks it's detecting")
    print("• Whether the segmentation masks make sense")
    print("• If the bbox and mask areas match")
    print("• Whether the detections are reasonable sizes")

if __name__ == "__main__":
    deep_investigation()
