#!/usr/bin/env python3
"""
Check empty annotation images to see if they should have annotations
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

def check_empty_image_samples():
    print("🔍 CHECKING EMPTY ANNOTATION IMAGES")
    print("=" * 60)
    
    train_dir = Path("data/sam2_annotations/train")
    
    if not train_dir.exists():
        print("❌ Training directory not found!")
        return
    
    # Get some empty annotation files
    empty_files = []
    for json_file in train_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            annotations = data.get('annotations', [])
            if len(annotations) == 0:
                image_info = data.get('image', {})
                image_name = image_info.get('file_name', 'unknown')
                empty_files.append((json_file, image_name))
        except:
            continue
    
    print(f"📸 Found {len(empty_files)} files with 0 annotations")
    print(f"🔍 Checking first 5 empty files...")
    
    # Check first 5 empty files
    for i, (json_file, image_name) in enumerate(empty_files[:5]):
        print(f"\n📁 Empty file {i+1}: {json_file.name}")
        print(f"  🖼️  Image: {image_name}")
        
        # Check if image exists
        image_file = train_dir / image_name
        if image_file.exists():
            print(f"  ✅ Image file exists")
            
            # Load and analyze image
            img = cv2.imread(str(image_file))
            if img is not None:
                height, width = img.shape[:2]
                print(f"  📐 Image dimensions: {width}x{height}")
                
                # Check image content
                print(f"  🔍 Image analysis:")
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Check if image is mostly empty/white
                mean_brightness = np.mean(gray)
                print(f"    • Mean brightness: {mean_brightness:.1f}")
                
                # Check if image has significant content (not just white/empty)
                if mean_brightness > 200:  # Very bright/white
                    print(f"    ⚠️  Image appears very bright/white - may be empty")
                elif mean_brightness < 100:  # Very dark
                    print(f"    ⚠️  Image appears very dark - may be unusable")
                else:
                    print(f"    ✅ Image has reasonable brightness - should have content")
                
                # Check for edges (indicates content)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (width * height)
                print(f"    • Edge density: {edge_density:.4f}")
                
                if edge_density < 0.001:
                    print(f"    ⚠️  Very low edge density - image may be empty/blank")
                elif edge_density < 0.01:
                    print(f"    ⚠️  Low edge density - image may have minimal content")
                else:
                    print(f"    ✅ Good edge density - image should have content")
                
                # Create visualization
                print(f"  🎨 Creating visualization...")
                create_image_analysis_visualization(img, f"empty_check_{image_name}")
                print(f"  ✅ Visualization saved")
                
            else:
                print(f"  ❌ Failed to load image")
        else:
            print(f"  ❌ Image file not found")
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 40)
    print("Empty annotation files could be due to:")
    print("  1. Images that are actually empty/blank")
    print("  2. Images that weren't annotated in Roboflow")
    print("  3. Export issues from Roboflow")
    print("  4. Images that were rejected during annotation")
    
    print(f"\n💡 RECOMMENDATION:")
    print("  • Check the generated visualizations")
    print("  • If images have content but no annotations, re-export from Roboflow")
    print("  • If images are empty/blank, they can be safely excluded")
    print("  • Current dataset has 797 annotations which should be sufficient for training")

def create_image_analysis_visualization(img, filename):
    """Create visualization showing the image and its analysis"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Original image
    ax1.imshow(img_rgb)
    ax1.set_title(f"Original Image\n{filename}")
    ax1.axis('off')
    
    # Grayscale image
    ax2.imshow(gray, cmap='gray')
    ax2.set_title("Grayscale Image")
    ax2.axis('off')
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    ax3.imshow(edges, cmap='gray')
    ax3.set_title("Edge Detection (Canny)")
    ax3.axis('off')
    
    # Histogram
    ax4.hist(gray.ravel(), bins=256, range=[0, 256], alpha=0.7, color='blue')
    ax4.set_title("Pixel Intensity Histogram")
    ax4.set_xlabel("Pixel Value")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    ax4.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    ax4.axvline(mean_val + std_val, color='orange', linestyle='--', label=f'Mean+Std: {mean_val + std_val:.1f}')
    ax4.axvline(mean_val - std_val, color='orange', linestyle='--', label=f'Mean-Std: {mean_val - std_val:.1f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f"empty_image_analysis_{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    check_empty_image_samples()
