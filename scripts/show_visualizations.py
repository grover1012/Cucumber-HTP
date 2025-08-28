#!/usr/bin/env python3
"""
Show Cucumber Trait Visualizations One at a Time
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse
import time

def show_visualizations(visualization_dir):
    """Show each visualization image one at a time."""
    vis_dir = Path(visualization_dir)
    
    # Get all visualization images
    image_files = list(vis_dir.glob('*_traits.jpg'))
    
    if not image_files:
        print("❌ No visualization images found!")
        return
    
    print(f"🎨 Found {len(image_files)} visualization images")
    print("🖼️ Press any key to move to next image, 'q' to quit")
    
    for i, img_path in enumerate(image_files):
        print(f"\n📊 Showing image {i+1}/{len(image_files)}: {img_path.name}")
        print("=" * 60)
        
        # Load and display image
        img = mpimg.imread(img_path)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.title(f"Cucumber Traits Visualization: {img_path.name}", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add information about what to look for
        info_text = """
        Look for:
        • Green/Orange bounding boxes around cucumbers
        • Trait measurements displayed next to each cucumber
        • Length, diameter, shape index, netting, etc.
        • Confidence scores and class labels
        """
        plt.figtext(0.02, 0.02, info_text, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Wait for user input
        user_input = input(f"\nPress Enter for next image, 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        
        plt.close()
    
    print("\n🎉 Visualization complete!")

def main():
    parser = argparse.ArgumentParser(description='Show Cucumber Trait Visualizations')
    parser.add_argument('--visualization-dir', default='results/trait_visualizations', 
                       help='Directory containing visualization images')
    
    args = parser.parse_args()
    
    show_visualizations(args.visualization_dir)

if __name__ == "__main__":
    main()
