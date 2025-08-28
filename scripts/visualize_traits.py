#!/usr/bin/env python3
"""
Cucumber Trait Visualization
Shows images with extracted traits overlaid on each cucumber
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
import os

class CucumberTraitVisualizer:
    def __init__(self, yolo_model_path, csv_path, output_dir):
        """Initialize the visualizer."""
        self.yolo_model = YOLO(yolo_model_path)
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names from your YOLO model
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Color scheme for different traits
        self.colors = {
            'cucumber': (0, 255, 0),      # Green
            'slice': (255, 165, 0),        # Orange
            'bbox': (255, 0, 0),          # Red
            'text': (255, 255, 255),      # White
            'background': (0, 0, 0)       # Black
        }
        
        print(f"📊 Loaded {len(self.df)} cucumber samples")
        print(f"📁 Output directory: {self.output_dir}")
    
    def find_image_path(self, image_name):
        """Find the actual image file path."""
        # Try different possible image directories
        possible_dirs = [
            'data/new_annotations/train/images',
            'data/new_annotations/valid/images', 
            'data/new_annotations/test/images',
            'data/images'
        ]
        
        for dir_path in possible_dirs:
            if Path(dir_path).exists():
                # Try to find the image with full name
                img_path = Path(dir_path) / image_name
                if img_path.exists():
                    return str(img_path)
                
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Try exact match
                    img_path = Path(dir_path) / f"{image_name}{ext}"
                    if img_path.exists():
                        return str(img_path)
                    
                    # Try with .rf. extension
                    if '.rf.' in image_name:
                        base_name = image_name.split('.rf.')[0]
                        img_path = Path(dir_path) / f"{base_name}{ext}"
                        if img_path.exists():
                            return str(img_path)
                    
                    # Try partial match
                    for existing_file in Path(dir_path).glob(f"*{image_name}*"):
                        if existing_file.exists():
                            return str(existing_file)
        
        return None
    
    def create_trait_overlay(self, image, bbox, traits, class_name, confidence):
        """Create overlay with trait values on the image."""
        # Convert to PIL for better text handling
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        draw = ImageDraw.Draw(image_pil)
        
        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Extract bbox coordinates
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        box_color = self.colors[class_name] if class_name in self.colors else (255, 255, 255)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        # Create background for text
        text_lines = [
            f"Class: {class_name}",
            f"Confidence: {confidence:.3f}",
            f"Length: {traits.get('curved_length_cm', 'N/A'):.1f} cm",
            f"Diameter: {traits.get('diameter_cm', 'N/A'):.1f} cm",
            f"Shape Index: {traits.get('fruit_shape_index', 'N/A'):.2f}",
            f"Hollowness: {traits.get('hollowness_percentage', 'N/A'):.1f}%",
            f"Netting: {traits.get('netting_description', 'N/A')}",
            f"Spine Density: {traits.get('spine_density_per_cm2', 'N/A'):.1f}/cm²"
        ]
        
        # Calculate text box dimensions
        max_width = max(draw.textlength(line, font=font) for line in text_lines)
        text_height = len(text_lines) * 20
        
        # Position text box (avoid going off-screen)
        text_x = x2 + 10
        text_y = y1
        
        if text_x + max_width > image_pil.width:
            text_x = x1 - max_width - 10
        
        if text_y + text_height > image_pil.height:
            text_y = image_pil.height - text_height - 10
        
        # Draw background rectangle for text
        draw.rectangle([text_x, text_y, text_x + max_width + 10, text_y + text_height + 10], 
                      fill=self.colors['background'], outline=box_color, width=2)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = text_y + 5 + i * 20
            draw.text((text_x + 5, y_pos), line, fill=self.colors['text'], font=font)
        
        # Add trait values on the cucumber itself
        if traits.get('curved_length_cm'):
            # Draw length measurement line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Draw length text on the cucumber
            length_text = f"{traits['curved_length_cm']:.1f}cm"
            draw.text((mid_x - 20, mid_y - 10), length_text, 
                     fill=self.colors['text'], font=small_font)
        
        return np.array(image_pil)
    
    def visualize_single_image(self, image_path, output_name):
        """Visualize traits for a single image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Find cucumbers in this image
        image_name = Path(image_path).name
        # Try to match the full path or just the filename
        cucumbers_in_image = self.df[
            (self.df['image_path'].str.contains(image_name)) |
            (self.df['image_path'].str.contains(Path(image_name).stem))
        ]
        
        if len(cucumbers_in_image) == 0:
            print(f"⚠️ No cucumber data found for image: {image_name}")
            return None
        
        print(f"🔍 Found {len(cucumbers_in_image)} cucumbers in {image_name}")
        
        # Run YOLO detection to get bounding boxes
        results = self.yolo_model(image, verbose=False)
        detections = results[0]
        
        # Create visualization
        vis_image = image.copy()
        
        # Process each cucumber detection
        cucumber_count = 0
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Only process cucumbers and slices
            if class_name not in ['cucumber', 'slice']:
                continue
            
            # Find matching trait data (if available)
            if cucumber_count < len(cucumbers_in_image):
                cucumber = cucumbers_in_image.iloc[cucumber_count]
                
                # Extract traits
                traits = {
                    'curved_length_cm': cucumber.get('curved_length_cm', 0),
                    'diameter_cm': cucumber.get('diameter_cm', 0),
                    'fruit_shape_index': cucumber.get('fruit_shape_index', 0),
                    'hollowness_percentage': cucumber.get('hollowness_percentage', 0),
                    'netting_description': cucumber.get('netting_description', 'Unknown'),
                    'spine_density_per_cm2': cucumber.get('spine_density_per_cm2', 0)
                }
                
                # Create overlay
                vis_image = self.create_trait_overlay(
                    vis_image, bbox, traits, 
                    class_name, confidence
                )
                
                cucumber_count += 1
            else:
                # If no trait data available, just draw basic bbox
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                             self.colors[class_name], 2)
                cv2.putText(vis_image, f"{class_name} {confidence:.2f}", 
                           (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           self.colors['text'], 2)
        
        # Save visualization
        output_path = self.output_dir / f"{output_name}_traits.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Saved visualization: {output_path}")
        return vis_image
    
    def create_comparison_grid(self, image_paths, max_images=9):
        """Create a grid comparison of multiple images with traits."""
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(len(image_paths))))
        rows = int(np.ceil(len(image_paths) / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        fig.suptitle('Cucumber Trait Visualization - Multiple Images', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (img_path, output_name) in enumerate(image_paths):
            if i < len(image_paths):
                vis_image = self.visualize_single_image(img_path, output_name)
                if vis_image is not None:
                    # Convert BGR to RGB for matplotlib
                    if len(vis_image.shape) == 3:
                        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    else:
                        vis_image_rgb = vis_image
                    
                    axes[i].imshow(vis_image_rgb)
                    axes[i].set_title(f"{output_name}\n{Path(img_path).name}")
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f"Failed to load\n{Path(img_path).name}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        grid_path = self.output_dir / 'trait_comparison_grid.png'
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved comparison grid: {grid_path}")
    
    def visualize_sample_images(self, num_samples=6):
        """Visualize traits for a sample of images."""
        print(f"\n🎨 VISUALIZING TRAITS FOR {num_samples} SAMPLE IMAGES")
        print("=" * 60)
        
        # Get unique images
        unique_images = self.df['image_path'].unique()
        
        if len(unique_images) == 0:
            print("❌ No image data found")
            return
        
        # Select sample images
        sample_images = unique_images[:num_samples]
        
        # Process each sample image
        processed_images = []
        
        for i, image_path in enumerate(sample_images):
            print(f"\n🔄 Processing sample {i+1}/{len(sample_images)}: {Path(image_path).name}")
            
            # Find actual image file
            actual_image_path = self.find_image_path(Path(image_path).name)
            
            if actual_image_path:
                output_name = f"sample_{i+1:02d}"
                vis_image = self.visualize_single_image(actual_image_path, output_name)
                
                if vis_image is not None:
                    processed_images.append((actual_image_path, output_name))
                    print(f"   ✅ Successfully processed")
                else:
                    print(f"   ❌ Failed to process")
            else:
                print(f"   ⚠️ Image file not found")
        
        # Create comparison grid
        if processed_images:
            print(f"\n📊 Creating comparison grid for {len(processed_images)} images...")
            self.create_comparison_grid(processed_images)
        else:
            print("❌ No images were successfully processed")
    
    def visualize_specific_image(self, image_name):
        """Visualize traits for a specific image by name."""
        print(f"\n🎯 VISUALIZING TRAITS FOR SPECIFIC IMAGE: {image_name}")
        print("=" * 60)
        
        # Find the image in our data
        matching_data = self.df[self.df['image_path'].str.contains(image_name)]
        
        if len(matching_data) == 0:
            print(f"❌ No data found for image: {image_name}")
            return
        
        # Find actual image file
        actual_image_path = self.find_image_path(image_name)
        
        if actual_image_path:
            output_name = f"specific_{image_name.replace('.', '_')}"
            vis_image = self.visualize_single_image(actual_image_path, output_name)
            
            if vis_image is not None:
                print(f"✅ Successfully visualized: {image_name}")
                return vis_image
            else:
                print(f"❌ Failed to visualize: {image_name}")
        else:
            print(f"❌ Image file not found: {image_name}")
        
        return None
    
    def run_complete_visualization(self):
        """Run complete visualization pipeline."""
        print("🚀 STARTING COMPLETE CUCUMBER TRAIT VISUALIZATION")
        print("=" * 60)
        
        # Visualize sample images
        self.visualize_sample_images(num_samples=6)
        
        print("\n🎉 VISUALIZATION COMPLETE!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print("🖼️ Check the output directory for trait overlays and comparison grids")

def main():
    parser = argparse.ArgumentParser(description='Cucumber Trait Visualizer')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--csv-path', required=True, help='Path to the comprehensive traits CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory for visualizations')
    parser.add_argument('--specific-image', help='Visualize a specific image by name')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = CucumberTraitVisualizer(args.yolo_model, args.csv_path, args.output_dir)
    
    if args.specific_image:
        # Visualize specific image
        visualizer.visualize_specific_image(args.specific_image)
    else:
        # Run complete visualization
        visualizer.run_complete_visualization()

if __name__ == "__main__":
    main()
