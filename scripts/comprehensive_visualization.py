#!/usr/bin/env python3
"""
Comprehensive Cucumber Visualization
Shows original image, YOLO detection, SAM2 segmentation, and extracted traits
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os

class ComprehensiveVisualizer:
    def __init__(self, yolo_model_path, csv_path, output_dir):
        """Initialize the comprehensive visualizer."""
        self.yolo_model = YOLO(yolo_model_path)
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class names from your YOLO model
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber', 
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Color scheme
        self.colors = {
            'cucumber': (0, 255, 0),      # Green
            'slice': (255, 165, 0),        # Orange
            'ruler': (255, 0, 0),         # Red
            'color_chart': (0, 0, 255),   # Blue
            'text': (255, 255, 255),      # White
            'background': (0, 0, 0)       # Black
        }
        
        print(f"📊 Loaded {len(self.df)} cucumber samples")
        print(f"📁 Output directory: {self.output_dir}")
    
    def find_image_path(self, image_name):
        """Find the actual image file path."""
        possible_dirs = [
            'data/new_annotations/train/images',
            'data/new_annotations/valid/images', 
            'data/new_annotations/test/images',
            'data/images'
        ]
        
        for dir_path in possible_dirs:
            if Path(dir_path).exists():
                # Try exact match
                img_path = Path(dir_path) / image_name
                if img_path.exists():
                    return str(img_path)
                
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
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
    
    def create_detection_visualization(self, image, detections):
        """Create visualization showing YOLO detections."""
        vis_image = image.copy()
        
        for detection in detections.boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(vis_image, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
    
    def create_trait_overlay(self, image, bbox, traits, class_name, confidence):
        """Create overlay with trait values."""
        # Convert to PIL for better text handling
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
        
        draw = ImageDraw.Draw(image_pil)
        
        # Try to use a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Extract bbox coordinates
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        box_color = self.colors[class_name] if class_name in self.colors else (255, 255, 255)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        # Create trait text
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
        text_height = len(text_lines) * 18
        
        # Position text box
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
            y_pos = text_y + 5 + i * 18
            draw.text((text_x + 5, y_pos), line, fill=self.colors['text'], font=font)
        
        return np.array(image_pil)
    
    def create_comprehensive_visualization(self, image_path, output_name):
        """Create comprehensive visualization showing all aspects."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Find cucumbers in this image
        image_name = Path(image_path).name
        cucumbers_in_image = self.df[
            (self.df['image_path'].str.contains(image_name)) |
            (self.df['image_path'].str.contains(Path(image_name).stem))
        ]
        
        if len(cucumbers_in_image) == 0:
            print(f"⚠️ No cucumber data found for image: {image_name}")
            return None
        
        print(f"🔍 Found {len(cucumbers_in_image)} cucumbers in {image_name}")
        
        # Run YOLO detection
        results = self.yolo_model(image, verbose=False)
        detections = results[0]
        
        # Create different visualizations
        # 1. Original image
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. YOLO detection only
        detection_only = self.create_detection_visualization(image, detections)
        detection_only = cv2.cvtColor(detection_only, cv2.COLOR_BGR2RGB)
        
        # 3. Full trait visualization
        trait_vis = image.copy()
        cucumber_count = 0
        
        for i, detection in enumerate(detections.boxes):
            bbox = detection.xyxy[0].cpu().numpy()
            class_id = int(detection.cls[0])
            confidence = float(detection.conf[0])
            class_name = self.class_names[class_id]
            
            # Only process cucumbers and slices
            if class_name not in ['cucumber', 'slice']:
                continue
            
            # Find matching trait data
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
                
                # Create trait overlay
                trait_vis = self.create_trait_overlay(
                    trait_vis, bbox, traits, 
                    class_name, confidence
                )
                
                cucumber_count += 1
        
        trait_vis = cv2.cvtColor(trait_vis, cv2.COLOR_BGR2RGB)
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'Comprehensive Cucumber Analysis: {image_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot 2: YOLO detection
        axes[0, 1].imshow(detection_only)
        axes[0, 1].set_title('2. YOLO Object Detection', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Plot 3: Trait visualization
        axes[1, 0].imshow(trait_vis)
        axes[1, 0].set_title('3. Extracted Traits Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Plot 4: Trait data table
        axes[1, 1].axis('off')
        
        # Create trait summary table
        if len(cucumbers_in_image) > 0:
            table_data = []
            for idx, cucumber in cucumbers_in_image.iterrows():
                table_data.append([
                    f"Cucumber {idx+1}",
                    cucumber['class'],
                    f"{cucumber.get('curved_length_cm', 0):.1f} cm",
                    f"{cucumber.get('diameter_cm', 0):.1f} cm",
                    f"{cucumber.get('fruit_shape_index', 0):.2f}",
                    cucumber.get('netting_description', 'N/A')
                ])
            
            table = axes[1, 1].table(
                cellText=table_data,
                colLabels=['ID', 'Class', 'Length', 'Diameter', 'Shape Index', 'Netting'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_data) + 1):
                for j in range(6):
                    if i == 0:  # Header row
                        table[(i, j)].set_facecolor('#4CAF50')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    else:
                        table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            axes[1, 1].set_title('4. Extracted Trait Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        output_path = self.output_dir / f"{output_name}_comprehensive.jpg"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved comprehensive visualization: {output_path}")
        return trait_vis
    
    def visualize_sample_images(self, num_samples=3):
        """Create comprehensive visualizations for sample images."""
        print(f"\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS FOR {num_samples} SAMPLE IMAGES")
        print("=" * 70)
        
        # Get unique images
        unique_images = self.df['image_path'].unique()
        
        if len(unique_images) == 0:
            print("❌ No image data found")
            return
        
        # Select sample images
        sample_images = unique_images[:num_samples]
        
        for i, image_path in enumerate(sample_images):
            print(f"\n🔄 Processing sample {i+1}/{len(sample_images)}: {Path(image_path).name}")
            
            # Find actual image file
            actual_image_path = self.find_image_path(Path(image_path).name)
            
            if actual_image_path:
                output_name = f"comprehensive_sample_{i+1:02d}"
                vis_image = self.create_comprehensive_visualization(actual_image_path, output_name)
                
                if vis_image is not None:
                    print(f"   ✅ Successfully processed")
                else:
                    print(f"   ❌ Failed to process")
            else:
                print(f"   ⚠️ Image file not found")
        
        print(f"\n🎉 COMPREHENSIVE VISUALIZATION COMPLETE!")
        print(f"📁 All results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Cucumber Visualization')
    parser.add_argument('--yolo-model', required=True, help='Path to YOLO model')
    parser.add_argument('--csv-path', required=True, help='Path to the comprehensive traits CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of sample images to process')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ComprehensiveVisualizer(args.yolo_model, args.csv_path, args.output_dir)
    
    # Create comprehensive visualizations
    visualizer.visualize_sample_images(args.num_samples)

if __name__ == "__main__":
    main()
