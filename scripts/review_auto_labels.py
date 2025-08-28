#!/usr/bin/env python3
"""
Visual Review Tool for Auto-Generated Labels
Shows bounding boxes and class names on images for easy review
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import random

class LabelReviewer:
    def __init__(self, images_dir, labels_dir):
        """Initialize label reviewer."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        # Class names mapping (from your dataset)
        self.class_names = [
            'big_ruler', 'blue_dot', 'cavity', 'color_chart', 'cucumber',
            'green_dot', 'hollow', 'label', 'objects', 'red_dot', 'ruler', 'slice'
        ]
        
        # Colors for different classes
        self.class_colors = [
            (255, 0, 0),    # Blue for big_ruler
            (0, 0, 255),    # Red for blue_dot
            (128, 0, 128),  # Purple for cavity
            (255, 255, 0),  # Cyan for color_chart
            (0, 255, 0),    # Green for cucumber
            (0, 255, 0),    # Green for green_dot
            (128, 128, 0),  # Olive for hollow
            (0, 0, 255),    # Red for label
            (255, 165, 0),  # Orange for objects
            (255, 0, 0),    # Blue for red_dot
            (0, 255, 255),  # Yellow for ruler
            (255, 192, 203) # Pink for slice
        ]
    
    def load_labels(self, label_file):
        """Load YOLO format labels from file."""
        labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        labels.append({
                            'class_id': class_id,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        return labels
    
    def draw_labels_on_image(self, image, labels):
        """Draw bounding boxes and labels on image."""
        img_height, img_width = image.shape[:2]
        
        for label in labels:
            # Convert normalized coordinates to pixel coordinates
            x_center = int(label['x_center'] * img_width)
            y_center = int(label['y_center'] * img_height)
            width = int(label['width'] * img_width)
            height = int(label['height'] * img_height)
            
            # Calculate bounding box corners
            x1 = x_center - width // 2
            y1 = y_center - height // 2
            x2 = x_center + width // 2
            y2 = y_center + height // 2
            
            # Get color for this class
            color = self.class_colors[label['class_id']] if label['class_id'] < len(self.class_colors) else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"{label['class_name']}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def review_single_image(self, image_name):
        """Review a single image with its labels."""
        # Find image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_file = None
        
        for ext in image_extensions:
            potential_file = self.images_dir / f"{image_name}{ext}"
            if potential_file.exists():
                image_file = potential_file
                break
        
        if not image_file:
            print(f"❌ Image {image_name} not found")
            return
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Could not load image {image_file}")
            return
        
        # Load labels
        label_file = self.labels_dir / f"{image_name}.txt"
        labels = self.load_labels(label_file)
        
        print(f"📸 Reviewing: {image_name}")
        print(f"🔍 Found {len(labels)} objects:")
        
        for i, label in enumerate(labels):
            print(f"  {i+1}. {label['class_name']} (ID: {label['class_id']})")
        
        # Draw labels on image
        labeled_image = self.draw_labels_on_image(image.copy(), labels)
        
        # Resize for display if too large
        height, width = labeled_image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            labeled_image = cv2.resize(labeled_image, (new_width, new_height))
        
        # Display image
        cv2.imshow(f'Review: {image_name}', labeled_image)
        print(f"🖼️ Image displayed. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def review_random_samples(self, num_samples=10):
        """Review random sample images."""
        # Get all label files
        label_files = list(self.labels_dir.glob("*.txt"))
        
        if not label_files:
            print("❌ No label files found")
            return
        
        # Select random samples
        sample_files = random.sample(label_files, min(num_samples, len(label_files)))
        
        print(f"🎲 Reviewing {len(sample_files)} random samples:")
        
        for i, label_file in enumerate(sample_files):
            image_name = label_file.stem
            print(f"\n{i+1}/{len(sample_files)}: {image_name}")
            
            self.review_single_image(image_name)
            
            # Ask if user wants to continue
            if i < len(sample_files) - 1:
                response = input("Continue to next image? (y/n): ").lower()
                if response != 'y':
                    break
    
    def batch_review(self, start_index=0, num_images=20):
        """Review a batch of images sequentially."""
        # Get all label files
        label_files = sorted(list(self.labels_dir.glob("*.txt")))
        
        if not label_files:
            print("❌ No label files found")
            return
        
        end_index = min(start_index + num_images, len(label_files))
        
        print(f"📋 Reviewing images {start_index+1} to {end_index} of {len(label_files)}")
        
        for i in range(start_index, end_index):
            label_file = label_files[i]
            image_name = label_file.stem
            
            print(f"\n{i+1}/{len(label_files)}: {image_name}")
            
            self.review_single_image(image_name)
            
            # Ask if user wants to continue
            if i < end_index - 1:
                response = input("Continue to next image? (y/n/q for quit): ").lower()
                if response == 'q':
                    break
                elif response != 'y':
                    continue

def main():
    """Main function for label review."""
    parser = argparse.ArgumentParser(description="Review auto-generated labels visually")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", required=True, help="Directory containing label files")
    parser.add_argument("--mode", choices=['single', 'random', 'batch'], default='random',
                       help="Review mode: single image, random samples, or batch")
    parser.add_argument("--image-name", help="Image name to review (for single mode)")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to review")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for batch review")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for review")
    
    args = parser.parse_args()
    
    # Initialize reviewer
    reviewer = LabelReviewer(args.images_dir, args.labels_dir)
    
    if args.mode == 'single':
        if not args.image_name:
            print("❌ Please provide --image-name for single mode")
            return
        reviewer.review_single_image(args.image_name)
    
    elif args.mode == 'random':
        reviewer.review_random_samples(args.num_samples)
    
    elif args.mode == 'batch':
        reviewer.batch_review(args.start_index, args.batch_size)

if __name__ == "__main__":
    main()
