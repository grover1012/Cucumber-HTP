#!/usr/bin/env python3
"""
Auto-Advance Label Review Tool
Shows images one by one with simple navigation
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random

class AutoAdvanceReviewer:
    def __init__(self, images_dir, labels_dir):
        """Initialize auto-advance reviewer."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        # Class names mapping
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
    
    def review_images_auto_advance(self, num_images=30):
        """Review images with auto-advance functionality."""
        # Get all label files
        label_files = list(self.labels_dir.glob("*.txt"))
        
        if not label_files:
            print("❌ No label files found")
            return
        
        # Select random samples
        sample_files = random.sample(label_files, min(num_images, len(label_files)))
        
        print(f"🎲 Reviewing {len(sample_files)} random samples")
        print("📱 Controls:")
        print("  - Press SPACE or ENTER to go to next image")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to restart from beginning")
        print("=" * 50)
        
        current_index = 0
        
        while current_index < len(sample_files):
            label_file = sample_files[current_index]
            image_name = label_file.stem
            
            print(f"\n🖼️ Image {current_index + 1}/{len(sample_files)}: {image_name}")
            
            # Find image file
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_file = None
            
            for ext in image_extensions:
                potential_file = self.images_dir / f"{image_name}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
            
            if not image_file:
                print(f"❌ Image {image_name} not found, skipping...")
                current_index += 1
                continue
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"❌ Could not load image {image_file}, skipping...")
                current_index += 1
                continue
            
            # Load labels
            labels = self.load_labels(label_file)
            
            print(f"🔍 Found {len(labels)} objects:")
            for i, label in enumerate(labels[:10]):  # Show first 10 objects
                print(f"  {i+1}. {label['class_name']} (ID: {label['class_id']})")
            if len(labels) > 10:
                print(f"  ... and {len(labels) - 10} more objects")
            
            # Draw labels on image
            labeled_image = self.draw_labels_on_image(image.copy(), labels)
            
            # Resize for display if too large
            height, width = labeled_image.shape[:2]
            if width > 1200 or height > 800:
                scale = min(1200/width, 800/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                labeled_image = cv2.resize(labeled_image, (new_width, new_height))
            
            # Add navigation info to image
            info_text = f"Image {current_index + 1}/{len(sample_files)}: {image_name} - {len(labels)} objects"
            cv2.putText(labeled_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(labeled_image, "SPACE/ENTER: Next | Q: Quit | R: Restart", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display image
            cv2.imshow(f'Review: {image_name}', labeled_image)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("👋 Review stopped by user")
                break
            elif key == ord('r') or key == ord('R'):
                print("🔄 Restarting from beginning...")
                current_index = 0
                continue
            elif key == ord(' ') or key == 13:  # SPACE or ENTER
                current_index += 1
            else:
                # Any other key also advances
                current_index += 1
        
        cv2.destroyAllWindows()
        print(f"\n✅ Review complete! Reviewed {min(current_index, len(sample_files))} images")

def main():
    """Main function for auto-advance review."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-advance label review tool")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", required=True, help="Directory containing label files")
    parser.add_argument("--num-images", type=int, default=30, help="Number of images to review")
    
    args = parser.parse_args()
    
    # Initialize reviewer
    reviewer = AutoAdvanceReviewer(args.images_dir, args.labels_dir)
    
    # Start auto-advance review
    reviewer.review_images_auto_advance(args.num_images)

if __name__ == "__main__":
    main()
